# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances, BitMasks, ROIMasks
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
import detectron2.utils.comm as comm

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads

from ..text_encoder.text_encoder import build_text_encoder


@META_ARCH_REGISTRY.register()
class FastRCNN(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        ret = {
            "backbone": backbone,
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }
        return ret

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            for inst in gt_instances:
                inst._pos_category_ids = inst.gt_classes.unique()
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        assert "proposals" in batched_inputs[0]
        proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        assert "proposals" in batched_inputs[0]
        proposals = [x["proposals"].to(self.device) for x in batched_inputs]
        results, _ = self.roi_heads(images, features, proposals, None)

        if self.vis_period > 0:
            self.vis_results(batched_inputs, results)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            results = FastRCNN._postprocess(results, batched_inputs, images.image_sizes)

        return results


    def vis_results(self, batched_inputs, results):
        import os
        import cv2
        import time
        from colormap import colormap
        color_maps = colormap()
        from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
        os.makedirs('output_fast_rcnn', exist_ok=True)

        for image_id in range(len(batched_inputs)):
            image = batched_inputs[image_id]["image"].permute(1,2,0)[:,:,[2,1,0]].numpy()
            boxes = results[image_id].pred_boxes.tensor.cpu().numpy()
            scores = results[image_id].scores.cpu().numpy()
            categories = results[image_id].pred_classes.cpu().numpy()
            image_show = image.copy()

            for instance_id, (bbox, score, category) in enumerate(zip(boxes, scores, categories)):
                if score < 0.1:
                    continue
                instance_color = color_maps[instance_id % len(color_maps)]
                image_show = cv2.rectangle(image_show, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), instance_color.tolist(), 2)
                cv2.putText(image_show, "{}-{:.2f}".format(COCO_CATEGORIES[category]["name"], score), (int(bbox[0]-5), int(bbox[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, instance_color.tolist(), 2)
            cv2.imwrite('output_fast_rcnn/show_{}.png'.format(time.time()), image_show)


    def vis_proposals(self, batched_inputs, proposals):
        import os
        import cv2
        from colormap import colormap
        color_maps = colormap()
        os.makedirs('output_vis_proposals', exist_ok=True)

        for image_id in range(len(batched_inputs)):
            image = batched_inputs[image_id]["image"].permute(1,2,0)[:,:,[2,1,0]].numpy()
            boxes = proposals[image_id].proposal_boxes.tensor.cpu().numpy()
            scores = proposals[image_id].objectness_logits.cpu().numpy()
            image_show = image.copy()

            for instance_id, (bbox, score) in enumerate(zip(boxes[:10], scores[:10])):
                instance_color = color_maps[instance_id % len(color_maps)]
                image_show = cv2.rectangle(image_show, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), instance_color.tolist(), 2)
                cv2.putText(image_show, "{:.2f}".format(score), (int(bbox[0]-5), int(bbox[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, instance_color.tolist(), 2)
            cv2.imwrite('output_vis_proposals/show_{}.png'.format(image_id), image_show)


    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images


    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
