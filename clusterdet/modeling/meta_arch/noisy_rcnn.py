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
from ..utils.detic import load_class_freq, get_fed_loss_inds
from ..roi_heads.roi_heads_joint import build_recognition_heads


@META_ARCH_REGISTRY.register()
class NoisyRCNN(nn.Module):
    """
    Support Generalized R-CNN to four settings:
    (1) fully-supervised
    (2) weakly-supervised
    (3) open_world
    (4) open_vocabulary
    """
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        recognition_heads: nn.Module,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        seperate_backbone=False,
        with_image_labels=False,
        sync_caption_batch=False,
        roi_head_name='',
        cap_batch_ratio=4,
        with_caption=False,
        dynamic_classifier=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.seperate_backbone = seperate_backbone
        self.recognition_heads = recognition_heads
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.with_image_labels = with_image_labels
        self.with_caption = with_caption
        self.sync_caption_batch = sync_caption_batch
        self.roi_head_name = roi_head_name
        self.cap_batch_ratio = cap_batch_ratio
        self.dynamic_classifier = dynamic_classifier
        if self.dynamic_classifier:
            self.freq_weight = kwargs.pop('freq_weight')
            self.num_classes = kwargs.pop('num_classes')
            self.num_sample_cats = kwargs.pop('num_sample_cats')

        if self.with_caption:
            assert not self.dynamic_classifier
            self.text_encoder = build_text_encoder(pretrain=True)
            for v in self.text_encoder.parameters():
                v.requires_grad = False

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
            "recognition_heads": build_recognition_heads(cfg, backbone),
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

        ret.update({
            'seperate_backbone': cfg.MODEL.SEPERATE_BACKBONE,
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'with_caption': cfg.MODEL.WITH_CAPTION,
            'sync_caption_batch': cfg.MODEL.SYNC_CAPTION_BATCH,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'roi_head_name': cfg.MODEL.ROI_HEADS.NAME,
            'cap_batch_ratio': cfg.MODEL.CAP_BATCH_RATIO,
        })
        if ret['dynamic_classifier']:
            ret['freq_weight'] = load_class_freq(
                cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
                cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT)
            ret['num_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            ret['num_sample_cats'] = cfg.MODEL.NUM_SAMPLE_CATS

        return ret

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        """
        if not self.training:
            return self.inference(batched_inputs)

        # annotation
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        for inst, x in zip(gt_instances, batched_inputs):
            inst._ann_type = x['ann_type'] if 'ann_type' in x else 'clean'
            inst._pos_category_ids = x['pos_category_ids'] if 'pos_category_ids' in x else \
                inst.gt_classes.unique()
        ann_types = [inst._ann_type for inst in gt_instances]
        assert len(set(ann_types)) == 1
        ann_type = ann_types[0]

        # image
        images = self.preprocess_image(batched_inputs)

        # backbone
        if self.seperate_backbone:
            backbone_features = self.backbone.forward_backbone(images.tensor)
            features = self.backbone(images.tensor, backbone_features)
        else:
            features = self.backbone(images.tensor)
            backbone_features = features

        # mine instances
        image_losses, mining_instances = self.recognition_heads(
            backbone_features, gt_instances, None, ann_type,
        )
        # self.vis_mining_instances(batched_inputs, mining_instances)

        # region proposal
        proposals, proposal_losses = self.proposal_generator(images, features, mining_instances)

        # roi head
        cls_features, cls_inds, caption_features = None, None, None
        if self.with_caption and 'caption' in ann_type:
            inds = [torch.randint(len(x['captions']), (1,))[0].item() \
                    for x in batched_inputs]
            caps = [x['captions'][ind] for ind, x in zip(inds, batched_inputs)]
            caption_features = self.text_encoder(caps).float()

        if self.sync_caption_batch:
            caption_features = self._sync_caption_features(
                caption_features, ann_type, len(batched_inputs))

        if self.dynamic_classifier and ann_type != 'caption':
            cls_inds = self._sample_cls_inds(gt_instances, ann_type)  # inds, inv_inds
            ind_with_bg = cls_inds[0].tolist() + [-1]
            cls_features = self.roi_heads.box_predictor[0].cls_score.zs_weight[:, ind_with_bg].permute(1,
                                                                                                       0).contiguous()
        classifier_info = cls_features, cls_inds, caption_features

        proposals, detector_losses = self.roi_heads(
            images, features, proposals, mining_instances, gt_instances,
            ann_type=ann_type, classifier_info=classifier_info,
        )

        # record losses
        losses = {}
        losses.update(image_losses)
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # # record losses
        # losses = {}
        # losses.update(image_losses)
        # losses.update({k: v * 0 for k, v in detector_losses.items()})
        # losses.update({k: v * 0 for k, v in proposal_losses.items()})
        return losses

    def _sync_caption_features(self, caption_features, ann_type, BS):
        has_caption_feature = (caption_features is not None)
        BS = (BS * self.cap_batch_ratio) if (ann_type == 'box') else BS
        rank = torch.full(
            (BS, 1), comm.get_rank(), dtype=torch.float32,
            device=self.device)
        if not has_caption_feature:
            caption_features = rank.new_zeros((BS, 512))
        caption_features = torch.cat([caption_features, rank], dim=1)
        global_caption_features = comm.all_gather(caption_features)
        caption_features = torch.cat(
            [x.to(self.device) for x in global_caption_features], dim=0) \
            if has_caption_feature else None  # (NB) x (D + 1)
        return caption_features

    def _sample_cls_inds(self, gt_instances, ann_type='box'):
        if ann_type == 'box':
            gt_classes = torch.cat(
                [x.gt_classes for x in gt_instances])
            C = len(self.freq_weight)
            freq_weight = self.freq_weight
        else:
            gt_classes = torch.cat(
                [torch.tensor(
                    x._pos_category_ids,
                    dtype=torch.long, device=x.gt_classes.device) \
                    for x in gt_instances])
            C = self.num_classes
            freq_weight = None
        assert gt_classes.max() < C, '{} {}'.format(gt_classes.max(), C)
        inds = get_fed_loss_inds(
            gt_classes, self.num_sample_cats, C,
            weight=freq_weight)
        cls_id_map = gt_classes.new_full(
            (self.num_classes + 1,), len(inds))
        cls_id_map[inds] = torch.arange(len(inds), device=cls_id_map.device)
        return inds, cls_id_map

    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            max_shape = images.tensor.shape[2:]
            return NoisyRCNN._postprocess(results, batched_inputs, images.image_sizes, max_shape)
        else:
            return results

    def vis_mining_instances(self, batched_inputs, results):
        import os
        import cv2
        import time
        from colormap import colormap
        color_maps = colormap()
        from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
        os.makedirs('output_minining_instances', exist_ok=True)

        for image_id in range(len(batched_inputs)):
            image = batched_inputs[image_id]["image"].permute(1,2,0)[:,:,[2,1,0]].numpy()
            boxes = results[image_id].gt_boxes.tensor.cpu().numpy()
            scores = results[image_id].gt_loss_weights.cpu().numpy()
            categories = results[image_id].gt_classes.cpu().numpy()
            image_show = image.copy()

            for instance_id, (bbox, score, category) in enumerate(zip(boxes, scores, categories)):
                instance_color = color_maps[instance_id % len(color_maps)]
                image_show = cv2.rectangle(image_show, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), instance_color.tolist(), 2)
                cv2.putText(image_show, "{}-{:.2f}".format(COCO_CATEGORIES[category]["name"], score), (int(bbox[0]-5), int(bbox[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, instance_color.tolist(), 2)
            cv2.imwrite('output_minining_instances/show_{}.png'.format(time.time()), image_show)


    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        original_images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in original_images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes, max_shape):
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
            r = custom_detector_postprocess(results_per_image, height, width, max_shape)
            processed_results.append({"instances": r})
        return processed_results


def custom_detector_postprocess(
        results: Instances, output_height: int, output_width: int,
        max_shape, mask_threshold: float = 0.5
):
    """
    detector_postprocess with support on global_masks
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )

    resized_h, resized_w = results.image_size
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_global_masks"):
        mask_pred_per_image = F.interpolate(results.pred_global_masks.unsqueeze(1), size=max_shape, mode="bilinear",
                                            align_corners=False)
        mask_pred_per_image = mask_pred_per_image[:, :, :resized_h, :resized_w]
        mask_pred_per_image = F.interpolate(mask_pred_per_image, size=new_size, mode='bilinear',
                                            align_corners=False).squeeze(1)
        results.pred_masks = mask_pred_per_image > mask_threshold

    elif results.has("pred_masks"):
        if isinstance(results.pred_masks, ROIMasks):
            roi_masks = results.pred_masks
        else:
            # pred_masks is a tensor of shape (N, 1, M, M)
            roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        results.pred_masks = roi_masks.to_bitmasks(
            results.pred_boxes, output_height, output_width, mask_threshold
        ).tensor  # TODO return ROIMasks/BitMask object in the future

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results
