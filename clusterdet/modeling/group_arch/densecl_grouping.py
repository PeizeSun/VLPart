# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from skimage import color

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, ROIMasks, pairwise_iou

from ..utils.adelaidet import matrix_nms, center_of_mass, unfold_wo_center


@META_ARCH_REGISTRY.register()
class DenseCL_Grouping(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        scale_factors: list = (1.0, 0.5, 0.25),
        box_to_image_ratio_thresh: float = 0.9,
        box_score_thresh: float = 0.7,
        box_topk: int = 100,
    ):
        super().__init__()
        self.backbone = backbone

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.scale_factors = scale_factors
        self.box_to_image_ratio_thresh = box_to_image_ratio_thresh
        self.box_score_thresh = box_score_thresh
        self.box_topk = box_topk

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        ret = {
            "backbone": backbone,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }
        ret.update({
            "scale_factors": cfg.MODEL.GROUP_HEADS.SCALE_FACTORS,
            "box_to_image_ratio_thresh": cfg.MODEL.GROUP_HEADS.BOX_TO_IMAGE_RATIO_THRESH,
            "box_score_thresh": cfg.MODEL.GROUP_HEADS.BOX_SCORE_THRESH,
            "box_topk": cfg.MODEL.GROUP_HEADS.BOX_TOPK,
        })

        return ret

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs, do_postprocess=True):
        assert not self.training, 'DenseCL_Grouping only supports inference mode'
        assert len(batched_inputs) == 1, "DenseCL_Grouping only supports one image in one batch"

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        grouping_proposals = self.attention_to_boxes(
            batched_inputs,
            features,
            images.image_sizes,
        )

        results = []
        for image_idx in range(len(batched_inputs)):
            result = Instances(images.image_sizes[image_idx])
            boxes = grouping_proposals[image_idx].proposal_boxes.tensor
            scores = grouping_proposals[image_idx].scores

            result.pred_boxes = Boxes(boxes)
            result.pred_classes = torch.zeros_like(scores, dtype=torch.int64)
            result.scores = scores.float()
            results.append(result)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            max_shape = images.tensor.shape[2:]
            return DenseCL_Grouping._postprocess(results, batched_inputs, images.image_sizes, max_shape)
        else:
            return results


    def attention_to_boxes(self, batched_inputs, features, image_sizes):
        predictions = features['res5']
        stride = 32

        pseudo_targets = []
        for image_idx in range(len(predictions)):
            pseudo_target = Instances(image_sizes[image_idx])
            height, width = image_sizes[image_idx]

            keys = predictions[image_idx]
            kh, kw = keys.shape[-2:]
            queries_list = []
            for scale_factor in self.scale_factors:
                cur_queries = F.interpolate(
                    keys[None, ...], size=(int(kh*scale_factor), int(kw*scale_factor)),
                    mode='bilinear', align_corners=False)[0]
                queries_list.append(cur_queries.reshape(keys.shape[0], -1).permute(1, 0))
            queries = torch.cat(queries_list)
            _, H, W = keys.shape
            keys = keys / keys.norm(dim=0, keepdim=True)
            queries = queries / queries.norm(dim=1, keepdim=True)
            attn = queries @ keys.reshape(keys.shape[0], -1)
            # normalize
            attn -= attn.min(-1, keepdim=True)[0]
            attn /= attn.max(-1, keepdim=True)[0]
            attn = attn.reshape(attn.shape[0], H, W)

            soft_masks = attn
            masks = soft_masks >= 0.5

            sum_masks = masks.sum((1, 2))
            keep = sum_masks > 1
            if keep.sum() == 0:
                pseudo_target.proposal_boxes = Boxes(keep.new_zeros((0, 4)))
                pseudo_target.scores = keep.new_ones((0,))
                pseudo_targets.append(pseudo_target)
                continue

            masks = masks[keep]
            soft_masks = soft_masks[keep]
            sum_masks = sum_masks[keep]

            # Matrix NMS
            maskness = (soft_masks * masks.float()).sum((1, 2)) / sum_masks
            sort_inds = torch.argsort(maskness, descending=True)
            maskness = maskness[sort_inds]
            masks = masks[sort_inds]
            sum_masks = sum_masks[sort_inds]
            soft_masks = soft_masks[sort_inds]
            maskness = matrix_nms(maskness * 0, masks, sum_masks, maskness, sigma=2, kernel='gaussian')

            sort_inds = torch.argsort(maskness, descending=True)
            if len(sort_inds) > self.box_topk:
                sort_inds = sort_inds[:self.box_topk]
            maskness = maskness[sort_inds]
            soft_masks = soft_masks[sort_inds]

            soft_masks = \
            F.interpolate(soft_masks[None, ...], size=(height, width), mode='bilinear', align_corners=False)[0]
            masks = (soft_masks >= 0.5).float()

            # mask to box
            width_proj = masks.max(1)[0]
            height_proj = masks.max(2)[0]
            box_width, box_height = width_proj.sum(1), height_proj.sum(1)
            center_ws, _ = center_of_mass(width_proj[:, None, :])
            _, center_hs = center_of_mass(height_proj[:, :, None])
            boxes = torch.stack([center_ws - 0.5 * box_width, center_hs - 0.5 * box_height, center_ws + 0.5 * box_width,
                                 center_hs + 0.5 * box_height], 1)

            keep_1 = (boxes[:, 2] - boxes[:, 0]) < self.box_to_image_ratio_thresh * width
            keep_2 = maskness >= self.box_score_thresh
            keep = keep_1 & keep_2

            if keep.sum() == 0:
                pseudo_target.proposal_boxes = Boxes(keep.new_zeros((0, 4)))
                pseudo_target.scores = keep.new_ones((0,))
                pseudo_targets.append(pseudo_target)
                continue

            maskness = maskness[keep]
            boxes = boxes[keep]

            pseudo_target.proposal_boxes = Boxes(boxes)
            pseudo_target.scores = maskness
            pseudo_targets.append(pseudo_target)

        return pseudo_targets


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

    return results
