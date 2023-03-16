# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from fvcore.nn import sigmoid_focal_loss_jit, giou_loss, smooth_l1_loss
import fvcore.nn.weight_init as weight_init

from detectron2.config import configurable
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
import detectron2.utils.comm as comm

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers, 
    fast_rcnn_inference,
    _log_classification_stats,
)

class OLNFastRCNNOutputLayers(nn.Module):
    """
    """
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        official_implement: bool = False,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        # self.cls_score = nn.Linear(input_size, num_classes + 1)
        self.bbox_score = nn.Linear(input_size, 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        # nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        # for l in [self.cls_score, self.bbox_pred]:
        for l in [self.bbox_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight

        self.official_implement = official_implement
        
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"               : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg"     : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"            : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"         : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"           : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"       : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"         : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"               : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},  # noqa
            "official_implement"        : cfg.OPENWORLD.OLN_OFFICIAL,
            # fmt: on
        }

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        
        # scores = self.cls_score(x)        
        proposal_deltas = self.bbox_pred(x)
        box_ious = self.bbox_score(x)
        return None, proposal_deltas, box_ious
    
    def losses(self, predictions, proposals):
        _, proposal_deltas, box_ious = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        # _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)


        loss_box_score = self.custom_cross_entropy_open_world(box_ious, gt_classes, proposal_boxes, gt_boxes)

        losses = {
            "loss_box_score": loss_box_score, 
            # "loss_cls": scores.sum() * 0,
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
    
    
    def custom_cross_entropy_open_world(self, box_ious, gt_classes, proposal_boxes, gt_boxes, reduction='mean'):
        if gt_classes.numel() == 0 and reduction == "mean":
            return box_ious.sum() * 0.0  # connect the gradient

        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        
        fg_iou = torch.diag(pairwise_iou(Boxes(proposal_boxes[fg_inds]), Boxes(gt_boxes[fg_inds])))
        loss = smooth_l1_loss(box_ious[fg_inds].sigmoid().view(-1), fg_iou, beta=0.0, reduction="sum",)
        
        loss_box_score = loss / max(gt_classes.numel(), 1.0)
        
        return loss_box_score
    
    
    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        loss_box_reg = _dense_box_regression_loss(
            [proposal_boxes[fg_inds]],
            self.box2box_transform,
            [fg_pred_deltas.unsqueeze(0)],
            [gt_boxes[fg_inds]],
            ...,
            self.box_reg_loss_type,
            self.smooth_l1_beta,
        )

        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty
    
    
    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances],
                  branch="supervised"):
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals, branch)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )    
    
    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances],
        branch="supervised",
    ):
        scores, _, box_ious = predictions
        num_prop_per_image = [len(p) for p in proposals]
        
        rcnn_iou = box_ious.sigmoid()
        rpn_ctrness = torch.cat([p.objectness_logits for p in proposals]).sigmoid().unsqueeze(1)
        
        if branch == "teacher":
            probs = torch.sqrt((rpn_ctrness * rcnn_iou + rcnn_iou) / 2)
        else:
            probs = torch.sqrt(rpn_ctrness * rcnn_iou)
           
        # Concat dummy zero-scores for the background class.
        probs = torch.cat([probs, torch.zeros_like(probs)], dim=-1)
        
        return probs.split(num_prop_per_image, dim=0)
    
    
    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        if not len(proposals):
            return []
        _, proposal_deltas, _ = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
                    
        if self.official_implement:
            predict_boxes = self.delta2bbox(proposal_deltas, proposal_boxes)  # Nx(KxB)
        else:
            predict_boxes = self.box2box_transform.apply_deltas(proposal_deltas, proposal_boxes)  # Nx(KxB)

        return predict_boxes.split(num_prop_per_image)
    

    def delta2bbox(self, deltas, rois,
                   means=(0., 0., 0., 0.), 
                   stds=(0.1, 0.1, 0.2, 0.2),
                   max_shape=None,
                   wh_ratio_clip=16 / 1000,
                   clip_border=True):

        means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 4)
        stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 4)
        denorm_deltas = deltas * stds + means
        dx = denorm_deltas[:, 0::4]
        dy = denorm_deltas[:, 1::4]
        dw = denorm_deltas[:, 2::4]
        dh = denorm_deltas[:, 3::4]
        max_ratio = np.abs(np.log(wh_ratio_clip))
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)
        # Compute center of each roi
        px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
        py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
        # Compute width/height of each roi
        pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
        ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)
        # Use exp(network energy) to enlarge/shrink each roi
        gw = pw * dw.exp()
        gh = ph * dh.exp()
        # Use network energy to shift the center of each roi
        gx = px + pw * dx
        gy = py + ph * dy
        # Convert center-xy/width/height to top-left, bottom-right
        x1 = gx - gw * 0.5
        y1 = gy - gh * 0.5
        x2 = gx + gw * 0.5
        y2 = gy + gh * 0.5
        if clip_border and max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view(deltas.size())
        return bboxes
