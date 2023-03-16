# Copyright (c) Facebook, Inc. and its affiliates.
import copy
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

from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats

from ..utils.detic import load_class_freq, get_fed_loss_inds
from ..utils.adelaidet import aligned_bilinear, center_of_mass
from .zero_shot_classifier import ZeroShotClassifier


class WeakFastRCNNOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        use_sigmoid_ce=False,
        image_label_loss='mil',
        image_loss_weight=1.0,
        region_loss_weight=3.0,
        with_refinement_score=False,
        refinement_iou=0.5,
        prior_prob=0.01,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            **kwargs,
        )
        self.use_sigmoid_ce = use_sigmoid_ce
        self.image_label_loss = image_label_loss
        self.image_loss_weight = image_loss_weight
        self.region_loss_weight = region_loss_weight
        self.with_refinement_score = with_refinement_score
        self.refinement_iou = refinement_iou

        input_size = input_shape.channels * \
                     (input_shape.width or 1) * (input_shape.height or 1)
        del self.bbox_pred

        if self.use_sigmoid_ce:
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.cls_score.bias, bias_value)

        del self.cls_score
        self.cls_score = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, self.num_classes + 1),
        )
        weight_init.c2_xavier_fill(self.cls_score[0])
        nn.init.normal_(self.cls_score[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.cls_score[-1].bias, 0)

        self.prop_score = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, self.num_classes + 1),
        )
        weight_init.c2_xavier_fill(self.prop_score[0])
        nn.init.normal_(self.prop_score[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.prop_score[-1].bias, 0)

        if self.with_refinement_score:
            self.ref_score = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(inplace=True),
                nn.Linear(input_size, self.num_classes + 1),
            )
            weight_init.c2_xavier_fill(self.ref_score[0])
            nn.init.normal_(self.ref_score[-1].weight, mean=0, std=0.001)
            nn.init.constant_(self.ref_score[-1].bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'use_sigmoid_ce': cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE,
            'image_label_loss': cfg.MODEL.ROI_BOX_HEAD.IMAGE_LABEL_LOSS,
            'image_loss_weight': cfg.MODEL.ROI_BOX_HEAD.IMAGE_LOSS_WEIGHT,
            'region_loss_weight': cfg.MODEL.ROI_BOX_HEAD.REGION_LOSS_WEIGHT,
            'with_refinement_score': cfg.MODEL.ROI_BOX_HEAD.WITH_REFINEMENT_SCORE,
            'refinement_iou': cfg.MODEL.ROI_BOX_HEAD.REFINEMENT_IOU,
            'prior_prob': cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB,
        })

        return ret

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        cls_scores, prop_scores, ref_scores = [], [], []
        cls_scores = self.cls_score(x)
        prop_scores = self.prop_score(x)
        if self.with_refinement_score:
            ref_scores = self.ref_score(x)
        return cls_scores, prop_scores, ref_scores

    def predict_probs(self, predictions, proposals):
        cls_scores, prop_scores, ref_scores = predictions
        num_inst_per_image = [len(p) for p in proposals]
        cls_scores = cls_scores.split(num_inst_per_image, dim=0)   # B x n x (C + 1)
        prop_scores = prop_scores.split(num_inst_per_image, dim=0) # B x n x (C + 1)

        if self.with_refinement_score:
            ref_scores = ref_scores.split(num_inst_per_image, dim=0)
        else:
            ref_scores = [None for _ in num_inst_per_image]

        final_scores = []
        for cls_score, prop_score, ref_score in zip(cls_scores, prop_scores, ref_scores):
            if self.with_refinement_score:
                final_score = ref_score.sigmoid() if self.use_sigmoid_ce else F.softmax(ref_score, dim=-1)
            else:
                final_score = F.softmax(cls_score[:,:-1], dim=1) * F.softmax(prop_score[:,:-1], dim=0)
                # Concat dummy zero-scores for the background class.
                final_score = torch.cat([final_score, torch.zeros_like(final_score[:,:1])], dim=-1)
            final_scores.append(final_score)

        return final_scores

    def image_label_losses(self, predictions, proposals, targets):

        num_inst_per_image = [len(p) for p in proposals]
        cls_scores, prop_scores, ref_scores = predictions
        cls_scores = cls_scores.split(num_inst_per_image, dim=0)   # B x n x (C + 1)
        prop_scores = prop_scores.split(num_inst_per_image, dim=0) # B x n x (C + 1)

        if self.with_refinement_score:
            ref_scores = ref_scores.split(num_inst_per_image, dim=0)
        else:
            ref_scores = [None for _ in num_inst_per_image]

        B = len(cls_scores)
        loss = cls_scores[0].new_zeros([1])[0]
        region_loss = cls_scores[0].new_zeros([1])[0]

        for idx, (cls_score, target, prop_score, ref_score, proposal) in enumerate(zip(
                cls_scores, targets, prop_scores, ref_scores, proposals)):
            if cls_score.shape[0] == 0:
                continue
            labels = target._pos_category_ids
            if isinstance(labels, list):
                labels = torch.tensor(target._pos_category_ids, device=cls_score.device, dtype=torch.int64)

            if self.image_label_loss == 'mil':
                loss_mil = self._mil_loss(cls_score, prop_score, labels)
                loss += loss_mil

            if self.image_label_loss == 'oicr':
                loss_mil, loss_oicr = self._oicr_loss(cls_score, prop_score, ref_score,
                                                      labels, proposal)
                loss += loss_mil
                region_loss += loss_oicr

        loss = loss / B
        region_loss = region_loss / B

        loss_dict = {'loss_image': loss * self.image_loss_weight}
        if self.image_label_loss == 'oicr':
            loss_dict['loss_region'] = region_loss * self.region_loss_weight
        else:
            loss_dict['loss_region'] = loss * 0.0

        return loss_dict

    def _mil_loss(self, cls_score, prop_score, labels):
        final_score = F.softmax(cls_score[:,:-1], dim=1) * \
                          F.softmax(prop_score[:,:-1], dim=0)
        img_score = torch.clamp(
            torch.sum(final_score, dim=0),
            min=1e-10, max=1 - 1e-10)  # (C + 1)
        target = img_score.new_zeros(img_score.shape)  # (C + 1)
        target[labels] = 1.
        loss = F.binary_cross_entropy(img_score, target)
        return loss

    def _oicr_loss(self, cls_score, prop_score, ref_score, labels, proposal):
        final_score = F.softmax(cls_score[:,:-1], dim=1) * \
                          F.softmax(prop_score[:,:-1], dim=0)
        img_score = torch.clamp(
            torch.sum(final_score, dim=0),
            min=1e-10, max=1 - 1e-10)
        target = img_score.new_zeros(img_score.shape)
        target[labels] = 1.
        loss_mil = F.binary_cross_entropy(img_score, target)

        B = cls_score.shape[0]
        C = cls_score.shape[1] - 1
        pseudo_labels, loss_weights = self.oicr_layer(proposal, final_score, labels, C)
        if self.use_sigmoid_ce:
            target = cls_score.new_zeros(B, C + 1)
            target[range(len(pseudo_labels)), pseudo_labels] = 1  # B x (C + 1)
            target = target[:, :C]  # B x C
            loss_oicr = torch.mean(F.binary_cross_entropy_with_logits(
                ref_score[:, :C], target, reduction='none').sum(-1) * loss_weights)
        else:
            loss_oicr = torch.mean(F.cross_entropy(
                ref_score[:, :C+1], pseudo_labels, reduction='none') * loss_weights)
        return loss_mil, loss_oicr


    @torch.no_grad()
    def oicr_layer(self, proposals, source_score, labels, num_classes):
        gt_boxes = torch.zeros((0, 4), dtype=torch.float, device=labels.device)
        gt_scores = torch.zeros((0, 1), dtype=torch.float, device=labels.device)

        prob = source_score.clone()
        for c in labels:
            cls_prob = prob[:, c]
            max_index = torch.argmax(cls_prob)
            gt_boxes = torch.cat((gt_boxes, proposals.proposal_boxes.tensor[max_index].view(1, -1)), dim=0)
            gt_scores = torch.cat((gt_scores, cls_prob[max_index].view(1, 1)), dim=0)
            prob[max_index].fill_(0)

        if gt_boxes.shape[0] == 0:
            num_rois = len(source_score)
            pseudo_labels = torch.ones(num_rois, dtype=labels.dtype, device=labels.device) * num_classes
            loss_weights = torch.zeros(num_rois, dtype=source_score.dtype, device=labels.device)
        else:
            overlaps = pairwise_iou(Boxes(proposals.proposal_boxes.tensor), Boxes(gt_boxes))
            max_overlaps, gt_assignment = overlaps.max(dim=1)
            pseudo_labels = labels[gt_assignment]
            loss_weights = gt_scores[gt_assignment].view(-1)
            pseudo_labels[max_overlaps < self.refinement_iou] = num_classes

        return pseudo_labels, loss_weights
