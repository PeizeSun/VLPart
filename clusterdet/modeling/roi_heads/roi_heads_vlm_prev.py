# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import inspect
import logging

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou, pairwise_ioa
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.sampling import subsample_labels

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.roi_heads import Res5ROIHeads

from .fast_rcnn_vlm import (
    BoxFastRCNNOutputLayers,
    ClassificationFastRCNNOutputLayers
)


@ROI_HEADS_REGISTRY.register()
class VLMROIHeads(StandardROIHeads):
    @configurable
    def __init__(
        self,
        *,
        classification_head: nn.Module,
        classification_predictor: nn.Module,
        add_image_box: bool = False,
        image_box_size: float = 1.0,
        ws_num_props: int = 512,
        mult_object_score: bool = False,
        refine_box: bool = False,
        with_image_labels: bool = False,
        mask_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classification_head = classification_head
        self.classification_predictor = classification_predictor
        self.add_image_box = add_image_box
        self.image_box_size = image_box_size
        self.ws_num_props = ws_num_props
        self.mult_object_score = mult_object_score
        self.refine_box = refine_box
        if self.refine_box:
            assert self.mult_object_score, "otherwise training and inference is not aligned"
        self.with_image_labels = with_image_labels
        self.mask_weight = mask_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'add_image_box': cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX,
            'image_box_size': cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE,
            'ws_num_props': cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS,
            'mult_object_score': cfg.MODEL.ROI_BOX_HEAD.MULT_OBJECT_SCORE,
            'refine_box': cfg.MODEL.ROI_BOX_HEAD.REFINE_BOX,
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'mask_weight': cfg.MODEL.ROI_HEADS.MASK_WEIGHT,
        })
        return ret

    @classmethod
    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        pooled_shape = ShapeSpec(
            channels=in_channels, height=pooler_resolution, width=pooler_resolution
        )
        box_head = build_box_head(cfg, pooled_shape)
        box_predictor = BoxFastRCNNOutputLayers(cfg, box_head.output_shape)

        classification_head = build_box_head(cfg, pooled_shape)
        classification_predictor = ClassificationFastRCNNOutputLayers(cfg, box_head.output_shape)

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "classification_head": classification_head,
            "classification_predictor": classification_predictor,
        }

    def forward(self, images, img_features, proposals, targets=None,
                ann_type='box', classifier_info=(None,None,None,None),
                dataset_source=None):
        del images
        features = [img_features[f] for f in self.box_in_features]
        image_sizes = [x.image_size for x in proposals]

        if self.training:
            pool_features = None
            if ann_type in ['box', 'part']:
                proposals = self.label_and_sample_proposals(proposals, targets)
                pool_boxes = [x.proposal_boxes for x in proposals]
                pool_features = self.box_pooler(features, pool_boxes)

                box_features = self.box_head(pool_features)
                box_predictions = self.box_predictor(box_features)
                box_loss = self.box_predictor.losses(box_predictions, proposals)

                if targets[0].has('gt_masks'):
                    mask_loss = self._forward_mask(img_features, proposals)
                else:
                    mask_loss = self._get_empty_mask_loss(
                        img_features, proposals, device=proposals[0].objectness_logits.device)
            else:
                box_loss = self._get_empty_box_loss(
                    img_features, proposals, device=proposals[0].objectness_logits.device)
                mask_loss = self._get_empty_mask_loss(
                    img_features, proposals, device=proposals[0].objectness_logits.device)

            if ann_type in ['box', 'part']:
                classification_features = self.classification_head(pool_features)
                classification_predictions = self.classification_predictor(
                    classification_features, dataset_source=dataset_source)
                classification_loss = self.classification_predictor.classification_losses(
                    classification_predictions, proposals,
                )
            elif ann_type in ['ppart']:
                proposals = self.label_and_sample_proposals(proposals, targets)
                pool_boxes = [x.proposal_boxes for x in proposals]
                pool_features = self.box_pooler(features, pool_boxes)

                classification_features = self.classification_head(pool_features)
                classification_predictions = self.classification_predictor(
                    classification_features, dataset_source=dataset_source)
                classification_loss = self.classification_predictor.classification_part_losses(
                    classification_predictions, proposals,
                )
            elif ann_type in ['box2part']:
                proposals = self.get_proposals_inside_objects(proposals, targets)
                pool_boxes = [x.proposal_boxes for x in proposals]
                pool_features = self.box_pooler(features, pool_boxes)
                classification_features = self.classification_head(pool_features)
                part_predictions = self.classification_predictor(
                    classification_features, dataset_source=dataset_source)
                classification_loss = self.classification_predictor.box2part_losses(
                    part_predictions, proposals, targets, classifier_info=classifier_info
                )
            elif ann_type in ['img2part']:
                pool_boxes = [x.proposal_boxes for x in proposals]
                pool_features = self.box_pooler(features, pool_boxes)
                classification_features = self.classification_head(pool_features)
                part_predictions = self.classification_predictor(
                    classification_features, dataset_source=dataset_source)
                classification_loss = self.classification_predictor.img2part_losses(
                    part_predictions, proposals, targets, classifier_info=classifier_info
                )
            else:
                proposals = self.get_top_proposals(proposals)
                pool_boxes = [x.proposal_boxes for x in proposals]
                pool_features = self.box_pooler(features, pool_boxes)
                classification_features = self.classification_head(pool_features)
                classification_predictions = self.classification_predictor(
                    classification_features, dataset_source=dataset_source,
                    classifier_info=classifier_info)
                classification_loss = self.classification_predictor.image_label_losses(
                    classification_predictions, proposals, targets, classifier_info=classifier_info
                )

            losses = {}
            losses.update(box_loss)
            losses.update(classification_loss)
            losses.update({k: v * self.mask_weight for k, v in mask_loss.items()})

            return proposals, losses

        else:
            box_predictions = self._run_box(features, proposals)
            object_boxes = self.box_predictor.predict_boxes(box_predictions, proposals)
            object_scores = self.box_predictor.predict_probs(box_predictions, proposals)

            if self.refine_box:
                proposals = self._create_proposals_from_boxes(object_boxes, image_sizes)
            predictions = self._run_classification(features, proposals)
            category_scores = self.classification_predictor.predict_probs(predictions, proposals)

            if self.refine_box:
                boxes = [p.proposal_boxes.tensor for p in proposals]
            else:
                boxes = object_boxes

            if self.mult_object_score:
                scores = [(cs * ps[:, :1]) ** 0.5 for cs, ps in zip(category_scores, object_scores)]
            else:
                scores = category_scores

            predictor = self.classification_predictor
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            pred_instances = self.forward_with_given_boxes(img_features, pred_instances)

            return pred_instances, {}

    def get_proposals_inside_objects(self, proposals, targets):
        new_proposals = []
        for proposal, target in zip(proposals, targets):
            fg_iou = pairwise_ioa(target.gt_boxes, proposal.proposal_boxes)
            new_proposals.append(proposal[torch.max(fg_iou, dim=0)[0] > 0.9])
        return new_proposals

    def get_top_proposals(self, proposals):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.clip(proposals[i].image_size)
        proposals = [p[:self.ws_num_props] for p in proposals]
        for i, p in enumerate(proposals):
            p.proposal_boxes.tensor = p.proposal_boxes.tensor.detach()
            if self.add_image_box:
                proposals[i] = self._add_image_box(p)
        return proposals

    @torch.no_grad()
    def _add_image_box(self, p):
        image_box = Instances(p.image_size)
        n = 1
        h, w = p.image_size
        f = self.image_box_size
        image_box.proposal_boxes = Boxes(
            p.proposal_boxes.tensor.new_tensor(
                [w * (1. - f) / 2.,
                    h * (1. - f) / 2.,
                    w * (1. - (1. - f) / 2.),
                    h * (1. - (1. - f) / 2.)]
                ).view(n, 4))
        image_box.objectness_logits = p.objectness_logits.new_ones(n)
        return Instances.cat([p, image_box])

    def _get_empty_box_loss(self, features, proposals, device):
        return {
            'loss_box_score': torch.zeros((1,), device=device, dtype=torch.float32)[0],
            'loss_box_reg': torch.zeros((1,), device=device, dtype=torch.float32)[0],
        }

    def _get_empty_mask_loss(self, features, proposals, device):
        return {'loss_mask': torch.zeros(
            (1,), device=device, dtype=torch.float32)[0]}

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals

    def _run_box(self, features, proposals):
        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = self.box_head(box_features)

        return self.box_predictor(box_features)

    def _run_classification(self, features, proposals, classifier_info=(None,None,None,None)):
        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = self.classification_head(box_features)

        return self.classification_predictor(
            box_features, classifier_info=classifier_info)
