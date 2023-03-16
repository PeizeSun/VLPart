# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

import copy

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads

from .fast_rcnn_weak import WeakFastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class Res5WeakROIHeads(Res5ROIHeads):
    @configurable
    def __init__(
        self,
        *,
        recognition_predictor: nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)
        del self.box_predictor
        self.recognition_predictor = recognition_predictor

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)

        stage_channel_factor = 2 ** 3
        out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        recognition_predictor = WeakFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1))
        ret["recognition_predictor"] = recognition_predictor
        return ret

    def forward(self, images, features, proposals, targets=None):
        del images
        image_sizes = [x.image_size for x in proposals]

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.recognition_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            losses = {}

            recognition_loss = self.recognition_predictor.image_label_losses(
                predictions, proposals, targets,
            )
            losses.update(recognition_loss)

            return proposals, losses

        else:
            boxes = [p.proposal_boxes.tensor for p in proposals]
            scores = self.recognition_predictor.predict_probs(predictions, proposals)

            predictor = self.recognition_predictor
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )

            return pred_instances, {}



@ROI_HEADS_REGISTRY.register()
class VGG16WeakROIHeads(nn.Module):
    def __init__(self, config, input_shape):
        super(VGG16WeakROIHeads, self).__init__()
        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type = config.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = ROIPooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.pooler = pooler

        self.classifier = nn.Sequential(
            nn.Identity(),
            # nn.Identity() is used to align parameter indices in pre-trained model,
            # so that pre-trained parameters can be loaded
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        output_shape = ShapeSpec(channels=4096)

        self.recognition_predictor = WeakFastRCNNOutputLayers(
            config, output_shape)

    def forward(self, images, img_features, proposals, targets=None):
        del images
        features = img_features
        image_sizes = [x.image_size for x in proposals]

        box_features = self.forward_head(features, proposals)
        predictions = self.recognition_predictor(box_features)

        if self.training:
            losses = {}
            recognition_loss = self.recognition_predictor.image_label_losses(
                predictions, proposals, targets,
            )
            losses.update(recognition_loss)

            return proposals, losses

        else:
            boxes = [p.proposal_boxes.tensor for p in proposals]
            scores = self.recognition_predictor.predict_probs(predictions, proposals)

            predictor = self.recognition_predictor
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            return pred_instances, {}

    def forward_head(self, features, proposals):
        # also pool featurs of multiple images into one huge ROI tensor
        pool_boxes = [p.proposal_boxes for p in proposals]
        roi_features = self.pooler(features, pool_boxes)
        flatten_features = roi_features.view(roi_features.shape[0], -1)
        x = self.classifier(flatten_features)
        return x
