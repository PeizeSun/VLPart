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
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
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

from .fast_rcnn_joint import (
    WeakFastRCNNOutputLayers,
    BoxFastRCNNOutputLayers,
    ClassificationFastRCNNOutputLayers
)


def build_recognition_heads(cfg, backbone):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.RECOGNITION_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, backbone)


@ROI_HEADS_REGISTRY.register()
class Res5RecognitionROIHeads(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        pooler: ROIPooler,
        # res5plus: nn.Module,
        layer_plus: nn.Module,
        recognition_predictor: nn.Module,
    ):
        super().__init__()
        self.in_features = in_features
        self.pooler = pooler
        # self.res5plus = res5plus
        self.layer_plus = layer_plus
        self.recognition_predictor = recognition_predictor

    @classmethod
    def from_config(cls, cfg, backbone):
        ret = {}
        in_features = ret["in_features"] = cfg.MODEL.RECOGNITION_HEADS.IN_FEATURES

        pooler_resolution = cfg.MODEL.RECOGNITION_HEADS.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.RECOGNITION_HEADS.POOLER_TYPE
        pooler_scales     = (1.0 / cfg.MODEL.RECOGNITION_HEADS.POOLER_SCALES, )
        sampling_ratio    = cfg.MODEL.RECOGNITION_HEADS.POOLER_SAMPLING_RATIO
        assert len(in_features) == 1

        ret["pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # ret["res5plus"], out_channels = cls._build_res5plus_block(cfg)
        if 'resnet' in cfg.MODEL.TIMM.BASE_NAME:
            ret["layer_plus"] = copy.deepcopy(backbone.bottom_up.base.layer4)
        elif 'convnext' in cfg.MODEL.TIMM.BASE_NAME:
            ret["layer_plus"] = copy.deepcopy(backbone.bottom_up.base.stages_3)
        else:
            raise NotImplementedError

        # stage_channel_factor = 2 ** 3
        # out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        out_channels = cfg.MODEL.RECOGNITION_HEADS.OUT_CHANNELS
        recognition_predictor = WeakFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1))

        ret["recognition_predictor"] = recognition_predictor

        return ret

    @classmethod
    def _build_res5plus_block(cls, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = ResNet.make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def forward(self, img_features, gt_instances, res5_module,
                ann_type='noisy', classifier_info=(None,None,None),):
        assert gt_instances is not None
        if ann_type == 'clean':
            mining_instances = gt_instances
            device = img_features[self.in_features[0]].device
            image_loss = {
                'loss_weak_image': torch.zeros((1,), device=device, dtype=torch.float32)[0],
                'loss_weak_region': torch.zeros((1,), device=device, dtype=torch.float32)[0],
            }
            return image_loss, mining_instances

        features = [img_features[f] for f in self.in_features]

        proposals = self.proposals_for_recognition(gt_instances)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        pool_features = self.pooler(features, proposal_boxes)
        # box_features = res5_module(pool_features)
        # box_features = self.res5plus(pool_features)
        box_features = self.layer_plus(pool_features)

        predictions = self.recognition_predictor(box_features.mean(dim=[2, 3]))
        image_loss, mining_instances = self.recognition_predictor.mining_instance_losses(
            predictions, proposals, gt_instances,
        )
        return image_loss, mining_instances

    def forward_test(self, img_features, res5_module, proposals):
        image_sizes = [x.image_size for x in proposals]

        features = [img_features[f] for f in self.in_features]
        proposal_boxes = [x.proposal_boxes for x in proposals]
        pool_features = self.pooler(features, proposal_boxes)
        # box_features = res5_module(pool_features)
        # box_features = self.res5plus(pool_features)
        box_features = self.layer_plus(pool_features)

        predictions = self.recognition_predictor(box_features.mean(dim=[2, 3]))
        boxes = [p.proposal_boxes.tensor for p in proposals]
        scores = self.recognition_predictor.predict_probs(predictions, proposals)

        pred_instances, _ = fast_rcnn_inference(
            boxes,
            scores,
            image_sizes,
            0.00001,
            0.5,
            100,
        )

        return pred_instances, {}

    @torch.no_grad()
    def proposals_for_recognition(self, proposals):
        new_proposals = []
        for i, p in enumerate(proposals):
            image_size = p.image_size
            boxes_per_image = p.gt_boxes
            boxes_per_image.clip(image_size)

            new_p = Instances(image_size)
            new_p.proposal_boxes = boxes_per_image
            new_proposals.append(new_p)

        return new_proposals


@ROI_HEADS_REGISTRY.register()
class JointROIHeads(StandardROIHeads):
    @configurable
    def __init__(
        self,
        *,
        classification_head: nn.Module,
        classification_predictor: nn.Module,
        add_image_box: bool = False,
        image_box_size: float = 1.0,
        mult_object_score: bool = False,
        with_image_labels: bool = False,
        mask_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classification_head = classification_head
        self.classification_predictor = classification_predictor
        self.add_image_box = add_image_box
        self.image_box_size = image_box_size
        self.mult_object_score = mult_object_score
        self.with_image_labels = with_image_labels
        self.mask_weight = mask_weight

        # if self.mask_on:
        #     for param in self.mask_head.parameters():
        #         param.requires_grad = False

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'add_image_box': cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX,
            'image_box_size': cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE,
            'mult_object_score': cfg.MODEL.ROI_BOX_HEAD.MULT_OBJECT_SCORE,
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

    def forward(self, images, img_features, proposals, box_targets=None, mask_targets=None,
                ann_type='noisy', classifier_info=(None,None,None),):
        del images
        features = [img_features[f] for f in self.box_in_features]
        image_sizes = [x.image_size for x in proposals]

        if self.training:
            box_proposals = self.label_and_sample_proposals(proposals, box_targets)
            pool_boxes = [x.proposal_boxes for x in box_proposals]
            pool_features = self.box_pooler(features, pool_boxes)

            box_features = self.box_head(pool_features)
            box_predictions = self.box_predictor(box_features)
            box_loss = self.box_predictor.losses(box_predictions, box_proposals)

            classification_features = self.classification_head(pool_features)
            classification_predictions = self.classification_predictor(classification_features)

            if ann_type == 'clean':
                classification_loss = self.classification_predictor.classification_losses(
                    classification_predictions, box_proposals, box_targets,
                )
            else:
                classification_loss = self.classification_predictor.image_label_losses(
                    classification_predictions, box_proposals, box_targets,
                )

            losses = {}
            losses.update(box_loss)
            losses.update(classification_loss)

            if mask_targets[0].has('gt_masks'):
                mask_targets = self.targets_for_mask(mask_targets)
                mask_proposals = self.label_and_sample_proposals(proposals, mask_targets)
                # frozen_features = {k: v.detach() for k, v in img_features.items()}
                mask_losses = self._forward_mask(img_features, mask_proposals)
                losses.update({k: v * self.mask_weight for k, v in mask_losses.items()})
            else:
                losses.update(self._get_empty_mask_loss(
                    img_features, proposals,
                    device=proposals[0].objectness_logits.device))

            return proposals, losses

        else:
            box_predictions = self._run_box(features, proposals)
            object_boxes = self.box_predictor.predict_boxes(box_predictions, proposals)
            object_scores = self.box_predictor.predict_probs(box_predictions, proposals)

            proposals = self._create_proposals_from_boxes(object_boxes, image_sizes)
            predictions = self._run_classification(features, proposals)
            category_scores = self.classification_predictor.predict_probs(predictions, proposals)

            boxes = [p.proposal_boxes.tensor for p in proposals]
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

    @torch.no_grad()
    def targets_for_mask(self, targets):
        new_targets = []
        for i, t in enumerate(targets):
            image_size = t.image_size
            classes_per_image = t.gt_classes
            boxes_per_image = t.gt_boxes
            masks_per_image = t.gt_masks
            boxes_per_image.clip(image_size)

            keep = masks_per_image.tensor.sum((1, 2)) > 16
            new_t = Instances(image_size)
            new_t.gt_classes = classes_per_image[keep]
            new_t.gt_boxes = boxes_per_image[keep]
            new_t.gt_masks = masks_per_image[keep]
            new_targets.append(new_t)

        return new_targets

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


    def _run_classification(self, features, proposals, classifier_info=(None,None,None)):
        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = self.classification_head(box_features)

        return self.classification_predictor(box_features)
