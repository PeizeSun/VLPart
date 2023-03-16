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
from detectron2.structures import Boxes, Instances, BitMasks, pairwise_iou
from detectron2.utils.events import get_event_storage
import detectron2.utils.comm as comm
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

from .zero_shot_classifier import ZeroShotClassifier


class WeakFastRCNNOutputLayers(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        num_classes: int,
        *,
        image_label_loss='mil',
        image_loss_weight=1.0,
        region_loss_weight=3.0,
        with_refinement_score=False,
        refinement_iou=0.5,
        use_sigmoid_ce=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_label_loss = image_label_loss
        self.image_loss_weight = image_loss_weight
        self.region_loss_weight = region_loss_weight
        self.with_refinement_score = with_refinement_score
        self.refinement_iou = refinement_iou
        self.use_sigmoid_ce = use_sigmoid_ce

        input_size = input_shape.channels * \
                     (input_shape.width or 1) * (input_shape.height or 1)

        self.category_score = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, self.num_classes + 1),
        )
        weight_init.c2_xavier_fill(self.category_score[0])
        nn.init.normal_(self.category_score[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.category_score[-1].bias, 0)

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
            if self.use_sigmoid_ce:
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                nn.init.constant_(self.ref_score[-1].bias, bias_value)
            else:
                nn.init.constant_(self.ref_score[-1].bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            "input_shape": input_shape,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'image_label_loss': cfg.MODEL.RECOGNITION_HEADS.IMAGE_LABEL_LOSS,
            'image_loss_weight': cfg.MODEL.RECOGNITION_HEADS.IMAGE_LOSS_WEIGHT,
            'region_loss_weight': cfg.MODEL.RECOGNITION_HEADS.REGION_LOSS_WEIGHT,
            'with_refinement_score': cfg.MODEL.RECOGNITION_HEADS.WITH_REFINEMENT_SCORE,
            'refinement_iou': cfg.MODEL.RECOGNITION_HEADS.REFINEMENT_IOU,
            'use_sigmoid_ce': cfg.MODEL.RECOGNITION_HEADS.USE_SIGMOID_CE,
        }
        return ret

    def forward(self, x, classifier_info=(None, None, None)):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        cls_scores, prop_scores, ref_scores = [], [], []
        if classifier_info[0] is not None:
            cls_scores = self.category_score(x, classifier=classifier_info[0])
            prop_scores = self.prop_score(x, classifier=classifier_info[0])
        else:
            cls_scores = self.category_score(x)
            prop_scores = self.prop_score(x)

        if self.with_refinement_score:
            if classifier_info[0] is not None:
                ref_scores = self.ref_score(x, classifier=classifier_info[0])
            else:
                ref_scores = self.ref_score(x)

        return cls_scores, prop_scores, ref_scores

    def predict_probs(self, predictions, proposals):
        cls_scores, prop_scores, ref_scores = predictions
        num_inst_per_image = [len(p) for p in proposals]
        cls_scores = cls_scores.split(num_inst_per_image, dim=0)  # B x n x (C + 1)
        prop_scores = prop_scores.split(num_inst_per_image, dim=0)  # B x n x (C + 1)

        if self.with_refinement_score:
            ref_scores = ref_scores.split(num_inst_per_image, dim=0)
        else:
            ref_scores = [None for _ in num_inst_per_image]

        final_scores = []
        for cls_score, prop_score, ref_score in zip(cls_scores, prop_scores, ref_scores):
            if self.with_refinement_score:
                final_score = ref_score.sigmoid() if self.use_sigmoid_ce else F.softmax(ref_score, dim=-1)
            else:
                final_score = F.softmax(cls_score[:, :-1], dim=1) * F.softmax(prop_score[:, :-1], dim=0)
                # Concat dummy zero-scores for the background class.
                final_score = torch.cat([final_score, torch.zeros_like(final_score[:, :1])], dim=-1)
            final_scores.append(final_score)

        return final_scores

    def mining_instance_losses(self, predictions, proposals, targets,
                               classifier_info=(None, None, None), ann_type='image'):

        num_inst_per_image = [len(p) for p in proposals]
        cls_scores, prop_scores, ref_scores = predictions
        cls_scores = cls_scores.split(num_inst_per_image, dim=0)  # B x n x (C + 1)
        prop_scores = prop_scores.split(num_inst_per_image, dim=0)  # B x n x (C + 1)

        if self.with_refinement_score:
            ref_scores = ref_scores.split(num_inst_per_image, dim=0)
        else:
            ref_scores = [None for _ in num_inst_per_image]

        B = len(cls_scores)
        image_loss = cls_scores[0].new_zeros([1])[0]
        region_loss = cls_scores[0].new_zeros([1])[0]
        mining_instances = []

        for idx, (cls_score, target, prop_score, ref_score, proposal) in enumerate(zip(
                cls_scores, targets, prop_scores, ref_scores, proposals)):
            if cls_score.shape[0] == 0:
                mining_instances.append(self.dummy_instance(target))
                continue

            labels = torch.tensor(target._pos_category_ids, device=cls_score.device, dtype=torch.int64)
            # if self.dynamic_classifier:
            #     labels = classifier_info[1][1][labels]

            if self.image_label_loss == 'mil':
                loss_mil, final_score = self._mil_loss(cls_score, prop_score, labels)
                image_loss += loss_mil
            elif self.image_label_loss == 'oicr':
                loss_mil, loss_oicr, final_score = self._oicr_loss(cls_score, prop_score, ref_score,
                                                                   labels, proposal)
                image_loss += loss_mil
                region_loss += loss_oicr
            else:
                raise NotImplementedError

            mining_target = self.mining_layer(final_score, labels, proposal)
            mining_instances.append(mining_target)

        image_loss = image_loss / B
        region_loss = region_loss / B

        loss_dict = {
            'loss_weak_image': image_loss * self.image_loss_weight,
            'loss_weak_region': region_loss * self.region_loss_weight,
        }

        return loss_dict, mining_instances

    def _mil_loss(self, cls_score, prop_score, labels):
        final_score = F.softmax(cls_score[:, :-1], dim=1) * \
                      F.softmax(prop_score[:, :-1], dim=0)
        img_score = torch.clamp(
            torch.sum(final_score, dim=0),
            min=1e-10, max=1 - 1e-10)  # (C + 1)
        target = img_score.new_zeros(img_score.shape)  # (C + 1)
        target[labels] = 1.
        loss = F.binary_cross_entropy(img_score, target)
        return loss, final_score

    def _oicr_loss(self, cls_score, prop_score, ref_score, labels, proposal):
        final_score = F.softmax(cls_score[:, :-1], dim=1) * \
                      F.softmax(prop_score[:, :-1], dim=0)
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
                ref_score[:, :C + 1], pseudo_labels, reduction='none') * loss_weights)

        final_score = ref_score.sigmoid() if self.use_sigmoid_ce else F.softmax(ref_score, dim=-1)

        return loss_mil, loss_oicr, final_score[:, :-1]

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

    @torch.no_grad()
    def mining_layer(self, source_score, labels, proposals):
        gt_boxes = torch.zeros((0, 4), dtype=torch.float, device=labels.device)
        gt_scores = torch.zeros((0,), dtype=torch.float, device=labels.device)

        prob = source_score.clone()
        for c in labels:
            cls_prob = prob[:, c]
            max_index = torch.argmax(cls_prob)
            gt_boxes = torch.cat((gt_boxes, proposals.proposal_boxes.tensor[max_index].view(1, -1)), dim=0)
            gt_scores = torch.cat((gt_scores, cls_prob[max_index].view(1)), dim=0)
            prob[max_index].fill_(0)

        pseudo_target = Instances(proposals.image_size)
        pseudo_target.gt_boxes = Boxes(gt_boxes)
        pseudo_target.gt_classes = labels
        pseudo_target.gt_loss_weights = gt_scores

        return pseudo_target

    def dummy_instance(self, target):
        dummy_target = Instances(target.image_size)
        dummy_target.gt_boxes = target.gt_boxes
        dummy_target.gt_classes = target.gt_classes
        dummy_target.gt_loss_weights = torch.zeros_like(target.gt_classes)
        return dummy_target


class BoxFastRCNNOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            box_score_reg_loss=False,
            **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            **kwargs,
        )

        self.box_score_reg_loss = box_score_reg_loss

        input_size = input_shape.channels * \
                     (input_shape.width or 1) * (input_shape.height or 1)
        del self.cls_score
        del self.bbox_pred

        self.bbox_pred = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, 4)
        )
        weight_init.c2_xavier_fill(self.bbox_pred[0])
        nn.init.normal_(self.bbox_pred[-1].weight, std=0.001)
        nn.init.constant_(self.bbox_pred[-1].bias, 0)

        self.bbox_score = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, 1)
        )
        weight_init.c2_xavier_fill(self.bbox_score[0])
        nn.init.normal_(self.bbox_score[-1].weight, std=0.01)
        nn.init.constant_(self.bbox_score[-1].bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'box_score_reg_loss': cfg.MODEL.ROI_BOX_HEAD.BOX_SCORE_REG_LOSS,
        })
        return ret

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        proposal_deltas = self.bbox_pred(x)
        box_scores = self.bbox_score(x)
        return [], proposal_deltas, box_scores

    def losses(self, predictions, proposals):
        scores, proposal_deltas, box_scores = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        num_classes = self.num_classes

        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        loss_box_reg = self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes, num_classes=num_classes)
        loss_box_score = self.box_score_loss(box_scores, gt_classes, proposal_boxes, gt_boxes)

        loss_dict = {
            "loss_box_score": loss_box_score,
            "loss_box_reg": loss_box_reg,
        }
        return loss_dict

    def box_score_loss(self, box_scores, gt_classes, proposal_boxes, gt_boxes):
        if gt_classes.numel() == 0:
            return box_scores.sum() * 0.0  # connect the gradient

        if not self.box_score_reg_loss:
            target = box_scores.new_zeros(gt_classes.shape)
            target[(gt_classes >= 0) & (gt_classes < self.num_classes)] = 1
            cls_loss = F.binary_cross_entropy_with_logits(
                box_scores.view(-1), target, reduction='none')  # B x 1
            loss_box_score = cls_loss.sum() / max(gt_classes.numel(), 1.0)
            return loss_box_score

        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        fg_iou = torch.diag(pairwise_iou(Boxes(proposal_boxes[fg_inds]), Boxes(gt_boxes[fg_inds])))
        loss = smooth_l1_loss(box_scores[fg_inds].sigmoid().view(-1), fg_iou, beta=0.0, reduction="sum", )
        loss_box_score = loss / max(gt_classes.numel(), 1.0)

        return loss_box_score

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes, num_classes=-1):
        """
        Allow custom background index
        """
        num_classes = num_classes if num_classes > 0 else self.num_classes
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum")
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        return loss_box_reg / max(gt_classes.numel(), 1.0)

    def predict_boxes(self, predictions, proposals):
        if not len(proposals):
            return []
        _, proposal_deltas, _ = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)

        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        if not len(proposals):
            return []
        _, _, box_scores = predictions
        num_inst_per_image = [len(p) for p in proposals]

        probs = box_scores.sigmoid()

        # Concat dummy zero-scores for the background class.
        probs = torch.cat([probs, torch.zeros_like(probs)], dim=-1)

        return probs.split(num_inst_per_image, dim=0)


class ClassificationFastRCNNOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            zeroshot_cls=None,
            sync_caption_batch=False,
            use_sigmoid_ce=False,
            use_fed_loss=False,
            ignore_zero_cats=False,
            fed_loss_num_cat=50,
            dynamic_classifier=False,
            use_zeroshot_cls=False,
            with_caption_loss=False,
            caption_weight=1.0,
            neg_cap_weight=1.0,
            add_image_box=False,
            prior_prob=0.01,
            cat_freq_path='',
            fed_loss_freq_weight=0.5,
            **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            **kwargs,
        )
        self.sync_caption_batch = sync_caption_batch
        self.use_sigmoid_ce = use_sigmoid_ce
        self.use_fed_loss = use_fed_loss
        self.ignore_zero_cats = ignore_zero_cats
        self.fed_loss_num_cat = fed_loss_num_cat
        self.dynamic_classifier = dynamic_classifier
        self.use_zeroshot_cls = use_zeroshot_cls
        self.with_caption_loss = with_caption_loss
        self.caption_weight = caption_weight
        self.neg_cap_weight = neg_cap_weight
        self.add_image_box = add_image_box

        input_size = input_shape.channels * \
                     (input_shape.width or 1) * (input_shape.height or 1)
        del self.bbox_pred

        if self.use_fed_loss or self.ignore_zero_cats:
            freq_weight = load_class_freq(cat_freq_path, fed_loss_freq_weight)
            self.register_buffer('freq_weight', freq_weight)
        else:
            self.freq_weight = None

        if self.use_fed_loss and len(self.freq_weight) < self.num_classes:
            print('Extending federated loss weight')
            self.freq_weight = torch.cat(
                [self.freq_weight,
                 self.freq_weight.new_zeros(
                     self.num_classes - len(self.freq_weight))]
            )

        assert (not self.dynamic_classifier) or (not self.use_fed_loss)

        # cls_score
        if self.use_zeroshot_cls:
            assert zeroshot_cls is not None
            self.cls_score = copy.deepcopy(zeroshot_cls)
        else:
            self.cls_score = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(inplace=True),
                nn.Linear(input_size, self.num_classes + 1),
            )
            weight_init.c2_xavier_fill(self.cls_score[0])
            nn.init.normal_(self.cls_score[-1].weight, mean=0, std=0.001)
            if self.use_sigmoid_ce:
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                nn.init.constant_(self.cls_score[-1].bias, bias_value)
            else:
                nn.init.constant_(self.cls_score[-1].bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'sync_caption_batch': cfg.MODEL.SYNC_CAPTION_BATCH,
            'use_sigmoid_ce': cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE,
            'use_fed_loss': cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS,
            'ignore_zero_cats': cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS,
            'fed_loss_num_cat': cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'use_zeroshot_cls': cfg.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS,
            'with_caption_loss': cfg.MODEL.ROI_BOX_HEAD.WITH_CAPTION_LOSS,
            'caption_weight': cfg.MODEL.ROI_BOX_HEAD.CAPTION_WEIGHT,
            'neg_cap_weight': cfg.MODEL.ROI_BOX_HEAD.NEG_CAP_WEIGHT,
            'add_image_box': cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX,
            'prior_prob': cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB,
            'cat_freq_path': cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
            'fed_loss_freq_weight': cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT,
        })
        if ret['use_zeroshot_cls']:
            ret['zeroshot_cls'] = ZeroShotClassifier(cfg, input_shape)

        return ret

    def forward(self, x, classifier_info=(None, None, None)):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        cls_scores = []
        if classifier_info[0] is not None:
            cls_scores.append(self.cls_score(x, classifier=classifier_info[0]))
        else:
            cls_scores.append(self.cls_score(x))

        if classifier_info[2] is not None:
            cap_cls = classifier_info[2]
            if self.sync_caption_batch:
                caption_scores = self.cls_score(x, classifier=cap_cls[:, :-1])
            else:
                caption_scores = self.cls_score(x, classifier=cap_cls)
            cls_scores.append(caption_scores)

        cls_scores = torch.cat(cls_scores, dim=1)  # B x C' or B x N or B x (C'+ N)

        return cls_scores

    def predict_probs(self, predictions, proposals):
        cls_scores = predictions
        num_inst_per_image = [len(p) for p in proposals]
        cls_scores = cls_scores.split(num_inst_per_image, dim=0)

        final_scores = []
        for cls_score in cls_scores:
            final_score = cls_score.sigmoid() if self.use_sigmoid_ce else F.softmax(cls_score, dim=-1)
            final_scores.append(final_score)
        return final_scores

    def classification_losses(self, predictions, proposals, targets, classifier_info=(None, None, None), ):
        scores = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        else:
            loss_cls = self.softmax_cross_entropy_loss(scores, gt_classes)

        loss_dict = {
            "loss_cls": loss_cls,
            'loss_weak_cls': scores.new_zeros([1])[0],
            'loss_caption': scores.new_zeros([1])[0],
        }
        return loss_dict

    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]  # This is more robust than .sum() * 0.

        B = pred_class_logits.shape[0]
        C = pred_class_logits.shape[1] - 1

        target = pred_class_logits.new_zeros(B, C + 1)
        target[range(len(gt_classes)), gt_classes] = 1  # B x (C + 1)
        target = target[:, :C]  # B x C

        weight = 1

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits[:, :-1], target, reduction='none')  # B x C
        loss = torch.sum(cls_loss * weight) / B
        return loss

    def softmax_cross_entropy_loss(self, pred_class_logits, gt_classes):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]
        loss = F.cross_entropy(
            pred_class_logits, gt_classes, reduction="mean")
        return loss

    def image_label_losses(self, predictions, proposals, targets, classifier_info=(None, None, None), ):
        num_inst_per_image = [len(p) for p in proposals]
        cls_scores = predictions
        cls_scores = cls_scores.split(num_inst_per_image, dim=0)  # B x n x (C + 1)

        B = len(cls_scores)
        cls_loss = cls_scores[0].new_zeros([1])[0]
        caption_loss = cls_scores[0].new_zeros([1])[0]

        for idx, (cls_score, proposal, target) in enumerate(zip(cls_scores, proposals, targets)):
            if cls_score.shape[0] == 0:
                continue

            # labels = torch.tensor(target._pos_category_ids, device=cls_score.device, dtype=torch.int64)
            # if self.dynamic_classifier:
            #     labels = classifier_info[1][1][labels]

            loss_cls = self._cls_loss(cls_score, proposal)
            cls_loss += loss_cls

            if self.with_caption_loss:
                loss_caption = self._caption_loss(cls_score, classifier_info, idx, B)
                caption_loss += loss_caption

        cls_loss = cls_loss / B
        caption_loss = caption_loss / B

        loss_dict = {
            'loss_weak_cls': cls_loss,
            'loss_caption': caption_loss * self.caption_weight,
            "loss_cls": cls_scores[0].new_zeros([1])[0],
        }

        return loss_dict

    def _cls_loss(self, cls_score, proposal):
        labels = proposal.gt_classes
        if proposal.has("gt_loss_weights"):
            loss_weights = proposal.gt_loss_weights
        else:
            loss_weights = torch.zeros_like(labels)
        # loss_weights = 1.0

        B = cls_score.shape[0]
        C = cls_score.shape[1] - 1
        if self.use_sigmoid_ce:
            target = cls_score.new_zeros(B, C + 1)
            target[range(len(labels)), labels] = 1  # B x (C + 1)
            target = target[:, :C]  # B x C
            loss_region = torch.mean(F.binary_cross_entropy_with_logits(
                cls_score[:, :C], target, reduction='none').sum(-1) * loss_weights)
        else:
            loss_region = torch.mean(F.cross_entropy(
                cls_score[:, :C + 1], labels, reduction='none') * loss_weights)
        return loss_region

    def _caption_loss(self, score, classifier_info, idx, B):
        assert (classifier_info[2] is not None)
        assert self.add_image_box
        cls_and_cap_num = score.shape[1]
        cap_num = classifier_info[2].shape[0]
        score, caption_score = score.split(
            [cls_and_cap_num - cap_num, cap_num], dim=1)
        # n x (C + 1), n x B
        caption_score = caption_score[-1:]  # 1 x B # -1: image level box
        caption_target = caption_score.new_zeros(
            caption_score.shape)  # 1 x B or 1 x MB, M: num machines
        if self.sync_caption_batch:
            # caption_target: 1 x MB
            rank = comm.get_rank()
            global_idx = B * rank + idx
            assert (classifier_info[2][global_idx, -1] - rank) ** 2 < 1e-8, \
                '{} {} {} {} {}'.format(
                    rank, global_idx,
                    classifier_info[2][global_idx, -1],
                    classifier_info[2].shape,
                    classifier_info[2][:, -1])
            caption_target[:, global_idx] = 1.
        else:
            assert caption_score.shape[1] == B
            caption_target[:, idx] = 1.
        caption_loss_img = F.binary_cross_entropy_with_logits(
            caption_score, caption_target, reduction='none')

        if self.sync_caption_batch:
            fg_mask = (caption_target > 0.5).float()
            assert (fg_mask.sum().item() - 1.) ** 2 < 1e-8, '{} {}'.format(
                fg_mask.shape, fg_mask)
            pos_loss = (caption_loss_img * fg_mask).sum()
            neg_loss = (caption_loss_img * (1. - fg_mask)).sum()
            caption_loss_img = pos_loss + self.neg_cap_weight * neg_loss
        else:
            caption_loss_img = caption_loss_img.sum()

        return caption_loss_img