# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import math
import copy
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

from ..utils.detic import load_class_freq, get_fed_loss_inds
from .zero_shot_classifier import ZeroShotClassifier
from .zero_shot_classifier_group import ZeroShotClassifierGroup
from ..utils.part import load_obj2part_mapping
from .meta_part_classifier import MetaPartClassifier


class BoxFastRCNNOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        use_fed_loss=False,
        box_score_reg_loss=False,
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            use_fed_loss=False,
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
            'use_fed_loss': cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS,
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
        # metapart_cls=None,
        zeroshot_cls=None,
        sync_caption_batch=False,
        use_sigmoid_ce=False,
        use_fed_loss=False,
        ignore_zero_cats=False,
        fed_loss_num_cat=50,
        dynamic_classifier=False,
        use_zeroshot_cls=False,
        use_zeroshot_cls_group=False,
        with_caption_loss=False,
        caption_loss_weight=1.0,
        neg_cap_weight=1.0,
        nouns_loss_weight=0.01,
        add_image_box=False,
        box2part_loss_weight=0.1,
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
        self.use_zeroshot_cls_group = use_zeroshot_cls_group
        self.with_caption_loss = with_caption_loss
        self.caption_loss_weight = caption_loss_weight
        self.neg_cap_weight = neg_cap_weight
        self.nouns_loss_weight = nouns_loss_weight
        self.add_image_box = add_image_box
        self.box2part_loss_weight = box2part_loss_weight

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
        if self.use_zeroshot_cls_group:
            self.cls_score = zeroshot_cls
        elif self.use_zeroshot_cls:
            assert zeroshot_cls is not None
            self.cls_score = zeroshot_cls
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

        # self.metapart_score = metapart_cls
        self.obj2part_mapping = load_obj2part_mapping(
            'datasets/metadata/pascal_obj2part_mapping.json'
        )

        part2metapart_mapping_dict = load_obj2part_mapping(
            'datasets/metadata/pascal_part_base2metapart_mapping.json'
        )
        part2metapart_mapping = torch.zeros((58,))
        for part_idx in part2metapart_mapping_dict:
            part2metapart_mapping[int(part_idx)] = part2metapart_mapping_dict[part_idx]
        part2metapart_mapping[57] = 30
        self.part2metapart_mapping = part2metapart_mapping


        part2metapart_mapping_dict2 = load_obj2part_mapping('datasets/metadata/pascal_part2metapart_mapping.json')
        part2metapart_mapping2 = torch.zeros((93,))
        for part_idx in part2metapart_mapping_dict2:
            part2metapart_mapping2[int(part_idx)] = part2metapart_mapping_dict2[part_idx]
        self.part2metapart_mapping2 = part2metapart_mapping2

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
            'use_zeroshot_cls_group': cfg.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS_GROUP,
            'with_caption_loss': cfg.MODEL.ROI_BOX_HEAD.WITH_CAPTION_LOSS,
            'caption_loss_weight': cfg.MODEL.ROI_BOX_HEAD.CAPTION_LOSS_WEIGHT,
            'neg_cap_weight': cfg.MODEL.ROI_BOX_HEAD.NEG_CAP_WEIGHT,
            'nouns_loss_weight': cfg.MODEL.ROI_BOX_HEAD.NOUNS_LOSS_WEIGHT,
            'add_image_box': cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX,
            'box2part_loss_weight': cfg.MODEL.ROI_BOX_HEAD.BOX2PART_LOSS_WEIGHT,
            'prior_prob': cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB,
            'cat_freq_path': cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
            'fed_loss_freq_weight': cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT,
        })
        if ret['use_zeroshot_cls_group']:
            ret['zeroshot_cls'] = ZeroShotClassifierGroup(cfg, input_shape)
        elif ret['use_zeroshot_cls']:
            ret['zeroshot_cls'] = ZeroShotClassifier(cfg, input_shape)

        # ret['metapart_cls'] = MetaPartClassifier(cfg, input_shape)
        return ret

    def forward(self, x, classifier_info=(None, None, None, None), dataset_source=None):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        cls_scores = []
        if classifier_info[0] is not None:
            cls_scores.append(self.cls_score(x, classifier=classifier_info[0]))
        else:
            if self.use_zeroshot_cls_group:
                cls_scores.append(self.cls_score(x, dataset_source=dataset_source))
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

    def classification_losses(self, predictions, proposals):
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
            # "loss_part_meta": scores.new_zeros([1])[0],
            # "loss_part_pseudo": scores.new_zeros([1])[0],
            "loss_ppart": scores.new_zeros([1])[0],
            'loss_noun': scores.new_zeros([1])[0],
            'loss_caption': scores.new_zeros([1])[0],
        }
        return loss_dict

    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]  # This is more robust than .sum() * 0.

        B = pred_class_logits.shape[0]
        C = pred_class_logits.shape[1] - 1
        gt_classes = torch.clamp(gt_classes, max=C)  # multi-dataset train

        target = pred_class_logits.new_zeros(B, C + 1)
        target[range(len(gt_classes)), gt_classes] = 1  # B x (C + 1)
        target = target[:, :C]  # B x C

        weight = 1
        if self.use_fed_loss and (self.freq_weight is not None):  # fedloss
            appeared = get_fed_loss_inds(
                gt_classes,
                num_sample_cats=self.fed_loss_num_cat,
                C=C,
                weight=self.freq_weight)
            appeared_mask = appeared.new_zeros(C + 1)
            appeared_mask[appeared] = 1  # C + 1
            appeared_mask = appeared_mask[:C]
            fed_w = appeared_mask.view(1, C).expand(B, C)
            weight = weight * fed_w.float()

        if self.ignore_zero_cats and (self.freq_weight is not None):
            w = (self.freq_weight.view(-1) > 1e-4).float()
            weight = weight * w.view(1, C).expand(B, C)

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits[:, :-1], target, reduction='none')  # B x C
        loss = torch.sum(cls_loss * weight) / B
        return loss

    def softmax_cross_entropy_loss(self, pred_class_logits, gt_classes):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]
        C = pred_class_logits.shape[1] - 1
        gt_classes = torch.clamp(gt_classes, max=C)  # multi-dataset train
        loss = F.cross_entropy(
            pred_class_logits, gt_classes, reduction="mean")
        return loss

    def forward_meta_part(self, x, classifier_info=(None, None, None, None), dataset_source=None):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        cls_scores = self.metapart_score(x)

        return cls_scores

    def meta_part_losses(self, meta_scores, proposals, scores):
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        gt_classes = torch.clamp(gt_classes, max=scores.shape[1] - 1)
        metapart_gt_classes = self.part2metapart_mapping[gt_classes].long()

        loss_part_meta = self.sigmoid_cross_entropy_loss(meta_scores, metapart_gt_classes)

        return loss_part_meta

    def box2part_losses(self, part_predictions, proposals, targets, classifier_info=(None, None, None), ):
        num_inst_per_image = [len(p) for p in proposals]
        part_scores = part_predictions.split(num_inst_per_image, dim=0)

        B = len(part_scores)
        part_loss = part_scores[0].new_zeros([1])[0]

        for idx, (part_score, proposal, target) in enumerate(zip(part_scores, proposals, targets)):
            if part_score.shape[0] == 0:
                continue
            gt_boxes, gt_classes = target.gt_boxes, target.gt_classes
            for obj_box, obj_cat in zip(gt_boxes, gt_classes):
                if str(obj_cat.item()) not in self.obj2part_mapping:
                    continue
                part_cat_idx = self.obj2part_mapping[str(obj_cat.item())]
                loss_part = self._box2part_loss(part_score, part_cat_idx, classifier_info)
                part_loss += loss_part

            # part_target_ind = self.obj2part_ind(target)
            # if len(part_target_ind) < 1:
            #     continue
            # loss_part = self._box2part_loss(cls_score, part_target_ind, classifier_info)
            #
            # part_loss += loss_part

        part_loss = part_loss / B
        loss_dict = {
            # "loss_part_pseudo": part_loss * self.box2part_loss_weight,
            # "loss_part_meta": part_scores[0].new_zeros([1])[0],
            "loss_ppart": part_loss * self.box2part_loss_weight,
            "loss_cls": part_scores[0].new_zeros([1])[0],
            'loss_noun': part_scores[0].new_zeros([1])[0],
            'loss_caption': part_scores[0].new_zeros([1])[0],
        }

        return loss_dict


    def _box2part_loss(self, score, part_target_ind, classifier_info):
        import pdb
        pdb.set_trace()

        proxy_ind = self.part2proxy_ind(part_target_ind)
        proxy_score = score[:, proxy_ind]
        part_pro_ind = proxy_score.argmax(dim=0)

        score_cur_img = score[part_pro_ind]
        score_target = score_cur_img.new_zeros(score_cur_img.shape)
        score_target[torch.arange(len(part_pro_ind)), part_target_ind] = 1.
        score_loss_img = F.binary_cross_entropy_with_logits(
            score_cur_img, score_target, reduction='none')
        score_loss_img = score_loss_img.sum()

        return score_loss_img


    def part2proxy_ind(self, part_target_ind):
        proxy_idxs = []
        dog2cat = {48:30, 49:31, 50:32, 51:33, 52:34, 53:35, 54:36, 56:37, 57:38,}
        for part_idx in part_target_ind:
            if part_idx in dog2cat:
                proxy_idxs.append(dog2cat[part_idx])
            else:
                proxy_idxs.append(part_idx)
        return proxy_idxs


    def obj2part_ind(self, target):
        box_idxs = target.gt_classes.unique()
        part_idxs = []
        for box_idx in box_idxs.cpu().tolist():
            if str(box_idx) in self.obj2part_mapping:
                part_idx = self.obj2part_mapping[str(box_idx)]
                part_idxs.extend(part_idx)
        return part_idxs


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

            loss_cls = self._noun_loss(cls_score, classifier_info, idx, B)
            cls_loss += loss_cls

            if self.with_caption_loss:
                loss_caption = self._caption_loss(cls_score, classifier_info, idx, B)
                caption_loss += loss_caption

        cls_loss = cls_loss / B
        caption_loss = caption_loss / B

        loss_dict = {
            'loss_noun': cls_loss * self.nouns_loss_weight,
            'loss_caption': caption_loss * self.caption_loss_weight,
            'loss_cls': cls_scores[0].new_zeros([1])[0],
            'loss_ppart': cls_scores[0].new_zeros([1])[0],
        }

        return loss_dict


    def _noun_loss(self, score, classifier_info, idx, B):
        assert (classifier_info[0] is not None)
        cls_and_cap_num = score.shape[1]
        cls_num = classifier_info[0].shape[0]
        score, caption_score = score.split([cls_num, cls_and_cap_num - cls_num], dim=1)
        # n x (C + 1), n x B
        score = score[:-1]  # n x C
        assert (classifier_info[3] is not None)
        score_per_img = score.split(classifier_info[3], dim=1)
        score_cur_img = score_per_img[idx]
        num_pro, num_noun = score_cur_img.shape
        score_target = score_cur_img.new_zeros(score_cur_img.shape)
        score_target[score_cur_img.argmax(dim=0), torch.arange(num_noun)] = 1.
        score_loss_img = F.binary_cross_entropy_with_logits(
            score_cur_img, score_target, reduction='none')
        score_loss_img = score_loss_img.sum()

        return score_loss_img


    def _caption_loss(self, score, classifier_info, idx, B):
        assert (classifier_info[2] is not None)
        assert self.add_image_box
        cls_and_cap_num = score.shape[1]
        cap_num = classifier_info[2].shape[0]
        score, caption_score = score.split([cls_and_cap_num - cap_num, cap_num], dim=1)
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

