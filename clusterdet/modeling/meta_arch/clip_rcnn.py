# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
import itertools
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like, FrozenBatchNorm2d
from detectron2.structures import ImageList, Instances, BitMasks, ROIMasks
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
import detectron2.utils.comm as comm

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from torch.cuda.amp import autocast

from ..text_encoder.text_encoder import build_text_encoder
from ..utils.detic import load_class_freq, get_fed_loss_inds
from ..roi_heads.roi_heads_joint import build_recognition_heads

import clip


@META_ARCH_REGISTRY.register()
class CLIPRCNN(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        clip_model: nn.Module,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        pixel_norm: bool = True,
        input_format: Optional[str] = None,
        vis_period: int = 0,
        freeze_clip_visual=True,
        freeze_clip_text=True,
        freeze_at=2,
        seperate_backbone=False,
        with_image_labels=False,
        fp16=False,
        sync_caption_batch=False,
        roi_head_name='',
        cap_batch_ratio=4,
        with_caption=False,
        text_encoder_type="ViT-B/32",
        text_encoder_dim=512,
        is_grounding=True,
        dynamic_classifier=False,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.freeze_clip_visual = freeze_clip_visual
        self.freeze_clip_text = freeze_clip_text
        self.freeze_at = freeze_at
        self.initialize_clip_parameters()

        self.backbone = backbone
        self.seperate_backbone = seperate_backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.with_image_labels = with_image_labels
        self.fp16 = fp16
        self.with_caption = with_caption
        self.sync_caption_batch = sync_caption_batch
        self.roi_head_name = roi_head_name
        self.cap_batch_ratio = cap_batch_ratio
        self.text_encoder_dim = text_encoder_dim
        self.is_grounding = is_grounding

        self.dynamic_classifier = dynamic_classifier
        if self.dynamic_classifier:
            self.freq_weight = kwargs.pop('freq_weight')
            self.num_classes = kwargs.pop('num_classes')
            self.num_sample_cats = kwargs.pop('num_sample_cats')

        # if self.with_caption:
            # self.text_encoder = self.clip_model.text_encoder
        # assert not self.dynamic_classifier
        # self.text_encoder = build_text_encoder(
        #     pretrain=True, visual_type=text_encoder_type)
        # for v in self.text_encoder.parameters():
        #     v.requires_grad = False

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        self.pixel_norm = pixel_norm

    def initialize_clip_parameters(self):
        def cnnblockbase_freeze(nn_module):
            for p in nn_module.parameters():
                p.requires_grad = False
        clip_model_visual = self.clip_model.visual
        FrozenBatchNorm2d.convert_frozen_batchnorm(clip_model_visual) # TODO: make it flexible
        if self.freeze_clip_visual:
            cnnblockbase_freeze(clip_model_visual)
        else:
            # stem
            if self.freeze_at >= 1:
                cnnblockbase_freeze(clip_model_visual.conv1)
                cnnblockbase_freeze(clip_model_visual.bn1)
                cnnblockbase_freeze(clip_model_visual.conv2)
                cnnblockbase_freeze(clip_model_visual.bn2)
                cnnblockbase_freeze(clip_model_visual.conv3)
                cnnblockbase_freeze(clip_model_visual.bn3)
            # each stage is a torch.nn.modules.container.Sequential
            for idx, stage in enumerate([clip_model_visual.layer1,
                                         clip_model_visual.layer2,
                                         clip_model_visual.layer3,
                                         clip_model_visual.layer4], start=2):
                if self.freeze_at >= idx:
                    for block in stage.children():  # each block is a Bottleneck
                        cnnblockbase_freeze(block)
            # always fix attnpool
            cnnblockbase_freeze(clip_model_visual.attnpool)

        ###  freeze_clip_text
        if self.freeze_clip_text:
            clip_model = self.clip_model
            for v in list(itertools.chain(
                    clip_model.transformer.parameters(),
                    clip_model.token_embedding.parameters(),
                    clip_model.ln_final.parameters()
            )):
                v.requires_grad = False
            for v in [clip_model.positional_embedding,
                      clip_model.logit_scale,
                      clip_model.text_projection]:
                v.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        clip_type = cfg.MODEL.CLIP_TYPE
        assert 'RN50' in clip_type, 'current version only supports RN50'
        clip_model, _ = clip.load(clip_type, device='cpu')
        backbone = build_backbone(cfg)
        ret = {
            "clip_model": clip_model,
            "freeze_at": cfg.MODEL.BACKBONE.FREEZE_AT,
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "pixel_norm": cfg.MODEL.PIXEL_NORM,
        }

        ret.update({
            'freeze_clip_visual': cfg.MODEL.FREEZE_CLIP_VISUAL,
            'freeze_clip_text': cfg.MODEL.FREEZE_CLIP_TEXT,
            'seperate_backbone': cfg.MODEL.SEPERATE_BACKBONE,
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'fp16': cfg.FP16,
            'with_caption': cfg.MODEL.WITH_CAPTION,
            'sync_caption_batch': cfg.MODEL.SYNC_CAPTION_BATCH,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'roi_head_name': cfg.MODEL.ROI_HEADS.NAME,
            'cap_batch_ratio': cfg.MODEL.CAP_BATCH_RATIO,
            'text_encoder_type': cfg.MODEL.TEXT_ENCODER_TYPE,
            'text_encoder_dim': cfg.MODEL.TEXT_ENCODER_DIM,
            'is_grounding': cfg.MODEL.IS_GROUNDING,
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

    def clip_resnet_backbone(self, tensor):
        def stem(x):
            x = resnet.relu1(resnet.bn1(resnet.conv1(x)))
            x = resnet.relu2(resnet.bn2(resnet.conv2(x)))
            x = resnet.relu3(resnet.bn3(resnet.conv3(x)))
            x = resnet.avgpool(x)
            return x
        resnet = self.clip_model.visual

        tensor = tensor.type(resnet.conv1.weight.dtype)
        res1 = stem(tensor)
        res2 = resnet.layer1(res1)
        res3 = resnet.layer2(res2)
        res4 = resnet.layer3(res3)
        res5 = resnet.layer4(res4)

        return {"res2": res2, "res3": res3, "res4": res4, "res5": res5}

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        """
        if not self.training:
            return self.inference(batched_inputs)

        # annotation
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        for inst, x in zip(gt_instances, batched_inputs):
            inst._ann_type = x['ann_type'] if 'ann_type' in x else 'box'
            inst._pos_category_ids = x['pos_category_ids'] if 'pos_category_ids' in x else \
                inst.gt_classes.unique()
        ann_types = [inst._ann_type for inst in gt_instances]
        assert len(set(ann_types)) == 1
        ann_type = ann_types[0]

        # image
        images = self.preprocess_image_clip(batched_inputs)

        # clip visual features
        if self.fp16:
            with autocast():
                backbone_features = self.clip_resnet_backbone(images.tensor.half())
            backbone_features = {k: v.float() for k, v in backbone_features.items()}
        else:
            backbone_features = self.clip_resnet_backbone(images.tensor)

        # fpn features
        fpn_features = self.backbone(images.tensor, backbone_features)

        # region proposal
        proposals, proposal_losses = self.proposal_generator(images, fpn_features, gt_instances)

        with torch.no_grad():
            cls_features, cls_inds, caption_features, nouns_num = None, None, None, None
            if self.with_caption and 'caption' in ann_type:
                inds = [torch.randint(len(x['captions']), (1,))[0].item() for x in batched_inputs]
                caps = [x['captions'][ind] for ind, x in zip(inds, batched_inputs)]
                caption_tokens = clip.tokenize(caps).to(self.device)
                caption_features = self.clip_model.encode_text(caption_tokens)

                # nouns = ['a ' + noun for x in batched_inputs for noun in x['nouns']]
                # nouns_num = [len(x['nouns']) for x in batched_inputs]
                # nouns_tokens = clip.tokenize(nouns).to(self.device)
                nouns_num = [len(inst._pos_category_ids) for inst in gt_instances]
                nouns_idx = [pos_cat_id for inst in gt_instances for pos_cat_id in inst._pos_category_ids]
                cls_features = self.roi_heads.classification_predictor.cls_score.nouns_weight[
                    :, nouns_idx].permute(1, 0).contiguous()

            if self.sync_caption_batch:
                caption_features = self._sync_caption_features(
                    caption_features, ann_type, len(batched_inputs))

            if self.dynamic_classifier and ann_type != 'caption':
                cls_inds = self._sample_cls_inds(gt_instances, ann_type)  # inds, inv_inds
                ind_with_bg = cls_inds[0].tolist() + [-1]
                cls_features = self.roi_heads.box_predictor[
                    0].cls_score.zs_weight[:, ind_with_bg].permute(1,0).contiguous()
            classifier_info = cls_features, cls_inds, caption_features, nouns_num

        # roi head
        proposals, detector_losses = self.roi_heads(
            images, fpn_features, proposals, gt_instances,
            ann_type=ann_type, classifier_info=classifier_info,
        )

        # record losses
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def _sync_caption_features(self, caption_features, ann_type, BS):
        has_caption_feature = (caption_features is not None)
        BS = (BS * self.cap_batch_ratio) if (ann_type == 'box') else BS
        rank = torch.full(
            (BS, 1), comm.get_rank(), dtype=torch.float32,
            device=self.device)
        if not has_caption_feature:
            caption_features = rank.new_zeros((BS, self.text_encoder_dim))
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


    def preprocess_image_clip(self, batched_inputs):
        original_images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        if self.input_format == 'BGR':
            original_images = [ori_img[[2, 1, 0], :, :] for ori_img in original_images]
        if self.pixel_norm:
            original_images = [x / 255.0 for x in original_images]
        images = [(x - self.pixel_mean) / self.pixel_std for x in original_images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images


    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        assert not self.training
        images = self.preprocess_image_clip(batched_inputs)
        backbone_features = self.clip_resnet_backbone(images.tensor)
        fpn_features = self.backbone(images.tensor, backbone_features)
        proposals, _ = self.proposal_generator(images, fpn_features)
        results, _ = self.roi_heads(images, fpn_features, proposals)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            max_shape = images.tensor.shape[2:]
            return CLIPRCNN._postprocess(results, batched_inputs, images.image_sizes, max_shape)
        else:
            return results

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
