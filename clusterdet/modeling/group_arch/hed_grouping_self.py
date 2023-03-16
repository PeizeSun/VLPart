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
from detectron2.layers import ShapeSpec, batched_nms, cat, nonzero_tuple, Conv2d, ConvTranspose2d
from detectron2.structures import ImageList, Instances, BitMasks, ROIMasks
import detectron2.utils.comm as comm

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, ROIMasks, pairwise_iou
from detectron2.utils.memory import retry_if_cuda_oom

from ..utils.adelaidet import aligned_bilinear, unfold_wo_center, mask_to_box
from .densecl_grouping import DenseCL_Grouping

from . import higra_self


@META_ARCH_REGISTRY.register()
class HED_Grouping_self(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        device: str = 'cuda',
        input_format: Optional[str] = None,
        vis_period: int = 0,
        num_classes: int = 80,
        size_divisibility: int = 0,
        edge_thresholds: list = [0.4, 0.6, 0.8],
        box_area_thresh: float = 100.0,
        box_output_stride: int = 4,
        box_to_image_ratio_thresh: float = 0.9,
        box_score_thresh: float = 0.1,
        box_iou_thresh: float = 0.5,
        box_per_level: int = 100,
        iou_by_gt_thresh: float = 0.5,
        box_topk: int = 3,
        hierarchy_group: bool = False,
        eval_grouping: bool = False,
        edge_levels: list = [-3, -2, -1],
        proposal_nums: list = [2, 4, 8, 32],
        down_sample_ratio: int = 2,
        ann_generator: bool = False,
    ):
        super().__init__()

        # Layers.
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=35)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.relu = nn.ReLU()
        # Note: ceil_mode – when True, will use ceil instead of floor to compute the output shape.
        #       The reason to use ceil mode here is that later we need to upsample the feature maps and crop the results
        #       in order to have the same shape as the original image. If ceil mode is not used, the up-sampled feature
        #       maps will possibly be smaller than the original images.
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.score_dsn1 = nn.Conv2d(64, 1, 1)  # Out channels: 1.
        self.score_dsn2 = nn.Conv2d(128, 1, 1)
        self.score_dsn3 = nn.Conv2d(256, 1, 1)
        self.score_dsn4 = nn.Conv2d(512, 1, 1)
        self.score_dsn5 = nn.Conv2d(512, 1, 1)
        self.score_final = nn.Conv2d(5, 1, 1)

        # Fixed bilinear weights.
        self.weight_deconv2 = make_bilinear_weights(4, 1).to(device)
        self.weight_deconv3 = make_bilinear_weights(8, 1).to(device)
        self.weight_deconv4 = make_bilinear_weights(16, 1).to(device)
        self.weight_deconv5 = make_bilinear_weights(32, 1).to(device)

        # Prepare for aligned crop.
        self.crop1_margin, self.crop2_margin, self.crop3_margin, self.crop4_margin, self.crop5_margin = \
            prepare_aligned_crop()

        self.input_format = input_format
        self.vis_period = vis_period
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.num_classes = num_classes
        self.size_divisibility = size_divisibility

        self.edge_thresholds = edge_thresholds
        self.box_area_thresh = box_area_thresh
        self.box_output_stride = box_output_stride
        self.box_to_image_ratio_thresh = box_to_image_ratio_thresh
        self.box_score_thresh = box_score_thresh
        self.box_iou_thresh = box_iou_thresh
        self.box_per_level = box_per_level
        self.iou_by_gt_thresh = iou_by_gt_thresh
        self.box_topk = box_topk
        self.hierarchy_group = hierarchy_group
        self.eval_grouping = eval_grouping
        self.edge_levels = edge_levels
        self.proposal_nums = proposal_nums
        self.down_sample_ratio = down_sample_ratio
        self.ann_generator = ann_generator

    @classmethod
    def from_config(cls, cfg):
        ret = {
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "device": cfg.MODEL.DEVICE,
        }
        ret.update({
            "num_classes": cfg.MODEL.GROUP_HEADS.NUM_CLASSES,
            "size_divisibility": cfg.MODEL.GROUP_HEADS.SIZE_DIVISIBILITY,
            "edge_thresholds": cfg.MODEL.GROUP_HEADS.EDGE_THRESHOLDS,
            "box_area_thresh": cfg.MODEL.GROUP_HEADS.BOX_AREA_THRESH,
            "box_output_stride": cfg.MODEL.GROUP_HEADS.BOX_OUTPUT_STRIDE,
            "box_to_image_ratio_thresh": cfg.MODEL.GROUP_HEADS.BOX_TO_IMAGE_RATIO_THRESH,
            "box_score_thresh": cfg.MODEL.GROUP_HEADS.BOX_SCORE_THRESH,
            "box_iou_thresh": cfg.MODEL.GROUP_HEADS.BOX_IOU_THRESH,
            "box_per_level": cfg.MODEL.GROUP_HEADS.BOX_PER_LEVEL,
            "iou_by_gt_thresh": cfg.MODEL.GROUP_HEADS.IOU_BY_GT_THRESH,
            "box_topk": cfg.MODEL.GROUP_HEADS.BOX_TOPK,
            "hierarchy_group": cfg.MODEL.GROUP_HEADS.HIERARCHY_GROUP,
            "eval_grouping": cfg.MODEL.GROUP_HEADS.EVALUATE,
            "edge_levels": cfg.MODEL.GROUP_HEADS.EDGE_LEVELS,
            "proposal_nums": cfg.MODEL.GROUP_HEADS.PROPOSAL_NUMS,
            "down_sample_ratio": cfg.MODEL.GROUP_HEADS.DOWN_SAMPLE_RATIO,
            "ann_generator": cfg.MODEL.ANN_GENERATOR,
        })

        return ret

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs, do_postprocess=True):
        assert not self.training, 'HED_Grouping only supports inference mode'
        assert len(batched_inputs) == 1, "HED_Grouping only supports one image in one batch"

        imageList = self.preprocess_image(batched_inputs)
        hed_results = self.forward_hed(batched_inputs, imageList.tensor)

        if self.vis_period:
            return self.visualize_hed(batched_inputs, hed_results)

        levels_proposals = []
        for edge_level in self.edge_levels:
            proposals = self.edge_to_boxes(
                batched_inputs,
                1.0 - hed_results[edge_level],
                imageList.image_sizes,
            )
            levels_proposals.append(proposals)
        grouping_proposals = self.merge_levels(levels_proposals, imageList.image_sizes)

        if self.ann_generator:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        pseudo_targets = []
        for image_idx in range(len(batched_inputs)):
            pseudo_target = Instances(imageList.image_sizes[image_idx])
            pseudo_boxes = grouping_proposals[image_idx].proposal_boxes.tensor
            pseudo_scores = grouping_proposals[image_idx].scores

            # select topk pseudo_instances
            _, idx = pseudo_scores.sort(descending=True)
            pseudo_boxes = pseudo_boxes[idx[:self.box_topk]]
            pseudo_scores = pseudo_scores[idx[:self.box_topk]]

            pseudo_target.pred_boxes = Boxes(pseudo_boxes)
            pseudo_target.pred_classes = torch.zeros_like(pseudo_scores, dtype=torch.int64)
            pseudo_target.scores = pseudo_scores.float()

            if self.ann_generator:
                pos_category_ids = gt_instances[image_idx].gt_classes.unique()
                pseudo_target.pos_category_ids = pos_category_ids[None].repeat(pseudo_scores.shape[0], 1)

            pseudo_targets.append(pseudo_target)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            max_shape = imageList.tensor.shape[2:]
            return DenseCL_Grouping._postprocess(pseudo_targets, batched_inputs, imageList.image_sizes, max_shape)
        else:
            return results

    def mask_scoring_func(self, ori_edge_maps, image_idx, pseudo_masks):
        soft_masks = ori_edge_maps[image_idx] * pseudo_masks
        inner_scores = soft_masks.sum((1, 2)).divide_(pseudo_masks.sum((1, 2)))
        avg_mask = F.avg_pool2d(pseudo_masks[None].float(), (3, 3), stride=1, padding=1)[0]
        boundary = (avg_mask != pseudo_masks)
        soft_boundary = ori_edge_maps[image_idx] * boundary
        outer_scores = soft_boundary.sum((1, 2)).divide_(boundary.sum((1, 2)))
        pseudo_scores = (inner_scores - outer_scores + 1.0) / 2.0
        return pseudo_scores

    def edge_to_boxes(self, batched_inputs, ori_edge_maps, image_sizes):
        pseudo_targets = []
        fast_ratio = self.down_sample_ratio
        mask_h, mask_w = ori_edge_maps.shape[-2:]
        edge_maps = F.interpolate(ori_edge_maps, size=(mask_h // fast_ratio, mask_w // fast_ratio),
                                  mode='bilinear', align_corners=False)
        box_area_thresh = self.box_area_thresh * self.box_output_stride ** 2 // fast_ratio // fast_ratio
        length_ratio = self.box_to_image_ratio_thresh
        for image_idx in range(len(edge_maps)):
            pseudo_target = Instances(image_sizes[image_idx])

            edge_map_cur_image = edge_maps[image_idx][0]
            binary_edge_map = (edge_map_cur_image > 0.9).float()
            edge_map = torch.ones_like(edge_map_cur_image) * binary_edge_map + \
                       edge_map_cur_image * (1.0 - binary_edge_map)

            # In gradient map, object boundary is light (tend to 1)
            gradient = (1.0 - edge_map).cpu().numpy()
            height, width = gradient.shape[:2]
            import time
            stime = time.time()
            _graph = higra_self.build_graph(gradient, width, height)
            graph = higra_self.Graph(_graph, shape=(height, width))
            print('build time:', time.time() - stime)

            stime = time.time()
            # tree, altitudes = higra_self.watershed_hierarchy_by_area(graph)
            areas, boxes = higra_self.watershed_hierarchy_by_area(graph)

            boxes = boxes[areas > box_area_thresh]
            boxes = torch.tensor(boxes, device=edge_map.device)

            width_boxes, height_boxes = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
            cx_boxes, cy_boxes = (boxes[:, 2] + boxes[:, 0]) / 2, (boxes[:, 3] + boxes[:, 1]) / 2
            image_h, image_w = image_sizes[image_idx]
            image_h, image_w = image_h // fast_ratio, image_w // fast_ratio
            keep1 = ((width_boxes < image_w * length_ratio) + (height_boxes < image_h * length_ratio)) > 0
            keep2 = (cx_boxes > image_w * (1 - length_ratio)) * (cx_boxes < image_w * length_ratio)
            keep3 = (cy_boxes > image_h * (1 - length_ratio)) * (cy_boxes < image_h * length_ratio)
            keep = keep1 * keep2 * keep3
            if keep.sum() < 1:
                pseudo_target.proposal_boxes = Boxes(edge_map.new_zeros((0, 4)))
                pseudo_target.scores = edge_map.new_ones((0,))
                pseudo_targets.append(pseudo_target)
                continue

            pseudo_boxes = boxes[keep]
            pseudo_scores = torch.ones_like(boxes[:, 0])

            # nms between different hierarchical level outputs
            keep = batched_nms(pseudo_boxes, pseudo_scores, torch.zeros_like(pseudo_scores), self.box_iou_thresh)
            pseudo_boxes = pseudo_boxes[keep]
            pseudo_scores = pseudo_scores[keep]

            # select topk pseudo_instances
            _, idx = pseudo_scores.sort(descending=True)
            pseudo_boxes = pseudo_boxes[idx[:self.box_per_level]]
            pseudo_scores = pseudo_scores[idx[:self.box_per_level]]

            pseudo_target.proposal_boxes = Boxes(pseudo_boxes * fast_ratio)
            pseudo_target.scores = pseudo_scores

            pseudo_targets.append(pseudo_target)

        return pseudo_targets

        #
        #
        #
        #     print('tree time:', time.time() - stime)
        #
        #     stime = time.time()
        #     cut_helper = higra_self.HorizontalCutExplorer(tree, altitudes)
        #     print('cut_helper time:', time.time() - stime)
        #     pseudo_boxes, pseudo_masks, pseudo_scores = [], [], []
        #     stime = time.time()
        #     for iter_idx, num_pro in enumerate(self.proposal_nums):
        #         cut = cut_helper.horizontal_cut_from_num_regions(num_pro)
        #         cc_out = torch.tensor(cut.labelisation_leaves(tree), device=edge_map.device)
        #         print('mask{} time:'.format(iter_idx), time.time() - stime)
        #         stime = time.time()
        #         cc_out = cc_out.reshape(height, width)
        #
        #         cc_labels = torch.unique(cc_out)
        #         if len(cc_labels) < 2:
        #             continue
        #         masks = (cc_out[None, :, :] == cc_labels[:, None, None])
        #         masks = masks[masks.sum((1, 2)) > box_area_thresh]
        #         boxes = mask_to_box(masks)
        #
        #         width_boxes, height_boxes = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        #         cx_boxes, cy_boxes = (boxes[:, 2] + boxes[:, 0]) / 2, (boxes[:, 3] + boxes[:, 1]) / 2
        #         image_h, image_w = image_sizes[image_idx]
        #         image_h, image_w = image_h // fast_ratio, image_w // fast_ratio
        #         keep1 = ((width_boxes < image_w * length_ratio) + (height_boxes < image_h * length_ratio)) > 0
        #         keep2 = (cx_boxes > image_w * (1 - length_ratio)) * (cx_boxes < image_w * length_ratio)
        #         keep3 = (cy_boxes > image_h * (1 - length_ratio)) * (cy_boxes < image_h * length_ratio)
        #         keep = keep1 * keep2 * keep3
        #         if keep.sum() < 1:
        #             continue
        #         boxes = boxes[keep]
        #         masks = masks[keep]
        #
        #         pseudo_boxes.append(boxes)
        #         pseudo_masks.append(masks)
        #
        #     if len(pseudo_boxes) < 1:
        #         pseudo_target.proposal_boxes = Boxes(edge_map.new_zeros((0, 4)))
        #         pseudo_target.proposal_masks = edge_map.new_zeros((0, mask_h, mask_w))
        #         pseudo_target.scores = edge_map.new_ones((0,))
        #         pseudo_targets.append(pseudo_target)
        #         continue
        #
        #     pseudo_boxes = torch.cat(pseudo_boxes)
        #     pseudo_masks = torch.cat(pseudo_masks)
        #
        #     pseudo_masks = F.interpolate(pseudo_masks[None].float(), size=(mask_h, mask_w),
        #                                  mode='bilinear', align_corners=False)[0] > 0.5
        #
        #     pseudo_scores = retry_if_cuda_oom(self.mask_scoring_func)(ori_edge_maps, image_idx, pseudo_masks)
        #     pseudo_scores = pseudo_scores.to(pseudo_boxes)
        #
        #     keep = pseudo_scores > self.box_score_thresh
        #     if keep.sum() < 1:
        #         pseudo_target.proposal_boxes = Boxes(edge_map.new_zeros((0, 4)))
        #         pseudo_target.proposal_masks = edge_map.new_zeros((0, mask_h, mask_w))
        #         pseudo_target.scores = edge_map.new_ones((0,))
        #         pseudo_targets.append(pseudo_target)
        #         continue
        #     pseudo_boxes = pseudo_boxes[keep]
        #     pseudo_masks = pseudo_masks[keep]
        #     pseudo_scores = pseudo_scores[keep]
        #
        #     # nms between different hierarchical level outputs
        #     keep = batched_nms(pseudo_boxes, pseudo_scores, torch.zeros_like(pseudo_scores), self.box_iou_thresh)
        #     pseudo_boxes = pseudo_boxes[keep]
        #     pseudo_scores = pseudo_scores[keep]
        #     pseudo_masks = pseudo_masks[keep]
        #
        #     # select topk pseudo_instances
        #     _, idx = pseudo_scores.sort(descending=True)
        #     pseudo_boxes = pseudo_boxes[idx[:self.box_per_level]]
        #     pseudo_masks = pseudo_masks[idx[:self.box_per_level]]
        #     pseudo_scores = pseudo_scores[idx[:self.box_per_level]]
        #
        #     pseudo_target.proposal_boxes = Boxes(pseudo_boxes * fast_ratio)
        #     pseudo_target.proposal_masks = pseudo_masks
        #     pseudo_target.scores = pseudo_scores
        #
        #     pseudo_targets.append(pseudo_target)
        #
        # return pseudo_targets

    def merge_levels(self, levels_proposals, image_sizes):
        num_levels = len(levels_proposals)
        num_imgs = len(levels_proposals[0])

        grouping_proposals = []
        for image_idx in range(num_imgs):
            pseudo_target = Instances(image_sizes[image_idx])

            pseudo_boxes, pseudo_masks, pseudo_scores = [], [], []
            for level_idx in range(num_levels):
                proposals = levels_proposals[level_idx][image_idx]
                pseudo_boxes.append(proposals.proposal_boxes.tensor)
                pseudo_scores.append(proposals.scores)
            pseudo_boxes = torch.cat(pseudo_boxes)
            pseudo_scores = torch.cat(pseudo_scores)

            # nms between outputs from different level of HED
            keep = batched_nms(pseudo_boxes, pseudo_scores, torch.zeros_like(pseudo_scores), self.box_iou_thresh)
            pseudo_boxes = pseudo_boxes[keep]
            pseudo_scores = pseudo_scores[keep]

            pseudo_target.proposal_boxes = Boxes(pseudo_boxes)
            pseudo_target.scores = pseudo_scores

            grouping_proposals.append(pseudo_target)

        return grouping_proposals

    def forward_hed(self, batched_inputs, x):
        # VGG-16 network.
        image_h, image_w = x.shape[2], x.shape[3]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))  # Side output 1.
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))  # Side output 2.
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))  # Side output 3.
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))  # Side output 4.
        pool4 = self.maxpool(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))  # Side output 5.

        score_dsn1 = self.score_dsn1(conv1_2)
        score_dsn2 = self.score_dsn2(conv2_2)
        score_dsn3 = self.score_dsn3(conv3_3)
        score_dsn4 = self.score_dsn4(conv4_3)
        score_dsn5 = self.score_dsn5(conv5_3)

        upsample2 = torch.nn.functional.conv_transpose2d(score_dsn2, self.weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(score_dsn3, self.weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(score_dsn4, self.weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(score_dsn5, self.weight_deconv5, stride=16)

        # Aligned cropping.
        crop1 = score_dsn1[:, :, self.crop1_margin:self.crop1_margin + image_h,
                self.crop1_margin:self.crop1_margin + image_w]
        crop2 = upsample2[:, :, self.crop2_margin:self.crop2_margin + image_h,
                self.crop2_margin:self.crop2_margin + image_w]
        crop3 = upsample3[:, :, self.crop3_margin:self.crop3_margin + image_h,
                self.crop3_margin:self.crop3_margin + image_w]
        crop4 = upsample4[:, :, self.crop4_margin:self.crop4_margin + image_h,
                self.crop4_margin:self.crop4_margin + image_w]
        crop5 = upsample5[:, :, self.crop5_margin:self.crop5_margin + image_h,
                self.crop5_margin:self.crop5_margin + image_w]

        # Concatenate according to channels.
        fuse_cat = torch.cat((crop1, crop2, crop3, crop4, crop5), dim=1)
        fuse = self.score_final(fuse_cat)  # Shape: [batch_size, 1, image_h, image_w].
        results = [crop1, crop2, crop3, crop4, crop5, fuse]
        results = [torch.sigmoid(r) for r in results]

        # if self.vis_period:
        #     return self.visualize_hed(batched_inputs, results)
        return results

    def visualize_hed(self, batched_inputs, results):
        import cv2
        import os
        import time
        image_show = batched_inputs[0]["image"].permute(1, 2, 0).numpy()
        dsn1 = (1.0 - results[0][0][0])[:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255
        dsn2 = (1.0 - results[1][0][0])[:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255
        dsn3 = (1.0 - results[2][0][0])[:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255
        dsn4 = (1.0 - results[3][0][0])[:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255
        dsn5 = (1.0 - results[4][0][0])[:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255
        fuse = (1.0 - results[5][0][0])[:, :, None].repeat(1, 1, 3).detach().cpu().numpy() * 255

        row1 = cv2.hconcat(
            [image_show.astype(np.uint8), dsn1.astype(np.uint8), dsn2.astype(np.uint8), dsn3.astype(np.uint8)])
        row2 = cv2.hconcat(
            [image_show.astype(np.uint8), dsn4.astype(np.uint8), dsn5.astype(np.uint8), fuse.astype(np.uint8)])
        show = cv2.vconcat([row1, row2])

        os.makedirs('output_edge_map', exist_ok=True)
        cv2.imwrite('output_edge_map/show{}.png'.format(time.time()), show)

        return [{}]

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        original_images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in original_images]
        images = ImageList.from_tensors(
            images,
            self.size_divisibility,
        )
        return images

def make_bilinear_weights(size, num_channels):
    """ Generate bi-linear interpolation weights as up-sampling filters (following FCN paper). """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False  # Set not trainable.
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


# noinspection PyMethodMayBeStatic
def prepare_aligned_crop():
    """ Prepare for aligned crop. """

    # Re-implement the logic in deploy.prototxt and
    #   /hed/src/caffe/layers/crop_layer.cpp of official repo.
    # Other reference materials:
    #   hed/include/caffe/layer.hpp
    #   hed/include/caffe/vision_layers.hpp
    #   hed/include/caffe/util/coords.hpp
    #   https://groups.google.com/forum/#!topic/caffe-users/YSRYy7Nd9J8

    def map_inv(m):
        """ Mapping inverse. """
        a, b = m
        return 1 / a, -b / a

    def map_compose(m1, m2):
        """ Mapping compose. """
        a1, b1 = m1
        a2, b2 = m2
        return a1 * a2, a1 * b2 + b1

    def deconv_map(kernel_h, stride_h, pad_h):
        """ Deconvolution coordinates mapping. """
        return stride_h, (kernel_h - 1) / 2 - pad_h

    def conv_map(kernel_h, stride_h, pad_h):
        """ Convolution coordinates mapping. """
        return map_inv(deconv_map(kernel_h, stride_h, pad_h))

    def pool_map(kernel_h, stride_h, pad_h):
        """ Pooling coordinates mapping. """
        return conv_map(kernel_h, stride_h, pad_h)

    x_map = (1, 0)
    conv1_1_map = map_compose(conv_map(3, 1, 35), x_map)
    conv1_2_map = map_compose(conv_map(3, 1, 1), conv1_1_map)
    pool1_map = map_compose(pool_map(2, 2, 0), conv1_2_map)

    conv2_1_map = map_compose(conv_map(3, 1, 1), pool1_map)
    conv2_2_map = map_compose(conv_map(3, 1, 1), conv2_1_map)
    pool2_map = map_compose(pool_map(2, 2, 0), conv2_2_map)

    conv3_1_map = map_compose(conv_map(3, 1, 1), pool2_map)
    conv3_2_map = map_compose(conv_map(3, 1, 1), conv3_1_map)
    conv3_3_map = map_compose(conv_map(3, 1, 1), conv3_2_map)
    pool3_map = map_compose(pool_map(2, 2, 0), conv3_3_map)

    conv4_1_map = map_compose(conv_map(3, 1, 1), pool3_map)
    conv4_2_map = map_compose(conv_map(3, 1, 1), conv4_1_map)
    conv4_3_map = map_compose(conv_map(3, 1, 1), conv4_2_map)
    pool4_map = map_compose(pool_map(2, 2, 0), conv4_3_map)

    conv5_1_map = map_compose(conv_map(3, 1, 1), pool4_map)
    conv5_2_map = map_compose(conv_map(3, 1, 1), conv5_1_map)
    conv5_3_map = map_compose(conv_map(3, 1, 1), conv5_2_map)

    score_dsn1_map = conv1_2_map
    score_dsn2_map = conv2_2_map
    score_dsn3_map = conv3_3_map
    score_dsn4_map = conv4_3_map
    score_dsn5_map = conv5_3_map

    upsample2_map = map_compose(deconv_map(4, 2, 0), score_dsn2_map)
    upsample3_map = map_compose(deconv_map(8, 4, 0), score_dsn3_map)
    upsample4_map = map_compose(deconv_map(16, 8, 0), score_dsn4_map)
    upsample5_map = map_compose(deconv_map(32, 16, 0), score_dsn5_map)

    crop1_margin = int(score_dsn1_map[1])
    crop2_margin = int(upsample2_map[1])
    crop3_margin = int(upsample3_map[1])
    crop4_margin = int(upsample4_map[1])
    crop5_margin = int(upsample5_map[1])

    return crop1_margin, crop2_margin, crop3_margin, crop4_margin, crop5_margin
