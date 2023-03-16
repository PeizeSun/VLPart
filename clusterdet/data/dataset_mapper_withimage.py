# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import random
import logging
import math
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F

import torchvision.transforms as transforms

from detectron2.config import configurable
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from .custom_build_augmentation import build_custom_augmentation
from .tar_dataset import DiskTarDataset
from .detection_utils import build_strong_augmentation
from .dataset_mapper_filterbybox import filter_empty_instances_by_box


logger = logging.getLogger('detectron2.clusterdet.data.dataset_mapper_withimage')

        
class DatasetMapperWithImage(DatasetMapper):
    @configurable
    def __init__(self, is_train: bool, 
        with_ann_type=False,
        dataset_ann=[],
        dataset_aug_type=[],
        strong_augmentation=[],
        use_diff_bs_size=False,
        dataset_augs=[],
        use_tar_dataset=False,
        tarfile_path='',
        tar_index_dir='',
        **kwargs):
        """
        add image labels
        """
        self.with_ann_type = with_ann_type
        self.dataset_ann = dataset_ann
        self.dataset_aug_type = dataset_aug_type
        self.use_diff_bs_size = use_diff_bs_size
        if self.use_diff_bs_size and is_train:
            self.dataset_augs = [T.AugmentationList(x) for x in dataset_augs]
        self.use_tar_dataset = use_tar_dataset
        if self.use_tar_dataset:
            logger.info('Using tar dataset')
            self.tar_dataset = DiskTarDataset(tarfile_path, tar_index_dir)
        super().__init__(is_train, **kwargs)
        self.strong_augmentation = strong_augmentation

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        ret.update({
            'with_ann_type': cfg.WITH_IMAGE_LABELS,
            'dataset_ann': cfg.DATALOADER.DATASET_ANN,
            'dataset_aug_type': cfg.DATALOADER.DATASET_AUG,
            'use_diff_bs_size': cfg.DATALOADER.USE_DIFF_BS_SIZE,
            'use_tar_dataset': cfg.DATALOADER.USE_TAR_DATASET,
            'tarfile_path': cfg.DATALOADER.TARFILE_PATH,
            'tar_index_dir': cfg.DATALOADER.TAR_INDEX_DIR,
        })
        if ret['use_diff_bs_size'] and is_train:
            if cfg.INPUT.CUSTOM_AUG == 'EfficientDetResizeCrop':
                dataset_scales = cfg.DATALOADER.DATASET_INPUT_SCALE
                dataset_sizes = cfg.DATALOADER.DATASET_INPUT_SIZE
                ret['dataset_augs'] = [
                    build_custom_augmentation(cfg, True, scale, size) \
                        for scale, size in zip(dataset_scales, dataset_sizes)]
            else:
                assert cfg.INPUT.CUSTOM_AUG == 'ResizeShortestEdge'
                min_sizes = cfg.DATALOADER.DATASET_MIN_SIZES
                max_sizes = cfg.DATALOADER.DATASET_MAX_SIZES
                ret['dataset_augs'] = [
                    build_custom_augmentation(
                        cfg, True, min_size=mi, max_size=ma) \
                        for mi, ma in zip(min_sizes, max_sizes)]
        else:
            ret['dataset_augs'] = []

        return ret

    def __call__(self, dataset_dict):
        """
        include image labels
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        if 'file_name' in dataset_dict:
            ori_image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format)
        else:
            ori_image, _, _ = self.tar_dataset[dataset_dict["tar_index"]]
            ori_image = utils._apply_exif_orientation(ori_image)
            ori_image = utils.convert_PIL_to_numpy(ori_image, self.image_format)
        # utils.check_image_size(dataset_dict, ori_image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(copy.deepcopy(ori_image), sem_seg=sem_seg_gt)
        if self.use_diff_bs_size and self.is_train:
            transforms = \
                self.dataset_augs[dataset_dict['dataset_source']](aug_input)
        else:
            transforms = self.augmentations(aug_input)
        image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image_weak_aug.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1)))

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, 
                proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            all_annos = [
                (utils.transform_instance_annotations(
                    obj, transforms, image_shape, 
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                ),  obj.get("iscrowd", 0))
                for obj in dataset_dict.pop("annotations")
            ]
            annos = [ann[0] for ann in all_annos if ann[1] == 0]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            
            del all_annos
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = filter_empty_instances_by_box(instances)

        if self.with_ann_type:
            dataset_dict["pos_category_ids"] = dataset_dict.get(
                'pos_category_ids', [])
            dataset_dict["ann_type"] = \
                self.dataset_ann[dataset_dict['dataset_source']]
        return dataset_dict
