# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data.datasets.register_coco import register_coco_instances
# from .coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

def _get_metadata():
    return _get_builtin_metadata('coco')

_PREDEFINED_SPLITS_COCO_UNSUPERVISED = {
    "coco_2017_train_densecl_r101": ("coco/train2017", "coco_unsupervised/annotations/instances_train2017_densecl_r101.json"),
    "coco_2017_train_unlabeled_densecl_r101": ("coco", "coco_unsupervised/annotations/instances_train2017_unlabeled2017_densecl_r101.json"),
    "coco_2017_train_hed": ("coco/train2017", "coco/unsupervised/instances_train2017_grouping_hed.json"),
    "coco_2017_train_hed5": ("coco/train2017", "coco/unsupervised/instances_train2017_grouping_hed5.json"),
    "coco_2017_train_densecl": ("coco/train2017", "coco/unsupervised/instances_train2017_grouping_densecl.json"),
    "coco_2017_train_edge": ("coco/train2017", "coco/unsupervised/instances_train2017_edge.json"),
    "coco_2017_train_edge_cc5": ("coco/train2017", "coco/unsupervised/instances_train2017_edge_cc5.json"),
    "coco_2017_unlabeled_edge": ("coco/unlabeled2017", "coco/unsupervised/instances_unlabeled2017_edge.json"),
}

def register_all_coco_unsupervised(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_UNSUPERVISED.items():
        register_coco_instances(
            key,
            _get_metadata(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )    


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_unsupervised(_root)