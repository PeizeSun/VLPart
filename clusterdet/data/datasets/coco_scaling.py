# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import datetime
import io
import json
import pickle
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json


logger = logging.getLogger("detectron2.clusterdet.data.datasets.coco_scaling")


def load_coco_percent_json(json_file, percent, image_root, dataset_name=None, extra_annotation_keys=None):
    dataset_dicts = load_coco_json(json_file, image_root, dataset_name, extra_annotation_keys)
    dataset_percent_dicts = []
    for idx, dataset_dict in enumerate(dataset_dicts):
        if idx % (100//percent) == 0:
            dataset_percent_dicts.append(dataset_dict)
    logger.info("Loaded {} images in dataset split {}".format(len(dataset_percent_dicts), dataset_name))

    return dataset_percent_dicts


def register_coco_percent_instances(name, metadata, json_file, percent, image_root):
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_percent_json(json_file, percent, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )

# ==== Predefined datasets and splits for COCO WEAKLY==========
_PREDEFINED_SPLITS_COCO_PERCENT = {
    "coco_2017_train_edge_10percent": ("coco/train2017", "coco/unsupervised/instances_train2017_edge.json", 10),
    "coco_2017_train_edge_25percent": ("coco/train2017", "coco/unsupervised/instances_train2017_edge.json", 25),
    "coco_2017_train_edge_50percent": ("coco/train2017", "coco/unsupervised/instances_train2017_edge.json", 50),
}

def register_all_coco_percent(root):
    for key, (image_root, json_file, percent) in _PREDEFINED_SPLITS_COCO_PERCENT.items():
        register_coco_percent_instances(
            key,
            _get_builtin_metadata('coco'),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            percent,
            os.path.join(root, image_root),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_percent(_root)
