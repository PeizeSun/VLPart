# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image

from detectron2.utils.file_io import PathManager
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger("detectron2.clusterdet.data.datasets.coco_unlabeled")


def load_coco_unlabeled_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None, all_images=False):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))
    dataset_dicts = []

    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["image_id"] = img_dict["id"]

        objs = []
        record["annotations"] = objs
        dataset_dicts.append(record)

    logger.info("Loaded {} images in dataset split {}".format(len(dataset_dicts), dataset_name))
    return dataset_dicts


def register_coco_unlabeled_instances(name, metadata, json_file, image_root, all_images=False):
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_unlabeled_json(json_file, image_root, name, all_images=all_images))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def _get_metadata():
    return _get_builtin_metadata('coco')


_PREDEFINED_SPLITS_COCO_UNLABELED = {
    "coco_2017_unlabeled": ("coco/unlabeled2017", "coco/annotations/image_info_unlabeled2017.json"),
}


def register_all_coco_unlabeled(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_UNLABELED.items():
        register_coco_unlabeled_instances(
            key,
            _get_metadata(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )    


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_unlabeled(_root)
