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

logger = logging.getLogger("detectron2.clusterdet.data.datasets.pascal_part_percent")

PASCAL_PART_CATEGORIES = [
    {"id": 1, "name": "aeroplane:body"},
    {"id": 2, "name": "aeroplane:wing"},
    {"id": 3, "name": "aeroplane:tail"},
    {"id": 4, "name": "aeroplane:wheel"},
    {"id": 5, "name": "bicycle:wheel"},
    {"id": 6, "name": "bicycle:handlebar"},
    {"id": 7, "name": "bicycle:saddle"},
    {"id": 8, "name": "bird:beak"},
    {"id": 9, "name": "bird:head"},
    {"id": 10, "name": "bird:eye"},
    {"id": 11, "name": "bird:leg"},
    {"id": 12, "name": "bird:foot"},
    {"id": 13, "name": "bird:wing"},
    {"id": 14, "name": "bird:neck"},
    {"id": 15, "name": "bird:tail"},
    {"id": 16, "name": "bird:torso"},
    {"id": 17, "name": "bottle:body"},
    {"id": 18, "name": "bottle:cap"},
    {"id": 19, "name": "bus:license plate", "abbr": "bus:liplate"},
    {"id": 20, "name": "bus:headlight"},
    {"id": 21, "name": "bus:door"},
    {"id": 22, "name": "bus:mirror"},
    {"id": 23, "name": "bus:window"},
    {"id": 24, "name": "bus:wheel"},
    {"id": 25, "name": "car:license plate", "abbr": "car:liplate"},
    {"id": 26, "name": "car:headlight"},
    {"id": 27, "name": "car:door"},
    {"id": 28, "name": "car:mirror"},
    {"id": 29, "name": "car:window"},
    {"id": 30, "name": "car:wheel"},
    {"id": 31, "name": "cat:head"},
    {"id": 32, "name": "cat:leg"},
    {"id": 33, "name": "cat:ear"},
    {"id": 34, "name": "cat:eye"},
    {"id": 35, "name": "cat:paw", "abbr": "cat:pa"},
    {"id": 36, "name": "cat:neck"},
    {"id": 37, "name": "cat:nose"},
    {"id": 38, "name": "cat:tail"},
    {"id": 39, "name": "cat:torso"},
    {"id": 40, "name": "cow:head"},
    {"id": 41, "name": "cow:leg"},
    {"id": 42, "name": "cow:ear"},
    {"id": 43, "name": "cow:eye"},
    {"id": 44, "name": "cow:neck"},
    {"id": 45, "name": "cow:horn"},
    {"id": 46, "name": "cow:muzzle"},
    {"id": 47, "name": "cow:tail"},
    {"id": 48, "name": "cow:torso"},
    {"id": 49, "name": "dog:head"},
    {"id": 50, "name": "dog:leg"},
    {"id": 51, "name": "dog:ear"},
    {"id": 52, "name": "dog:eye"},
    {"id": 53, "name": "dog:paw", "abbr": "dog:pa"},
    {"id": 54, "name": "dog:neck"},
    {"id": 55, "name": "dog:nose"},
    {"id": 56, "name": "dog:muzzle"},
    {"id": 57, "name": "dog:tail"},
    {"id": 58, "name": "dog:torso"},
    {"id": 59, "name": "horse:head"},
    {"id": 60, "name": "horse:leg"},
    {"id": 61, "name": "horse:ear"},
    {"id": 62, "name": "horse:eye"},
    {"id": 63, "name": "horse:neck"},
    {"id": 64, "name": "horse:muzzle"},
    {"id": 65, "name": "horse:tail"},
    {"id": 66, "name": "horse:torso"},
    {"id": 67, "name": "motorbike:wheel"},
    {"id": 68, "name": "motorbike:handlebar"},
    {"id": 69, "name": "motorbike:headlight"},
    {"id": 70, "name": "motorbike:saddle"},
    {"id": 71, "name": "person:hair"},
    {"id": 72, "name": "person:head"},
    {"id": 73, "name": "person:ear"},
    {"id": 74, "name": "person:eye"},
    {"id": 75, "name": "person:nose"},
    {"id": 76, "name": "person:neck"},
    {"id": 77, "name": "person:mouth"},
    {"id": 78, "name": "person:arm"},
    {"id": 79, "name": "person:hand"},
    {"id": 80, "name": "person:leg"},
    {"id": 81, "name": "person:foot"},
    {"id": 82, "name": "person:torso"},
    {"id": 83, "name": "pottedplant:plant"},
    {"id": 84, "name": "pottedplant:pot"},
    {"id": 85, "name": "sheep:head"},
    {"id": 86, "name": "sheep:leg"},
    {"id": 87, "name": "sheep:ear"},
    {"id": 88, "name": "sheep:eye"},
    {"id": 89, "name": "sheep:neck"},
    {"id": 90, "name": "sheep:horn"},
    {"id": 91, "name": "sheep:muzzle"},
    {"id": 92, "name": "sheep:tail"},
    {"id": 93, "name": "sheep:torso"},
]


def load_percent_json(json_file, percent, image_root, dataset_name=None, extra_annotation_keys=None):
    dataset_dicts = load_coco_json(json_file, image_root, dataset_name, extra_annotation_keys)
    dataset_percent_dicts = []
    for idx, dataset_dict in enumerate(dataset_dicts):
        if idx % (100//percent) == 0:
            dataset_percent_dicts.append(dataset_dict)
    logger.info("Loaded {} images in dataset split {}".format(len(dataset_percent_dicts), dataset_name))

    return dataset_percent_dicts


def register_percent_instances(name, metadata, json_file, percent, image_root):
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_percent_json(json_file, percent, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )

def _get_partimagenet_metadata():
    id_to_name = {x['id']: x['name'] for x in PASCAL_PART_CATEGORIES}
    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}


# ==== Predefined datasets and splits ==========
_PREDEFINED_PERCENT = {
    "pascal_part_train10": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train.json", 10),
    "pascal_part_train50": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train.json", 50),
}

def register_all_pascal_part_percent(root):
    for key, (image_root, json_file, percent) in _PREDEFINED_PERCENT.items():
        register_percent_instances(
            key,
            _get_partimagenet_metadata(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            percent,
            os.path.join(root, image_root),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_pascal_part_percent(_root)
