# Copyright (c) Facebook, Inc. and its affiliates.
import os
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from .lvis_v1 import custom_load_lvis_json


logger = logging.getLogger("detectron2.clusterdet.data.datasets.coco_edgetext")


def load_percent_json(json_file, percent, image_root, dataset_name=None):
    dataset_dicts = custom_load_lvis_json(json_file, image_root, dataset_name)
    dataset_percent_dicts = []
    for idx, dataset_dict in enumerate(dataset_dicts):
        if idx % (100//percent) == 0:
            dataset_percent_dicts.append(dataset_dict)
    logger.info("Loaded {} images in dataset split {}".format(len(dataset_percent_dicts), dataset_name))

    return dataset_percent_dicts


def register_edgetext_instances(name, metadata, json_file, percent, image_root):
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

_COCO_PERCENT_SPLITS = {
    "coco_2017_train_edgetag": ("coco/train2017/", "coco/edgetag/instances_train2017_coco_edgetag.json", 100),
    "coco_2017_train_edgetext": ("coco/train2017/", "coco/edgetext/instances_train2017_coco_edgetext.json", 100),
    "coco_2017_train_edgetext_5percent": ("coco/train2017", "coco/edgetext/instances_train2017_coco_edgetext.json", 5),
    "coco_2017_train_edgetext_2percent": ("coco/train2017", "coco/edgetext/instances_train2017_coco_edgetext.json", 2),
    "coco_2017_train_5percent": ("coco/train2017", "coco/annotations/instances_train2017.json", 5),
    "coco_2017_train_2percent": ("coco/train2017", "coco/annotations/instances_train2017.json", 2),
}

def register_all_coco_edgetext(root):
    for key, (image_root, json_file, percent) in _COCO_PERCENT_SPLITS.items():
        register_edgetext_instances(
            key,
            _get_builtin_metadata('coco'),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            percent,
            os.path.join(root, image_root),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_edgetext(_root)
