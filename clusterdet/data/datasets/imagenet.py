# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis import get_lvis_instances_meta
from .lvis_v1 import custom_load_lvis_json, get_lvis_22k_meta
from .voc import _get_builtin_metadata as get_voc_instances_meta
from .partimagenet import _get_partimagenet_metadata as get_partimagenet_instances_meta


def _get_imagenet_metadata(key):
    if 'lvis_v1' in key:
        return get_lvis_instances_meta('lvis_v1')
    elif 'voc' in key:
        return get_voc_instances_meta()
    elif 'partimagenet' in key:
        return get_partimagenet_instances_meta(key)


def custom_register_imagenet_instances(name, metadata, json_file, image_root):
    DatasetCatalog.register(name, lambda: custom_load_lvis_json(
        json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, 
        evaluator_type="imagenet", **metadata
    )


_CUSTOM_SPLITS_IMAGENET = {
    "imagenet_lvis_v1": ("imagenet/ImageNet-LVIS/", "imagenet/annotations/imagenet_lvis_image_info.json"),
    "imagenet_voc": ("imagenet/train/", "imagenet/imagenet_voc_image_info.json"),
    "imagenet_voc_mini": ("imagenet/train/", "imagenet/imagenet_voc_mini_image_info.json"),
    "partimagenet_supercat": ("partimagenet/train/", "partimagenet/partimagenet_supercat_image_info.json"),
}


def register_all_lvis_imagenet(root):
    for key, (image_root, json_file) in _CUSTOM_SPLITS_IMAGENET.items():
        custom_register_imagenet_instances(
            key,
            _get_imagenet_metadata(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# _CUSTOM_SPLITS_IMAGENET_22K = {
#     "imagenet_lvis-22k": ("imagenet/ImageNet-LVIS/", "imagenet/annotations/imagenet-22k_image_info_lvis-22k.json"),
# }

# for key, (image_root, json_file) in _CUSTOM_SPLITS_IMAGENET_22K.items():
#     custom_register_imagenet_instances(
#         key,
#         get_lvis_22k_meta(),
#         os.path.join("datasets", json_file) if "://" not in json_file else json_file,
#         os.path.join("datasets", image_root),
#     )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_lvis_imagenet(_root)


