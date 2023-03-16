# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os

from detectron2.data.datasets.register_coco import register_coco_instances

PARTIMAGENET_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 0, "name": "Quadruped Head"},
    {"color": [119, 11, 32], "isthing": 1, "id": 1, "name": "Quadruped Body"},
    {"color": [0, 0, 142], "isthing": 1, "id": 2, "name": "Quadruped Foot"},
    {"color": [0, 0, 230], "isthing": 1, "id": 3, "name": "Quadruped Tail"},
    {"color": [106, 0, 228], "isthing": 1, "id": 4, "name": "Biped Head"},
    {"color": [0, 60, 100], "isthing": 1, "id": 5, "name": "Biped Body"},
    {"color": [0, 80, 100], "isthing": 1, "id": 6, "name": "Biped Hand"},
    {"color": [0, 0, 70], "isthing": 1, "id": 7, "name": "Biped Foot"},
    {"color": [0, 0, 192], "isthing": 1, "id": 8, "name": "Biped Tail"},
    {"color": [250, 170, 30], "isthing": 1, "id": 9, "name": "Fish Head"},
    {"color": [100, 170, 30], "isthing": 1, "id": 10, "name": "Fish Body"},
    {"color": [220, 220, 0], "isthing": 1, "id": 11, "name": "Fish Fin"},
    {"color": [175, 116, 175], "isthing": 1, "id": 12, "name": "Fish Tail"},
    {"color": [250, 0, 30], "isthing": 1, "id": 13, "name": "Bird Head"},
    {"color": [165, 42, 42], "isthing": 1, "id": 14, "name": "Bird Body"},
    {"color": [255, 77, 255], "isthing": 1, "id": 15, "name": "Bird Wing"},
    {"color": [0, 226, 252], "isthing": 1, "id": 16, "name": "Bird Foot"},
    {"color": [182, 182, 255], "isthing": 1, "id": 17, "name": "Bird Tail"},
    {"color": [0, 82, 0], "isthing": 1, "id": 18, "name": "Snake Head"},
    {"color": [120, 166, 157], "isthing": 1, "id": 19, "name": "Snake Body"},
    {"color": [110, 76, 0], "isthing": 1, "id": 20, "name": "Reptile Head"},
    {"color": [174, 57, 255], "isthing": 1, "id": 21, "name": "Reptile Body"},
    {"color": [199, 100, 0], "isthing": 1, "id": 22, "name": "Reptile Foot"},
    {"color": [72, 0, 118], "isthing": 1, "id": 23, "name": "Reptile Tail"},
    {"color": [255, 179, 240], "isthing": 1, "id": 24, "name": "Car Body"},
    {"color": [0, 125, 92], "isthing": 1, "id": 25, "name": "Car Tier"},
    {"color": [209, 0, 151], "isthing": 1, "id": 26, "name": "Car Side Mirror"},
    {"color": [188, 208, 182], "isthing": 1, "id": 27, "name": "Bicycle Body"},
    {"color": [0, 220, 176], "isthing": 1, "id": 28, "name": "Bicycle Head"},
    {"color": [255, 99, 164], "isthing": 1, "id": 29, "name": "Bicycle Seat"},
    {"color": [92, 0, 73], "isthing": 1, "id": 30, "name": "Bicycle Tier"},
    {"color": [133, 129, 255], "isthing": 1, "id": 31, "name": "Boat Body"},
    {"color": [78, 180, 255], "isthing": 1, "id": 32, "name": "Boat Sail"},
    {"color": [0, 228, 0], "isthing": 1, "id": 33, "name": "Aeroplane Head"},
    {"color": [174, 255, 243], "isthing": 1, "id": 34, "name": "Aeroplane Body"},
    {"color": [45, 89, 255], "isthing": 1, "id": 35, "name": "Aeroplane Engine"},
    {"color": [134, 134, 103], "isthing": 1, "id": 36, "name": "Aeroplane Wing"},
    {"color": [145, 148, 174], "isthing": 1, "id": 37, "name": "Aeroplane Tail"},
    {"color": [255, 208, 186], "isthing": 1, "id": 38, "name": "Bottle Mouth"},
    {"color": [197, 226, 255], "isthing": 1, "id": 39, "name": "Bottle Body"},
]

PARTIMAGENET_SUPER_CATEGORIES = [
    {"id": 0, "name": "Quadruped"},
    {"id": 1, "name": "Biped"},
    {"id": 2, "name": "Fish"},
    {"id": 3, "name": "Bird"},
    {"id": 4, "name": "Snake"},
    {"id": 5, "name": "Reptile"},
    {"id": 6, "name": "Car"},
    {"id": 7, "name": "Bicycle"},
    {"id": 8, "name": "Boat"},
    {"id": 9, "name": "Aeroplane"},
    {"id": 10, "name": "Bottle"},
]

def _get_partimagenet_metadata(key):
    if 'supercat' in key:
        id_to_name = {x['id']: x['name'] for x in PARTIMAGENET_SUPER_CATEGORIES}
    else:
        id_to_name = {x['id']: x['name'] for x in PARTIMAGENET_CATEGORIES}
    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PARTIMAGENET = {
    "partimagenet_train": ("partimagenet/train/", "partimagenet/train_format.json"),
    "partimagenet_val": ("partimagenet/val/", "partimagenet/val_format.json"),
    "partimagenet_parsed": ("partimagenet/train", "partimagenet/partimagenet_parsed.json"),
}

def register_all_partimagenet(root):
    for key, (image_root, json_file) in _PARTIMAGENET.items():
        register_coco_instances(
            key,
            _get_partimagenet_metadata(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_partimagenet(_root)
