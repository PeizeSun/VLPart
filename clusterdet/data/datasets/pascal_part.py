# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json

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

PASCAL_PART_NOVEL_CATEGORIES = [
    {"id": 1, "name": "aeroplane:body"},
    {"id": 2, "name": "aeroplane:wing"},
    {"id": 3, "name": "aeroplane:tail"},
    {"id": 4, "name": "aeroplane:wheel"},
    {"id": 5, "name": "bicycle:wheel"},
    {"id": 6, "name": "bicycle:handlebar"},
    {"id": 7, "name": "bicycle:saddle"},
    {"id": 17, "name": "bottle:body"},
    {"id": 18, "name": "bottle:cap"},
    {"id": 19, "name": "bus:license plate", "abbr": "bus:liplate"},
    {"id": 20, "name": "bus:headlight"},
    {"id": 21, "name": "bus:door"},
    {"id": 22, "name": "bus:mirror"},
    {"id": 23, "name": "bus:window"},
    {"id": 24, "name": "bus:wheel"},
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
    {"id": 83, "name": "pottedplant:plant"},
    {"id": 84, "name": "pottedplant:pot"},
]

PASCAL_PART_BASE_CATEGORIES = [
    {"id": 1, "name": "bird:beak"},
    {"id": 2, "name": "bird:head"},
    {"id": 3, "name": "bird:eye"},
    {"id": 4, "name": "bird:leg"},
    {"id": 5, "name": "bird:foot"},
    {"id": 6, "name": "bird:wing"},
    {"id": 7, "name": "bird:neck"},
    {"id": 8, "name": "bird:tail"},
    {"id": 9, "name": "bird:torso"},
    {"id": 10, "name": "car:license plate", "abbr": "car:liplate"},
    {"id": 11, "name": "car:headlight"},
    {"id": 12, "name": "car:door"},
    {"id": 13, "name": "car:mirror"},
    {"id": 14, "name": "car:window"},
    {"id": 15, "name": "car:wheel"},
    {"id": 16, "name": "cat:head"},
    {"id": 17, "name": "cat:leg"},
    {"id": 18, "name": "cat:ear"},
    {"id": 19, "name": "cat:eye"},
    {"id": 20, "name": "cat:paw", "abbr": "cat:pa"},
    {"id": 21, "name": "cat:neck"},
    {"id": 22, "name": "cat:nose"},
    {"id": 23, "name": "cat:tail"},
    {"id": 24, "name": "cat:torso"},
    {"id": 25, "name": "horse:head"},
    {"id": 26, "name": "horse:leg"},
    {"id": 27, "name": "horse:ear"},
    {"id": 28, "name": "horse:eye"},
    {"id": 29, "name": "horse:neck"},
    {"id": 30, "name": "horse:muzzle"},
    {"id": 31, "name": "horse:tail"},
    {"id": 32, "name": "horse:torso"},
    {"id": 33, "name": "motorbike:wheel"},
    {"id": 34, "name": "motorbike:handlebar"},
    {"id": 35, "name": "motorbike:headlight"},
    {"id": 36, "name": "motorbike:saddle"},
    {"id": 37, "name": "person:hair"},
    {"id": 38, "name": "person:head"},
    {"id": 39, "name": "person:ear"},
    {"id": 40, "name": "person:eye"},
    {"id": 41, "name": "person:nose"},
    {"id": 42, "name": "person:neck"},
    {"id": 43, "name": "person:mouth"},
    {"id": 44, "name": "person:arm"},
    {"id": 45, "name": "person:hand"},
    {"id": 46, "name": "person:leg"},
    {"id": 47, "name": "person:foot"},
    {"id": 48, "name": "person:torso"},
    {"id": 49, "name": "sheep:head"},
    {"id": 50, "name": "sheep:leg"},
    {"id": 51, "name": "sheep:ear"},
    {"id": 52, "name": "sheep:eye"},
    {"id": 53, "name": "sheep:neck"},
    {"id": 54, "name": "sheep:horn"},
    {"id": 55, "name": "sheep:muzzle"},
    {"id": 56, "name": "sheep:tail"},
    {"id": 57, "name": "sheep:torso"},
]

PASCAL_PART_DOG_CATEGORIES = [
    {"id": 1, "name": "dog:head"},
    {"id": 2, "name": "dog:leg"},
    {"id": 3, "name": "dog:ear"},
    {"id": 4, "name": "dog:eye"},
    {"id": 5, "name": "dog:paw", "abbr": "dog:pa"},
    {"id": 6, "name": "dog:neck"},
    {"id": 7, "name": "dog:nose"},
    {"id": 8, "name": "dog:muzzle"},
    {"id": 9, "name": "dog:tail"},
    {"id": 10, "name": "dog:torso"},
]

PASCAL_PART_DOG2_CATEGORIES = [
    {"id": 1, "name": "dog:head"},
    {"id": 2, "name": "dog:leg"},
    # {"id": 3, "name": "dog:ear"},
    # {"id": 4, "name": "dog:eye"},
    {"id": 5, "name": "dog:paw", "abbr": "dog:pa"},
    # {"id": 6, "name": "dog:neck"},
    # {"id": 7, "name": "dog:nose"},
    # {"id": 8, "name": "dog:muzzle"},
    {"id": 9, "name": "dog:tail"},
    {"id": 10, "name": "dog:torso"},
]

PASCAL_META_PART_CATEGORIES = [
  {'id': 1, 'name': 'body'},
  {'id': 2, 'name': 'wing'},
  {'id': 3, 'name': 'tail'},
  {'id': 4, 'name': 'wheel'},
  {'id': 5, 'name': 'handlebar'},
  {'id': 6, 'name': 'saddle'},
  {'id': 7, 'name': 'beak'},
  {'id': 8, 'name': 'head'},
  {'id': 9, 'name': 'eye'},
  {'id': 10, 'name': 'leg'},
  {'id': 11, 'name': 'foot'},
  {'id': 12, 'name': 'neck'},
  {'id': 13, 'name': 'torso'},
  {'id': 14, 'name': 'cap'},
  {'id': 15, 'name': 'license plate'},
  {'id': 16, 'name': 'headlight'},
  {'id': 17, 'name': 'door'},
  {'id': 18, 'name': 'mirror'},
  {'id': 19, 'name': 'window'},
  {'id': 20, 'name': 'ear'},
  {'id': 21, 'name': 'paw'},
  {'id': 22, 'name': 'nose'},
  {'id': 23, 'name': 'horn'},
  {'id': 24, 'name': 'muzzle'},
  {'id': 25, 'name': 'hair'},
  {'id': 26, 'name': 'mouth'},
  {'id': 27, 'name': 'arm'},
  {'id': 28, 'name': 'hand'},
  {'id': 29, 'name': 'plant'},
  {'id': 30, 'name': 'pot'},
]


def _get_partimagenet_metadata(key):
    if 'base' in key:
        id_to_name = {x['id']: x['name'] for x in PASCAL_PART_BASE_CATEGORIES}
    # elif 'dog' in key:
    #     id_to_name = {x['id']: x['name'] for x in PASCAL_PART_DOG_CATEGORIES}
    # elif 'metapart' in key:
    #     id_to_name = {x['id']: x['name'] for x in PASCAL_META_PART_CATEGORIES}
    else:
        id_to_name = {x['id']: x['name'] for x in PASCAL_PART_CATEGORIES}
    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}


def register_pascal_part_instances(name, metadata, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_coco_json(
        json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root,
        evaluator_type="pascal_part", **metadata
    )

_PASCAL_PART = {
    "pascal_part_train": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train.json"),
    "pascal_part_train_one": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train_one.json"),
    "pascal_part_val": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/val.json"),
    "pascal_part_base_train": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train_base.json"),
    "pascal_part_base_train_one": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train_base_one.json"),
    "pascal_part_dog_train": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train_dog.json"),
    "pascal_part_val_metapart": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/val.json"), # visualization only
    "imagenet_dog_train_ppart": ("imagenet_dog/Images", "imagenet_dog/train_ppart.json"),
    "imagenet_voc_parsed": ("imagenet/train", "imagenet/imagenet_voc_parsed.json"),
    "imagenet_voc_mini_parsed": ("imagenet/train", "imagenet/imagenet_voc_mini_parsed.json"),
}


def register_all_pascal_part(root):
    for key, (image_root, json_file) in _PASCAL_PART.items():
        register_pascal_part_instances(
            key,
            _get_partimagenet_metadata(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_pascal_part(_root)
