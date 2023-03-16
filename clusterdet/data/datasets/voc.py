from detectron2.data.datasets.register_coco import register_coco_instances
import os

VOC_CATEGORIES = [
  {'id': 1, 'name': 'aeroplane'},
  {'id': 2, 'name': 'bicycle'},
  {'id': 3, 'name': 'bird'},
  {'id': 4, 'name': 'boat'},
  {'id': 5, 'name': 'bottle'},
  {'id': 6, 'name': 'bus'},
  {'id': 7, 'name': 'car'},
  {'id': 8, 'name': 'cat'},
  {'id': 9, 'name': 'chair'},
  {'id': 10, 'name': 'cow'},
  {'id': 11, 'name': 'dining table'},
  {'id': 12, 'name': 'dog'},
  {'id': 13, 'name': 'horse'},
  {'id': 14, 'name': 'motorbike'},
  {'id': 15, 'name': 'person'},
  {'id': 16, 'name': 'pottedplant'},
  {'id': 17, 'name': 'sheep'},
  {'id': 18, 'name': 'sofa'},
  {'id': 19, 'name': 'train'},
  {'id': 20, 'name': 'tv monitor'},
]


def _get_builtin_metadata():
    id_to_name = {x['id']: x['name'] for x in VOC_CATEGORIES}
    thing_dataset_id_to_contiguous_id = {
        x['id']: i for i, x in enumerate(
            sorted(VOC_CATEGORIES, key=lambda x: x['id']))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

# 80,000 images, 1,240,587 annotations
_PREDEFINED_SPLITS_OBJECTS365 = {
    "my_voc_test": ("voc/VOCdevkit/VOC2007/JPEGImages", "voc/pascal_test2007.json"),
    "my_voc_val": ("voc/VOCdevkit/VOC2007/JPEGImages", "voc/pascal_val2007.json"),
    "imagenet_dog_train": ("imagenet_dog/Images", "imagenet_dog/train_image_info.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_OBJECTS365.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )