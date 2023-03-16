from detectron2.data.datasets.register_coco import register_coco_instances
import os

def _get_builtin_metadata():
    return {"thing_classes": ['entity']}


_PREDEFINED_SPLITS_ADE20K = {
    "ade20k_full_val": ("../", "ADE20K_2021_17_01/ade20k_full_entity_val.json")
}

def register_ade20k(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_ADE20K.items():
        register_coco_instances(
            key,
            _get_builtin_metadata(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_ade20k(_root)
