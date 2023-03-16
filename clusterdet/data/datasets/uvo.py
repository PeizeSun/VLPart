from detectron2.data.datasets.register_coco import register_coco_instances
import os

def _get_builtin_metadata():
    return {"thing_classes": ['object']}


_PREDEFINED_SPLITS_UVO = {
    "uvo_frame_val": ("uvo/collect_val_images", "uvo/UVO_frame_val.json")
}

def register_uvo(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_UVO.items():
        register_coco_instances(
            key,
            _get_builtin_metadata(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_uvo(_root)
