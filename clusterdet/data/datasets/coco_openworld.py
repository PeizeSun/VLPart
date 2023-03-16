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

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

# from detectron2.data.datasets.register_coco import register_coco_instances
# from .coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger("detectron2.clusterdet.data.datasets.coco_openworld")

voc_categories = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
    "airplane", "tv", "dining table", "motorcycle",
     "potted plant", "couch",)
# person(1), motorcycle(2), train(1), cat(1), sheep(1), bottle(1), couch(2), 
# dinning table(2), bicycle(1), airplane(2), dog(1),    cow(1),    potted plant(2),
# car(1), bus(1), boat(1),  bird(1),  horse(1),        chair(1),   tv(2)


def load_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None, all_images=False):
    """
    coco_openworld setting 
    this load func selects invoc and outvoc categories
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

#     id_map = None
#     if dataset_name is not None:
#         meta = MetadataCatalog.get(dataset_name)
#         cat_ids = sorted(coco_api.getCatIds())
#         cats = coco_api.loadCats(cat_ids)
#         # The categories in a custom json file may not be sorted.
#         thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
#         meta.thing_classes = thing_classes

#         # In COCO, certain category ids are artificially removed,
#         # and by convention they are always ignored.
#         # We deal with COCO's id issue and translate
#         # the category ids to contiguous ids in [0, 80).

#         # It works by looking at the "categories" field in the json, therefore
#         # if users' own json also have incontiguous ids, we'll
#         # apply this mapping as well but print a warning.
#         if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
#             if "coco" not in dataset_name:
#                 logger.warning(
#                     """
# Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
# """
#                 )
#         id_map = {v: i for i, v in enumerate(cat_ids)}
#         meta.thing_dataset_id_to_contiguous_id = id_map

    # here we use the pre-defined meta
    meta = MetadataCatalog.get(dataset_name)
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    id_map = meta.thing_dataset_id_to_contiguous_id
    
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )
    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]

        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))
    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])
    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )
            
            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]

                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            cat_name = cats[obj["category_id"]]

            ## For VOC-COCO experiments, we exclude annotations of non-voc categories.
            if "invoc" in dataset_name:
                if cat_name['name'] in voc_categories:
                    objs.append(obj)
                elif all_images:
                    obj["category_id"] = 80
                    objs.append(obj)                    
            elif "outvoc" in dataset_name:
                if not cat_name['name'] in voc_categories:
                    obj["ignored_split"] = 1
                objs.append(obj)
            else:
                objs.append(obj)
        
        if len(objs) > 0:
            record["annotations"] = objs
            dataset_dicts.append(record)
            

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )

    logger.info("Loaded {} images in dataset split {}".format(len(dataset_dicts), dataset_name))
    return dataset_dicts


def register_coco_instances_all_images(name, metadata, json_file, image_root, all_images=False):
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name, all_images=all_images))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )

    
def _get_metadata(dataset_name):
    if dataset_name == "coco_openworld":
        coco_meta = _get_builtin_metadata('coco')
        coco_meta["thing_dataset_id_to_contiguous_id"].update({-1:80})
        coco_meta["thing_classes"].append('fake_object')
        coco_meta["thing_colors"].append([255, 255, 255])
        return coco_meta
    elif dataset_name == "coco":
        return _get_builtin_metadata('coco')

    
# ==== Predefined datasets and splits for COCO OPEN-WORLR==========

# _PREDEFINED_SPLITS_COCO_OPENWORLD = {
#     "coco_2017_train_invoc": ("coco/train2017", "coco/annotations/instances_train2017.json"),
#     "coco_2017_val_outvoc": ("coco/val2017", "coco/annotations/instances_val2017.json"),
#     "coco_2017_train_invoc_all_images": ("coco/train2017", "coco/annotations/instances_train2017.json"),
#     "coco_2017_train_grouping_reg": ("coco/train2017", "coco/openworld/instances_train2017_grouping_reg.json"),
#     "coco_2017_train_grouping_binary": ("coco/train2017", "coco/openworld/instances_train2017_grouping_binary.json"),
#     "coco_2017_train_grouping_hed": ("coco/train2017", "coco/openworld/instances_train2017_grouping_hed.json"),
#     "coco_2017_train_invoc_hed": ("coco/train2017", "coco/openworld/instances_train2017_invoc_hed.json"),
#     "coco_2017_train_coco_hed": ("coco/train2017", "coco/openworld/instances_train2017_coco_hed.json"),
#     "coco_2017_train_invoc_edge": ("coco/train2017", "coco/openworld/instances_train2017_invoc_edge.json"),
#     "coco_2017_train_coco_edge": ("coco/train2017", "coco/openworld/instances_train2017_coco_edge.json"),
#     "coco_2017_unlabeled_invoc": ("coco/unlabeled2017", "coco/openworld/instances_unlabeled2017_invoc.json"),
#     "coco_2017_unlabeled_invoc_edge": ("coco/unlabeled2017", "coco/openworld/instances_unlabeled2017_invoc_edge.json"),
# }

_PREDEFINED_SPLITS_COCO_OPENWORLD = {
    "coco_2017_train_invoc": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val_outvoc": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_train_invoc_all_images": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_train_invoc_edge": ("coco/train2017", "coco/openworld/instances_train2017_invoc_edge.json"),
    "coco_2017_train_coco_edge": ("coco/train2017", "coco/openworld/instances_train2017_coco_edge.json"),
    "coco_2017_unlabeled_invoc": ("coco/unlabeled2017", "coco/openworld/instances_unlabeled2017_invoc.json"),
    "coco_2017_unlabeled_invoc_edge": ("coco/unlabeled2017", "coco/openworld/instances_unlabeled2017_invoc_edge.json"),
    "coco_2017_unlabeled_coco_edge": ("coco/unlabeled2017", "coco/openworld/instances_unlabeled2017_coco_edge.json"),
}

        
def register_all_coco_openworld(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_OPENWORLD.items():
        metadata_name = 'coco_openworld' if "all_images" in key else 'coco'
        register_coco_instances_all_images(
            key,
            _get_metadata(metadata_name),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            all_images="all_images" in key,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_openworld(_root)
