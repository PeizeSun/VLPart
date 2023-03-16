# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import io
import cv2
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

from ..data.datasets.voc import VOC_CATEGORIES
from ..data.datasets.pascal_part import (
    PASCAL_PART_BASE_CATEGORIES,
    PASCAL_PART_CATEGORIES,
)

from ..data.datasets.partimagenet import (
    PARTIMAGENET_CATEGORIES,
    PARTIMAGENET_SUPER_CATEGORIES
)

PART_SYNONYMS = {'body': ['body', 'torso'],
                 'torso': ['body', 'torso'],
                 'paw': ['hand', 'foot'],
                 'cap': ['mouth', ],
                 }
class AnnJsonGenerator(COCOEvaluator):
    def __init__(self, cfg, **kwargs, ):
        super().__init__(**kwargs)
        test_set = cfg.DATASETS.TEST[0]
        if 'voc' in test_set:
            OBJECT = VOC_CATEGORIES
            BASE_PART = PASCAL_PART_BASE_CATEGORIES
            NOVEL_PART = PASCAL_PART_CATEGORIES
            self.conj = ':'
            self.base_obj_cat = [2,6,7,12,13,14,16]
        else: # 'partimagenet' in test_set
            OBJECT = PARTIMAGENET_SUPER_CATEGORIES
            BASE_PART = PASCAL_PART_CATEGORIES
            NOVEL_PART = PARTIMAGENET_CATEGORIES
            self.conj = ' '
            self.base_obj_cat = []

        self.base_part_id2name = {cat['id']: cat['name'].lower() for cat in BASE_PART}
        self.novel_part_name2id = {cat['name'].lower(): cat['id'] for cat in NOVEL_PART}
        self.novel_part = NOVEL_PART

        self.obj_id2name = {cat['id']: cat['name'].lower() for cat in OBJECT}
        self.obj_contid2catid = {i: x['id'] for i, x in enumerate(OBJECT)}

        self.base_part_contid2catid = {i: x['id'] for i, x in enumerate(BASE_PART)}

    def base2novel(self, base_part_id, obj_id_cont):
        obj_id = self.obj_contid2catid[obj_id_cont]
        base_full_name = self.base_part_id2name[base_part_id]
        base_part_name = base_full_name.split(':')[1]
        part_synonyms = PART_SYNONYMS[base_part_name] if base_part_name in PART_SYNONYMS \
            else [base_part_name]
        for base_part_synonym in part_synonyms:
            novel_full_name = self.obj_id2name[obj_id] + self.conj + base_part_synonym
            if novel_full_name in self.novel_part_name2id:
                # print(base_full_name, novel_full_name, self.novel_part_name2id[novel_full_name])
                return True, self.novel_part_name2id[novel_full_name]
        # print(base_full_name, self.obj_id2name[obj_id] + self.conj + base_part_name, -1)
        return False, -1

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {}
            file_name_set = input["file_name"].split('/')
            prediction["image"] = {
                "file_name": file_name_set[-2] + '/' + file_name_set[-1],
                # TODO: make file_name flexible
                "id": input["image_id"],
                "height": input["height"],
                "width": input["width"],
            }

            if "instances" in output:
                # instances = output["instances"].to(self._cpu_device)
                instances = output["instances"]
                prediction["instances"] = self.instances_to_coco_ann_json(
                    instances, input["image_id"], input["pos_category_ids"][0], self._cpu_device)

            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self, img_ids=None):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[AnnJsonGenerator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            file_path = self._output_dir
            PathManager.mkdirs(file_path[:-len(file_path.split('/')[-1])])

            ann_json = {}
            images, annotations = [], []
            anno_count = 0
            for predictions_per_img in predictions:
                for ann in predictions_per_img["instances"]:
                    anno_count += 1
                    ann['id'] = anno_count
                    annotations.append(ann)
                if len(predictions_per_img["instances"]) > 0:
                    images.append(predictions_per_img["image"])

            ann_json['images'] = images
            ann_json['annotations'] = annotations
            ann_json['categories'] = self.novel_part

            with open(file_path, "w") as f:
                json.dump(ann_json, f)

            return {"ann_json_path": file_path}

        return {"ann_json_path": ''}

    def instances_to_coco_ann_json(self, instances, img_id, obj_cat_id, cpu_device):
        num_instance = len(instances)
        if num_instance == 0:
            return []

        has_box = instances.has("pred_boxes")
        if has_box:
            boxes = instances.pred_boxes.to(cpu_device).tensor.numpy()
            boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            boxes = boxes.tolist()

        classes = instances.pred_classes.tolist()
        results = []
        for k in range(num_instance):
            base_part_id = classes[k]
            if obj_cat_id in self.base_obj_cat:
                base_part_id = self.base_part_contid2catid[base_part_id]
            matched, novel_part_id = self.base2novel(base_part_id, obj_cat_id)

            if not matched:
                continue
            result = {
                "image_id": img_id,
                # "category_id": classes[k],
                "category_id": novel_part_id,
            }
            if has_box:
                result["bbox"] = boxes[k]

            results.append(result)

        return results
