# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import json
import numpy as np
from torch.nn import functional as F

def load_obj2part_mapping(
    path='datasets/metadata/voc_obj2part_mapping.json'):
    mapping = json.load(open(path, 'r'))
    return mapping
