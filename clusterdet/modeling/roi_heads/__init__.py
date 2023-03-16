# Copyright (c) Facebook, Inc. and its affiliates.
from .roi_heads_oln import OLNROIHeads
from .roi_heads_detic import DeticROIHeads, DeticRes5ROIHeads, DeticCascadeROIHeads
from .roi_heads_weak import Res5WeakROIHeads, VGG16WeakROIHeads
from .roi_heads_joint import JointROIHeads
from .roi_heads_vlm import VLMROIHeads
