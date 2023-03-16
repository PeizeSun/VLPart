# Copyright (c) Facebook, Inc. and its affiliates.
from .timm import (
    build_timm_backbone,
    build_timm_fpn_backbone,
)
from .separable_fpn import (
    build_separable_resnet_fpn_backbone,
    build_separable_timm_fpn_backbone,
)
from .vgg16 import build_vgg16_backbone
from .swintransformer import build_swintransformer_fpn_backbone

from .clip_fpn import build_clip_fpn_backbone
