# Copyright (c) Facebook, Inc. and its affiliates.
# --------------------------------------------------------
# Modified from Wetectron (https://github.com/NVlabs/wetectron)
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# --------------------------------------------------------
import math
from os.path import join
import numpy as np
import copy
from functools import partial

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers.batch_norm import get_norm, FrozenBatchNorm2d
from detectron2.modeling.backbone import Backbone


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x


class VGG_Base(Backbone):
    def __init__(self, features, cfg, init_weights=True):
        super(VGG_Base, self).__init__()
        self.features = features
        if init_weights:
            self._initialize_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_AT)

    def forward(self, x):
        x = self.features(x)
        return [x]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        assert freeze_at in [1, 2, 3, 4, 5]
        layer_index = [5, 10, 17, 23, 29]
        for layer in range(layer_index[freeze_at - 1]):
            for p in self.features[layer].parameters(): p.requires_grad = False

    def output_shape(self):
        return 512

def make_layers(cfg, dim_in=3, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'I':
            layers += [Identity()]
        # following OICR paper, make conv5_x layers to have dilation=2
        elif isinstance(v, str) and '-D' in v:
            _v = int(v.split('-')[0])
            conv2d = nn.Conv2d(in_channels, _v, kernel_size=3, padding=2, dilation=2)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(_v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = _v
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # remove the last relu
    return nn.Sequential(*layers[:-1])


vgg_cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG16-OICR': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'I', '512-D', '512-D', '512-D'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

@BACKBONE_REGISTRY.register()
def build_vgg16_backbone(cfg, input_shape):
    dim_in = 3
    archi_name = cfg.MODEL.BACKBONE.CONV_BODY
    body = VGG_Base(make_layers(vgg_cfg[archi_name], dim_in), cfg)
    # model = nn.Sequential(OrderedDict([("body", body)]))
    # model.out_shape = 512
    # return model
    return body
