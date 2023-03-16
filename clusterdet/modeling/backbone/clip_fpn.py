# Copyright (c) Facebook, Inc. and its affiliates.
import math
import pdb

import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool, _assert_strides_are_log2_contiguous


class CLIP_FPN(Backbone):
    def __init__(
        self,
        # bottom_up,
        strides,
        in_channels_per_feature,
        in_features,
        out_channels,
        norm="",
        top_block=None,
        fuse_type="sum",
        square_pad=0,
    ):
        super(CLIP_FPN, self).__init__()
        # assert isinstance(bottom_up, Backbone)
        assert in_features, in_features

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        # input_shapes = bottom_up.output_shape()
        # strides = [input_shapes[f].stride for f in in_features]
        # in_channels_per_feature = [input_shapes[f].channels for f in in_features]
        # print(strides)
        # print(in_channels_per_feature)
        # exit()
        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels_per_feature):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            stage = int(math.log2(strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = tuple(in_features)

        # self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     del self.bottom_up

    @property
    def size_divisibility(self):
        return self._size_divisibility

    @property
    def padding_constraints(self):
        return {"square_size": self._square_pad}

    def forward(self, x, bottom_up_features=None):
        assert bottom_up_features is not None
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
                zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

@BACKBONE_REGISTRY.register()
def build_clip_fpn_backbone(cfg, input_shape: ShapeSpec):
    # bottom_up = build_resnet_backbone(cfg, input_shape)
    # strides_per_feature = cfg.MODEL.CLIP_FPN.STRIDES
    # in_channels_per_feature = cfg.MODEL.CLIP_FPN.IN_CHANNELS
    # in_features = cfg.MODEL.CLIP_FPN.IN_FEATURES
    # out_channels = cfg.MODEL.CLIP_FPN.OUT_CHANNELS
    backbone = CLIP_FPN(
        # bottom_up=bottom_up,
        strides=cfg.MODEL.CLIP_FPN.STRIDES,
        in_channels_per_feature=cfg.MODEL.CLIP_FPN.IN_CHANNELS,
        in_features=cfg.MODEL.CLIP_FPN.IN_FEATURES,
        out_channels=cfg.MODEL.CLIP_FPN.OUT_CHANNELS,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone