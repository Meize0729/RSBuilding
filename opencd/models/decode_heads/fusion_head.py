# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule
from mmengine.model import BaseModule, ModuleList, caffe2_xavier_init
from torch import Tensor

from opencd.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from mmseg.models.utils import resize


@MODELS.register_module()
class FusionHead(BaseModule):

    def __init__(self, 
                 out_channels=256, 
                 out_size_index=0,
                 in_channels=[128, 256, 512, 1024],
                 norm_cfg: ConfigType = dict(type='GN', num_groups=32),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__()

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.init_cfg = init_cfg
        self.align_corners = None

        self.out_channels = out_channels
        self.out_size_index = out_size_index
        self.in_channels = in_channels

        num_inputs = len(self.in_channels)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.out_channels * num_inputs,
            out_channels=self.out_channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)


    def init_weights(self) -> None:
        """Initialize weights."""
        for i in range(0, len(self.in_channels)):
            caffe2_xavier_init(self.convs[i], bias=0)

        caffe2_xavier_init(self.fusion_conv, bias=0)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[self.out_size_index].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))     

        return out


