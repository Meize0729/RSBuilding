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

@MODELS.register_module()
class UpsampleDecoder(BaseModule):


    def __init__(self,
                 channels=256,
                 upsample_time=4,
                 norm_cfg: ConfigType = dict(type='GN', num_groups=32),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.upsample_time = upsample_time
        self.channels = channels
        self.use_bias = norm_cfg is None
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        # 每次的conv + upsample + conv
        for _ in range(0, self.upsample_time):
            lateral_conv = ConvModule(
                channels,
                channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            output_conv = ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.last_feat_conv = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=self.use_bias,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.mask_feature = Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1)

    def init_weights(self) -> None:
        """Initialize weights."""
        for i in range(0, self.upsample_time):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)
        caffe2_xavier_init(self.last_feat_conv, bias=0)

    def forward(self, feats):

        y = self.last_feat_conv(feats)
        for i in range(0, self.upsample_time):
            # y = self.lateral_convs[i](y)
            y = F.interpolate(y, size=[y.shape[-2]*2, y.shape[-1]*2], mode='nearest')
            y = self.output_convs[i](y)
        mask_feature = self.mask_feature(y)
        return mask_feature