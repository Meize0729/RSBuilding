# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader




def convert_vit(ckpt):

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        k = k.replace('backbone.', '')
        new_ckpt[k] = v
    new_ckpt['layers.7.attn.cross_pic_rel_pos_h'] = new_ckpt['layers.7.attn.rel_pos_h']
    new_ckpt['layers.7.attn.cross_pic_rel_pos_w'] = new_ckpt['layers.7.attn.rel_pos_w']
    new_ckpt['layers.15.attn.cross_pic_rel_pos_h'] = new_ckpt['layers.15.attn.rel_pos_h']
    new_ckpt['layers.15.attn.cross_pic_rel_pos_w'] = new_ckpt['layers.15.attn.rel_pos_w']
    new_ckpt['layers.23.attn.cross_pic_rel_pos_h'] = new_ckpt['layers.23.attn.rel_pos_h']
    new_ckpt['layers.23.attn.cross_pic_rel_pos_w'] = new_ckpt['layers.23.attn.rel_pos_w']
    new_ckpt['layers.31.attn.cross_pic_rel_pos_h'] = new_ckpt['layers.31.attn.rel_pos_h']
    new_ckpt['layers.31.attn.cross_pic_rel_pos_w'] = new_ckpt['layers.31.attn.rel_pos_w']
    # import pdb;pdb.set_trace()
    return new_ckpt



checkpoint = CheckpointLoader.load_checkpoint('/mnt/public/usr/wangmingze/pretrain/sam_vit_h_origin.pth', map_location='cpu')
state_dict = checkpoint['state_dict']
# import pdb; pdb.set_trace()
weight = convert_vit(state_dict)
mmengine.mkdir_or_exist(osp.dirname('/mnt/public/usr/wangmingze/pretrain/sam_vit_h_mm_allin.pth'))
torch.save(weight, '/mnt/public/usr/wangmingze/pretrain/sam_vit_h_mm_allin.pth')