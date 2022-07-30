# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import warnings
import torch.nn.functional as F
from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import gradcheck, Variable

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .SwinTransformer import SwinTransformerSys

from .UPerNet import UPerHead, FCNHead
from .CrossAttention import CrossAttention

logger = logging.getLogger(__name__)


class SwinDRNet(nn.Module):
    """ SwinDRNet.
        A PyTorch impl of SwinDRNet, a depth restoration network proposed in: 
        `Domain Randomization-Enhanced Depth Simulation and Restoration for 
        Perceiving and Grasping Specular and Transparent Objects' (ECCV2022)
    """

    def __init__(self, config, img_size=224, num_classes=3):
        super(SwinDRNet, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.img_size = img_size

        self.backbone_rgb_branch = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        self.backbone_xyz_branch = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

        # self.decode_head_sem_seg = UPerHead(num_classes=self.num_classes, img_size = self.img_size)
        # self.decode_head_coord = UPerHead(num_classes=3, img_size = self.img_size)
 
        self.decode_head_depth_restoration = UPerHead(num_classes=1, in_channels=[288, 576, 1152, 2304], img_size = self.img_size)
        self.decode_head_confidence = UPerHead(num_classes=2, in_channels=[288, 576, 1152, 2304], img_size = self.img_size)

        self.cross_attention_0 = CrossAttention(in_channel=96, depth=1, num_heads=1)
        self.cross_attention_1 = CrossAttention(in_channel=192, depth=1, num_heads=1)
        self.cross_attention_2 = CrossAttention(in_channel=384, depth=1, num_heads=1)
        self.cross_attention_3 = CrossAttention(in_channel=768, depth=1, num_heads=1)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, rgb, depth):
        """Forward function."""

        rgb = rgb.repeat(1,3,1,1) if rgb.size()[1] == 1 else rgb       # B, C, H, W
        depth = depth.repeat(1,3,1,1) if depth.size()[1] == 1 else xyz       # B, C, H, W        
        
        # depth = torch.unsqueeze(xyz[:, 2, :, :], 1)
        # depth = depth.repeat(1, 3, 1, 1)
                
        input_org_shape = rgb.shape[2:]
        rgb_feature = self.backbone_rgb_branch(rgb)
        depth_feature = self.backbone_xyz_branch(depth)
    
        shortcut = torch.unsqueeze(depth[:, 2, :, :], 1)

        # fusion
        x = []
        out = self.cross_attention_0(tuple([rgb_feature[0], depth_feature[0]]))       # [B, 96, 56, 56]
        x.append(torch.cat((out, rgb_feature[0], depth_feature[0]), 1))
        out = self.cross_attention_1(tuple([rgb_feature[1], depth_feature[1]]))       # [B, 192, 28, 28]
        x.append(torch.cat((out, rgb_feature[1], depth_feature[1]), 1))
        out = self.cross_attention_2(tuple([rgb_feature[2], depth_feature[2]]))       # [B, 384, 14, 14]
        x.append(torch.cat((out, rgb_feature[2], depth_feature[2]), 1))
        out = self.cross_attention_3(tuple([rgb_feature[3], depth_feature[3]]))       # [B, 768, 7, 7]
        x.append(torch.cat((out, rgb_feature[3], depth_feature[3]), 1))
        
        # pred_sem_seg = self.decode_head_sem_seg(x, input_org_shape)
        # pred_coord = self.decode_head_coord(x, input_org_shape)
        pred_depth_initial = self.decode_head_depth_restoration(x, input_org_shape)
        confidence = self.softmax(self.decode_head_confidence(x, input_org_shape))

        confidence_depth = confidence[:, 0, :, :].unsqueeze(1)
        confidence_initial = confidence[:, 1, :, :].unsqueeze(1)

        pred_depth = confidence_depth * shortcut + confidence_initial * pred_depth_initial

        return pred_depth, pred_depth_initial, confidence_depth, confidence_initial# , pred_sem_seg, pred_coord


    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.backbone_rgb_branch.init_weights(pretrained=pretrained)
        self.backbone_xyz_branch.init_weights(pretrained=pretrained)
        self.decode_head_confidence.init_weights()
        self.decode_head_depth_restoration.init_weights()
        self.cross_attention_0.init_weights()
        self.cross_attention_1.init_weights()
        self.cross_attention_2.init_weights()
        self.cross_attention_3.init_weights()
        # self.decode_head_sem_seg.init_weights()
        # self.decode_head_coord.init_weights()
        