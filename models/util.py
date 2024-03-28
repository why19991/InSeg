import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


def try_index(scalar_or_list, i):
    try:
        return scalar_or_list[i]
    except TypeError:
        return scalar_or_list

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, norm_act,use_transpose=True):
        super(DecoderBlock, self).__init__()
        if use_transpose:
            self.up_op = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 3, stride=2, padding=1, output_padding=1)
        else:
            self.up_op = Upsample(scale_factor=2, align_corners=True)

        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.norm1 = norm_act(in_channels//2)
        # self.norm1 = nn.BatchNorm2d(in_channels // 2)
        # self.relu1 = nn.ReLU(inplace=True)

        # self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = norm_act(in_channels // 2)

        self.conv3 = nn.Conv2d(in_channels // 2, n_filters, 1)
        self.norm3 = norm_act(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.up_op(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        return x