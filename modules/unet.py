import torch
import torch.nn as nn
import torch.nn.functional as functional

from models.util import try_index, DecoderBlock


class Unet(nn.Module):
    def __init__(self,
                 in_channels,
                 filters,
                 norm_act=nn.BatchNorm2d,
                 ):

        super(Unet, self).__init__()


        self.u_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters[3], kernel_size=1)
        self.u_conv2 = nn.Conv2d(in_channels=filters[1], out_channels=filters[0], kernel_size=1)

        self.u_decoder4 = DecoderBlock(filters[3], filters[2],norm_act=norm_act)
        self.u_decoder3 = DecoderBlock(filters[2], filters[1],norm_act=norm_act)
        self.u_decoder2 = DecoderBlock(filters[0], filters[0],norm_act=norm_act)
        self.u_decoder1 = DecoderBlock(filters[0], filters[0],norm_act=norm_act)

        self.u_conv1_new = nn.Conv2d(in_channels=in_channels, out_channels=filters[3], kernel_size=1)
        self.u_conv2_new = nn.Conv2d(in_channels=filters[1], out_channels=filters[0], kernel_size=1)

        self.u_decoder4_new = DecoderBlock(filters[3], filters[2], norm_act=norm_act)
        self.u_decoder3_new = DecoderBlock(filters[2], filters[1], norm_act=norm_act)
        self.u_decoder2_new = DecoderBlock(filters[0], filters[0], norm_act=norm_act)
        self.u_decoder1_new = DecoderBlock(filters[0], filters[0], norm_act=norm_act)

    def forward(self, x):

        e = functional.leaky_relu_(self.u_conv1(x[4]),negative_slope=0.01)
        d4 = self.u_decoder4(e+x[3]) + x[2]
        d3 = self.u_decoder3(d4) + x[1]
        e = functional.leaky_relu_(self.u_conv2(d3),negative_slope=0.01)
        d2 = self.u_decoder2(e+x[0])
        d1 = self.u_decoder1(d2)
        return functional.leaky_relu(d1,negative_slope=0.01)



class Unet_2(nn.Module):
    def __init__(self,
                 in_channels,
                 filters,
                 norm_act=nn.BatchNorm2d,
                 ):

        super(Unet_2, self).__init__()


        self.u_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters[3], kernel_size=1)
        self.u_conv2 = nn.Conv2d(in_channels=filters[1], out_channels=filters[0], kernel_size=1)

        self.u_decoder4 = DecoderBlock(filters[3], filters[2],norm_act=norm_act)
        self.u_decoder3 = DecoderBlock(filters[2], filters[1],norm_act=norm_act)
        self.u_decoder2 = DecoderBlock(filters[0], filters[0],norm_act=norm_act)
        self.u_decoder1 = DecoderBlock(filters[0], filters[0],norm_act=norm_act)

        self.u_conv1_new = nn.Conv2d(in_channels=in_channels, out_channels=filters[3], kernel_size=1)
        self.u_conv2_new = nn.Conv2d(in_channels=filters[1], out_channels=filters[0], kernel_size=1)

        self.u_decoder4_new = DecoderBlock(filters[3], filters[2], norm_act=norm_act)
        self.u_decoder3_new = DecoderBlock(filters[2], filters[1], norm_act=norm_act)
        self.u_decoder2_new = DecoderBlock(filters[0], filters[0], norm_act=norm_act)
        self.u_decoder1_new = DecoderBlock(filters[0], filters[0], norm_act=norm_act)

    def forward(self, x):

        # Decoder
        e = functional.leaky_relu_(self.u_conv1(x[4]),negative_slope=0.01)
        d4 = self.u_decoder4(e+x[3]) + x[2]
        d3 = self.u_decoder3(d4) + x[1]
        e = functional.leaky_relu_(self.u_conv2(d3),negative_slope=0.01)
        d2 = self.u_decoder2(e+x[0])
        d1 = self.u_decoder1(d2)

        e_new = functional.leaky_relu_(self.u_conv1_new(x[4]), negative_slope=0.01)
        d4_new = self.u_decoder4_new(e_new + x[3]) + x[2]
        d3_new = self.u_decoder3_new(d4_new) + x[1]
        e_new = functional.leaky_relu_(self.u_conv2_new(d3_new), negative_slope=0.01)
        d2_new = self.u_decoder2_new(e_new + x[0])
        d1_new = self.u_decoder1_new(d2_new)

        # r = torch.rand(1, d1.shape[1], 1, 1, dtype=torch.float32)
        # if self.training == False:
        #     r[:, :, :, :] = 1.0
        # weight_out_branch = torch.zeros_like(r)
        # weight_out_new_branch = torch.zeros_like(r)
        # weight_out_branch[r < 0.33] = 2.
        # weight_out_new_branch[r < 0.33] = 0.
        # weight_out_branch[(r < 0.66) * (r >= 0.33)] = 0.
        # weight_out_new_branch[(r < 0.66) * (r >= 0.33)] = 2.
        # weight_out_branch[r >= 0.66] = 1.
        # weight_out_new_branch[r >= 0.66] = 1.
        # out = d1 * weight_out_branch.to(d1.device) * 0.5 + d1_new * weight_out_new_branch.to(d1_new.device) * 0.5

        out = d1 * 0.5 + d1_new  * 0.5
        return functional.leaky_relu(out,negative_slope=0.01)

class Unet_Intermediate(nn.Module):
    def __init__(self,
                 in_channels,
                 filters,
                 norm_act=nn.BatchNorm2d,
                 ):

        super(Unet_Intermediate, self).__init__()


        # self.u_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters[3], kernel_size=1)
        self.u_conv2 = nn.Conv2d(in_channels=filters[1], out_channels=filters[0], kernel_size=1)

        self.u_decoder4 = DecoderBlock(filters[3], filters[2],norm_act=norm_act)
        self.u_decoder3 = DecoderBlock(filters[2], filters[1],norm_act=norm_act)
        self.u_decoder2 = DecoderBlock(filters[0], filters[0],norm_act=norm_act)
        self.u_decoder1 = DecoderBlock(filters[0], filters[0],norm_act=norm_act)

        # self.u_conv1_new = nn.Conv2d(in_channels=in_channels, out_channels=filters[3], kernel_size=1)
        # self.u_conv2_new = nn.Conv2d(in_channels=filters[1], out_channels=filters[0], kernel_size=1)

        # self.u_decoder4_new = DecoderBlock(filters[3], filters[2], norm_act=norm_act)
        # self.u_decoder3_new = DecoderBlock(filters[2], filters[1], norm_act=norm_act)
        # self.u_decoder2_new = DecoderBlock(filters[0], filters[0], norm_act=norm_act)
        # self.u_decoder1_new = DecoderBlock(filters[0], filters[0], norm_act=norm_act)

    def forward(self, x):
        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        # print(x[3].shape)
        # print(x[4].shape)
        # Decoder
        # e = functional.leaky_relu_(self.u_conv1(x[4]),negative_slope=0.01)
        d4 = self.u_decoder4(x[3]) + x[2]
        d3 = self.u_decoder3(d4) + x[1]
        e = functional.leaky_relu_(self.u_conv2(d3),negative_slope=0.01)
        d2 = self.u_decoder2(e+x[0])
        d1 = self.u_decoder1(d2)

        # e_new = functional.leaky_relu_(self.u_conv1_new(x[4]), negative_slope=0.01)
        # d4_new = self.u_decoder4_new(e_new + x[3]) + x[2]
        # d3_new = self.u_decoder3_new(d4_new) + x[1]
        # e_new = functional.leaky_relu_(self.u_conv2_new(d3_new), negative_slope=0.01)
        # d2_new = self.u_decoder2_new(e_new + x[0])
        # d1_new = self.u_decoder1_new(d2_new)
        #
        # r = torch.rand(1, d1.shape[1], 1, 1, dtype=torch.float32)
        # if self.training == False:
        #     r[:, :, :, :] = 1.0
        # weight_out_branch = torch.zeros_like(r)
        # weight_out_new_branch = torch.zeros_like(r)
        # weight_out_branch[r < 0.33] = 2.
        # weight_out_new_branch[r < 0.33] = 0.
        # weight_out_branch[(r < 0.66) * (r >= 0.33)] = 0.
        # weight_out_new_branch[(r < 0.66) * (r >= 0.33)] = 2.
        # weight_out_branch[r >= 0.66] = 1.
        # weight_out_new_branch[r >= 0.66] = 1.
        # out = d1 * weight_out_branch.to(d1.device) * 0.5 + d1_new * weight_out_new_branch.to(d1_new.device) * 0.5


        return functional.leaky_relu(d1,negative_slope=0.01)