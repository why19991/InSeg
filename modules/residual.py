from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as functional
import torch

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)

class ResidualBlock(nn.Module):
    """Configurable residual block

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : list of int
        Number of channels in the internal feature maps. Can either have two or three elements: if three construct
        a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
        `3 x 3` then `1 x 1` convolutions.
    stride : int
        Stride of the first `3 x 3` convolution
    dilation : int
        Dilation to apply to the `3 x 3` convolutions.
    groups : int
        Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
        bottleneck blocks.
    norm_act : callable
        Function to create normalization / activation Module.
    dropout: callable
        Function to create Dropout Module.
    """

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        dilation=1,
        groups=1,
        norm_act=nn.BatchNorm2d,
        dropout=None,
        last=False,
        block_id=None
    ):
        super(ResidualBlock, self).__init__()

        self.block_id = block_id

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        if not is_bottleneck:
            bn2 = norm_act(channels[1])
            bn2.activation = "identity"
            layers = [
                (
                    "conv1",
                    nn.Conv2d(
                        in_channels,
                        channels[0],
                        3,
                        stride=stride,
                        padding=dilation,
                        bias=False,
                        dilation=dilation
                    )
                ), ("bn1", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(
                        channels[0],
                        channels[1],
                        3,
                        stride=1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation
                    )
                ), ("bn2", bn2)
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            bn3 = norm_act(channels[2])
            bn3.activation = "identity"
            layers = [
                ("conv1", nn.Conv2d(in_channels, channels[0], 1, stride=1, padding=0, bias=False)),
                ("bn1", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(channels[0],channels[1],3,stride=stride,padding=dilation,bias=False,groups=groups,dilation=dilation)
                ),
                ("bn2", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False)),
                ("bn3", bn3)
            ]

            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(
                in_channels, channels[-1], 1, stride=stride, padding=0, bias=False
            )
            self.proj_bn = norm_act(channels[-1])
            self.proj_bn.activation = "identity"

        self._last = last

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            residual = self.proj_conv(x)
            residual = self.proj_bn(residual)
        else:
            residual = x

        x = self.convs(x) + residual

        if self.convs.bn1.activation == "leaky_relu":
            act = functional.leaky_relu(
                x, negative_slope=self.convs.bn1.activation_param, inplace=not self._last
            )
        elif self.convs.bn1.activation == "elu":
            act = functional.elu(x, alpha=self.convs.bn1.activation_param, inplace=not self._last)
        elif self.convs.bn1.activation == "identity":
            act = x

        if self._last:
            return act, x
        return act


class IdentityResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        dilation=1,
        groups=1,
        norm_act=nn.BatchNorm2d,
        dropout=None
    ):
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps. Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        """
        super(IdentityResidualBlock, self).__init__()

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                (
                    "conv1",
                    nn.Conv2d(
                        in_channels,
                        channels[0],
                        3,
                        stride=stride,
                        padding=dilation,
                        bias=False,
                        dilation=dilation
                    )
                ), ("bn2", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(
                        channels[0],
                        channels[1],
                        3,
                        stride=1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation
                    )
                )
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                (
                    "conv1",
                    nn.Conv2d(in_channels, channels[0], 1, stride=stride, padding=0, bias=False)
                ), ("bn2", norm_act(channels[0])),
                (
                    "conv2",
                    nn.Conv2d(
                        channels[0],
                        channels[1],
                        3,
                        stride=1,
                        padding=dilation,
                        bias=False,
                        groups=groups,
                        dilation=dilation
                    )
                ), ("bn3", norm_act(channels[1])),
                ("conv3", nn.Conv2d(channels[1], channels[2], 1, stride=1, padding=0, bias=False))
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))

        if need_proj_conv:
            self.proj_conv = nn.Conv2d(
                in_channels, channels[-1], 1, stride=stride, padding=0, bias=False
            )

    def forward(self, x):
        if hasattr(self, "proj_conv"):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out.add_(shortcut)

        return out
