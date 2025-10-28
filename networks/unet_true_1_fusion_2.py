from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
import torch.nn as nn
import monai
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, export

class UNet_true_1_fusion_2(nn.Module):
    '''
    bottom add skip add
    '''
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        act: tuple | str = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:
        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 1:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
                bottom_flag = False
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_down_layer(c,c,1,is_top)
                upc = c
                bottom_flag = True

            down1 = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            down2 = self._get_down_layer(inc, c, s, is_top)
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path
            # return self._get_connection_block(down1, down2, up, subblock)
            if bottom_flag:
                return CustomContainer(DualInputModule(down1, down2, bottom_flag), subblock, up, bottom_flag)
            return CustomContainer(DualInputModule(down1, down2), DualSkipConnection(subblock), up)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    # def _get_connection_block(self, down_path_1: nn.Module, down_path_2: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
    #     """
    #     Returns the block object defining a layer of the UNet structure including the implementation of the skip
    #     between encoding (down) and decoding (up) sides of the network.
    #
    #     Args:
    #         down_path: encoding half of the layer
    #         up_path: decoding half of the layer
    #         subblock: block defining the next layer in the network.
    #     Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
    #     """
    #     return nn.Sequential(DualInputModule(down_path_1, down_path_2), DualSkipConnection(subblock), up_path)  # nn.Sequential 只接受一个输入,而forward时model有两个输入，这样会报错

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if is_top:
            mod = Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
        else:
            if self.num_res_units > 0:
                mod = ResidualUnit(
                    self.dimensions,
                    in_channels,
                    out_channels,
                    strides=strides,
                    kernel_size=self.kernel_size,
                    subunits=self.num_res_units,
                    act=self.act,
                    norm=self.norm,
                    dropout=self.dropout,
                    bias=self.bias,
                    adn_ordering=self.adn_ordering,
                )
                return mod
            mod = Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
        return mod

    # def _get_bottom_layer(self, in_channels: int, out_channels: int, strides: int, bottom_flag=True) -> nn.Module:
    #     """
    #     Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.
    #
    #     Args:
    #         in_channels: number of input channels.
    #         out_channels: number of output channels.
    #     """
    #     # return self._get_down_layer(in_channels, out_channels, 1, False)
    #     return DualInputModule(
    #         self._get_down_layer(in_channels, out_channels, strides, False),
    #         self._get_down_layer(in_channels, out_channels, strides, False),
    #         bottom_flag
    #     )

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Convolution | nn.Sequential

        if is_top:  # 相当于是最浅层后面接了个3*3卷积
            conv = Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.up_kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                conv_only=True,  # 为True的话就没有激活函数和正则化
                adn_ordering=self.adn_ordering,
            )

        else:
            conv = Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.up_kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                is_transposed=True,
                adn_ordering=self.adn_ordering,
            )

            if self.num_res_units > 0:
                ru = ResidualUnit(
                    self.dimensions,
                    out_channels,
                    out_channels,
                    strides=1,
                    kernel_size=self.kernel_size,
                    subunits=1,
                    act=self.act,
                    norm=self.norm,
                    dropout=self.dropout,
                    bias=self.bias,
                    last_conv_only=is_top,
                    adn_ordering=self.adn_ordering,
                )
                conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = self.model(x1, x2)
        return x


class DualInputModule(nn.Module):
    def __init__(self, down1: nn.Module, down2: nn.Module, bottom_flag=False):
        super(DualInputModule, self).__init__()
        self.down1 = down1
        self.down2 = down2
        self.bottom_flag = bottom_flag

    def forward(self, x1, x2):
        if self.bottom_flag:
            return torch.add(self.down1(x1), self.down2(x2))
        return self.down1(x1), self.down2(x2)


class DualSkipConnection(nn.Module):
    def __init__(self, subblock: nn.Module, dim: int = 1):
        super(DualSkipConnection, self).__init__()
        self.subblock = subblock
        self.dim = dim

    def forward(self, x1, x2):
        y = self.subblock(x1, x2)
        x = torch.add(x1,x2)
        return torch.cat([x,y],dim=self.dim)


class CustomContainer(nn.Module):  # 代替nn.Sequential，接受两个输入
    def __init__(self, dual_input_module: DualInputModule, subblock: nn.Module, up: nn.Module, bottom_flag=False):
        super(CustomContainer, self).__init__()
        self.dual_input_module = dual_input_module
        self.subblock = subblock
        self.up = up
        self.bottom_flag=bottom_flag

    def forward(self, x1, x2):
        if self.bottom_flag:
            x = self.dual_input_module(x1, x2)
            x=self.subblock(x)
        else:
            x1, x2 = self.dual_input_module(x1, x2)
            x = self.subblock(x1, x2)
        x = self.up(x)
        return x