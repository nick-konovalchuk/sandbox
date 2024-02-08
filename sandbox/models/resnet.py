from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn


class PadShortcut(nn.Module):
    """Variant A from the paper"""

    def __init__(self, pad: int):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.pad, self.pad))


class MaxPoolShortcut(nn.Module):
    """Almost variant A from the paper"""

    def __init__(self, pad: int):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor):
        return F.pad(F.max_pool2d(x, 2, 2), (0, 0, 0, 0, self.pad, self.pad))


class ConvShortcut(nn.Module):
    """Variant B from the paper"""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))


class BaseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 shortcut: Optional[str] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            match shortcut:
                case "pad":
                    self.shortcut = PadShortcut(out_channels // (stride * 2))
                case "conv":
                    self.shortcut = ConvShortcut(in_channels, out_channels, stride)
                case "max_pool":
                    self.shortcut = MaxPoolShortcut(out_channels // (stride * 2))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.shortcut(identity)

        return self.relu(x + identity)


class BlockGroup(nn.Module):
    def __init__(self, n: int, in_channels: int, out_channels: int, stride: int,
                 shortcut: Optional[str] = None):
        super().__init__()
        blocks = [BaseBlock(in_channels, out_channels, stride, shortcut)]
        for _ in range(1, n):
            blocks.append(BaseBlock(out_channels, out_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        return self.blocks(x)


class ResNet(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.conv1 = nn.Conv2d(**config.conv1)
        self.blocks = nn.Sequential(
            *[BlockGroup(**block_group) for block_group in config.block_groups]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(**config.linear)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.linear(x)
