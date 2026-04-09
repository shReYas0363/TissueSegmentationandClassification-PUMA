"""Reusable CNN blocks for encoders and decoders."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm(norm_type: str, channels: int) -> nn.Module:
    """Build a normalization layer."""
    norm_type = norm_type.lower()
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "group":
        num_groups = min(8, channels)
        while channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    raise ValueError(f"Unsupported norm type: {norm_type}")


def build_activation(name: str) -> nn.Module:
    """Build an activation layer."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvNormAct(nn.Module):
    """Convolution followed by normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        norm_type: str = "batch",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            build_norm(norm_type, out_channels),
            build_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """Two stacked Conv-Norm-Activation layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = [
            ConvNormAct(in_channels, out_channels, norm_type=norm_type, activation=activation),
            ConvNormAct(out_channels, out_channels, norm_type=norm_type, activation=activation),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownsampleBlock(nn.Module):
    """Stride-2 downsampling followed by feature refinement."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(
                in_channels,
                out_channels,
                stride=2,
                norm_type=norm_type,
                activation=activation,
            ),
            DoubleConv(
                out_channels,
                out_channels,
                norm_type=norm_type,
                activation=activation,
                dropout=dropout,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock(nn.Module):
    """Upsample by 2x and refine features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = DoubleConv(
            in_channels,
            out_channels,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class SkipFusionBlock(nn.Module):
    """Upsample and fuse with a skip connection."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = DoubleConv(
            in_channels + skip_channels,
            out_channels,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
