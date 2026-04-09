"""Reusable convolutional encoder for all experiments."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from task1.PretrainedAutoencoder.models.blocks import DoubleConv, DownsampleBlock


class ConvEncoder(nn.Module):
    """Moderate shared encoder with intermediate features for skip connections."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_stages: int = 4,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_stages < 2:
            raise ValueError("num_stages must be at least 2.")

        feature_channels = [base_channels * (2**idx) for idx in range(num_stages)]
        self.feature_channels = feature_channels
        self.bottleneck_channels = feature_channels[-1] * 2

        self.stem = DoubleConv(
            in_channels,
            feature_channels[0],
            norm_type=norm_type,
            activation=activation,
            dropout=dropout,
        )

        stages = []
        in_ch = feature_channels[0]
        for out_ch in feature_channels[1:]:
            stages.append(
                DownsampleBlock(
                    in_ch,
                    out_ch,
                    norm_type=norm_type,
                    activation=activation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.bottleneck = DownsampleBlock(
            feature_channels[-1],
            self.bottleneck_channels,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor] | torch.Tensor]:
        """Return intermediate feature maps and the bottleneck representation."""
        features: List[torch.Tensor] = []
        x = self.stem(x)
        features.append(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        bottleneck = self.bottleneck(x)
        return {
            "features": features,
            "bottleneck": bottleneck,
        }
