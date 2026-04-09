"""Plain convolutional autoencoder built around the shared encoder."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from task1.PretrainedAutoencoder.models.blocks import UpsampleBlock
from task1.PretrainedAutoencoder.models.encoders import ConvEncoder


class Autoencoder(nn.Module):
    """Moderate convolutional autoencoder for raw tissue image reconstruction."""

    def __init__(
        self,
        encoder: ConvEncoder,
        out_channels: int = 3,
        decoder_channels: List[int] | None = None,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder

        if decoder_channels is None:
            decoder_channels = list(reversed(self.encoder.feature_channels))

        current_channels = self.encoder.bottleneck_channels
        self.decoder_blocks = nn.ModuleList()
        for channels in decoder_channels:
            self.decoder_blocks.append(
                UpsampleBlock(
                    current_channels,
                    channels,
                    norm_type=norm_type,
                    activation=activation,
                    dropout=dropout,
                )
            )
            current_channels = channels

        self.output_head = nn.Conv2d(current_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct an RGB image patch."""
        encoded = self.encoder(x)
        reconstruction = encoded["bottleneck"]
        for block in self.decoder_blocks:
            reconstruction = block(reconstruction)
        reconstruction = self.output_head(reconstruction)
        return torch.sigmoid(reconstruction)
