"""UNet-like segmentation decoder using skip connections from the pretrained encoder."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from task1.PretrainedAutoencoder.models.blocks import SkipFusionBlock
from task1.PretrainedAutoencoder.models.encoders import ConvEncoder
from task1.PretrainedAutoencoder.utils.model_utils import freeze_module


class UNetLikeSegmentationModel(nn.Module):
    """Skip-connected decoder built on top of the shared encoder."""

    def __init__(
        self,
        encoder: ConvEncoder,
        num_classes: int = 3,
        decoder_channels: Sequence[int] | None = None,
        freeze_encoder: bool = True,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            freeze_module(self.encoder)

        skip_channels = list(reversed(self.encoder.feature_channels))
        if decoder_channels is None:
            decoder_channels = skip_channels
        decoder_channels = [int(channels) for channels in decoder_channels]
        if len(decoder_channels) != len(skip_channels):
            raise ValueError(
                "decoder_channels must match the number of skip stages: "
                f"expected {len(skip_channels)}, got {len(decoder_channels)}"
            )
        current_channels = self.encoder.bottleneck_channels

        self.decoder_blocks = nn.ModuleList()
        for skip_channels_i, decoder_channels_i in zip(skip_channels, decoder_channels):
            self.decoder_blocks.append(
                SkipFusionBlock(
                    current_channels,
                    skip_channels_i,
                    decoder_channels_i,
                    norm_type=norm_type,
                    activation=activation,
                    dropout=dropout,
                )
            )
            current_channels = decoder_channels_i

        self.output_head = nn.Conv2d(current_channels, num_classes, kernel_size=1)

    def _encode(self, x: torch.Tensor) -> dict:
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(x)
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict segmentation logits using skip fusion."""
        encoded = self._encode(x)
        x = encoded["bottleneck"]
        skip_features = list(reversed(encoded["features"]))
        for block, skip in zip(self.decoder_blocks, skip_features):
            x = block(x, skip)
        return self.output_head(x)
