"""Factory helpers for shared encoder, autoencoder, and segmentation models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from task1.PretrainedAutoencoder.models.autoencoder import Autoencoder
from task1.PretrainedAutoencoder.models.encoders import ConvEncoder
from task1.PretrainedAutoencoder.models.segmentation_unet_like import UNetLikeSegmentationModel
from task1.PretrainedAutoencoder.utils.model_utils import load_encoder_weights


def build_encoder(model_config: Dict[str, object]) -> ConvEncoder:
    """Build the shared convolutional encoder from config."""
    return ConvEncoder(
        in_channels=3,
        base_channels=int(model_config.get("encoder_base_channels", 32)),
        num_stages=int(model_config.get("encoder_num_stages", 4)),
        norm_type=str(model_config.get("norm_type", "batch")),
        activation=str(model_config.get("activation", "relu")),
        dropout=float(model_config.get("dropout", 0.0)),
    )


def build_autoencoder(config: Dict[str, object]) -> Autoencoder:
    """Build an autoencoder from config."""
    model_config = config["model"]
    encoder = build_encoder(model_config)
    return Autoencoder(
        encoder=encoder,
        out_channels=3,
        decoder_channels=model_config.get("decoder_channels"),
        norm_type=str(model_config.get("norm_type", "batch")),
        activation=str(model_config.get("activation", "relu")),
        dropout=float(model_config.get("dropout", 0.0)),
    )


def build_segmentation_model(config: Dict[str, object]):
    """Build the UNet-like segmentation model using the shared encoder."""
    model_config = config["model"]
    encoder = build_encoder(model_config)

    pretrained_encoder_path = model_config.get("pretrained_encoder_path")
    if pretrained_encoder_path:
        checkpoint_path = Path(str(pretrained_encoder_path))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Pretrained encoder weights not found: {checkpoint_path}")
        load_encoder_weights(encoder, checkpoint_path)

    model_name = str(model_config.get("name", "unet_like_decoder")).lower()
    if model_name not in {"unet_like_decoder", "unet_like"}:
        raise ValueError(f"Unsupported segmentation model name: {model_name}")

    num_classes = int(config["dataset"].get("num_classes", 3))
    freeze_encoder = bool(model_config.get("freeze_encoder", True))
    return UNetLikeSegmentationModel(
        encoder=encoder,
        num_classes=num_classes,
        decoder_channels=model_config.get("decoder_channels"),
        freeze_encoder=freeze_encoder,
        norm_type=str(model_config.get("norm_type", "batch")),
        activation=str(model_config.get("activation", "relu")),
        dropout=float(model_config.get("dropout", 0.0)),
    )
