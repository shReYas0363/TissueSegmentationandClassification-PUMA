"""ConvNeXt-Tiny wrapper for Task 2 nuclei classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights


def _resolve_weights(weights_name: str | None):
    if weights_name is None:
        return None

    normalized = str(weights_name).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    if normalized == "default":
        return ConvNeXt_Tiny_Weights.DEFAULT

    enum_name = str(weights_name).strip().upper()
    if hasattr(ConvNeXt_Tiny_Weights, enum_name):
        return getattr(ConvNeXt_Tiny_Weights, enum_name)
    raise ValueError(f"Unsupported ConvNeXt-Tiny weights setting: {weights_name}")


class ConvNeXtTinyClassifier(nn.Module):
    """Thin wrapper around torchvision ConvNeXt-Tiny with a 3-class head."""

    def __init__(self, num_classes: int = 3, weights: str | None = "default") -> None:
        super().__init__()
        resolved_weights = _resolve_weights(weights)
        self.model = models.convnext_tiny(weights=resolved_weights)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward one batch through the classifier."""
        return self.model(inputs)
