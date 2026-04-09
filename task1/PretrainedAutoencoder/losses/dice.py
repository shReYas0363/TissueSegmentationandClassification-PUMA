"""Multiclass Dice loss."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassDiceLoss(nn.Module):
    """Dice loss for multiclass semantic segmentation."""

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1.0,
        class_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.smooth = float(smooth)
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            weights = None
        self.register_buffer("class_weights", weights, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multiclass Dice loss from logits and integer targets."""
        probabilities = F.softmax(logits, dim=1)
        one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probabilities * one_hot_targets).sum(dim=dims)
        cardinality = probabilities.sum(dim=dims) + one_hot_targets.sum(dim=dims)
        dice_scores = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_scores

        if self.class_weights is not None:
            weights = self.class_weights / self.class_weights.sum()
            return torch.sum(weights * dice_loss)
        return dice_loss.mean()
