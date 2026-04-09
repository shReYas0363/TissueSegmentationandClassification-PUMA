"""Combined cross-entropy and Dice loss."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from task1.PretrainedAutoencoder.losses.dice import MulticlassDiceLoss


class CombinedCrossEntropyDiceLoss(nn.Module):
    """Default segmentation loss for the coursework experiments."""

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[Sequence[float]] = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        ce_class_weights = None
        if class_weights is not None:
            ce_class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("ce_class_weights", ce_class_weights, persistent=False)
        self.cross_entropy = nn.CrossEntropyLoss(weight=self.ce_class_weights)
        self.dice = MulticlassDiceLoss(
            num_classes=num_classes,
            smooth=smooth,
            class_weights=class_weights,
        )
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the total loss and detached component values."""
        ce_loss = self.cross_entropy(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        components = {
            "ce_loss": ce_loss.detach(),
            "dice_loss": dice_loss.detach(),
        }
        return total_loss, components
