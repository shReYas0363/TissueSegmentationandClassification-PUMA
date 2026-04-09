"""Segmentation metrics built from confusion matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch


DEFAULT_CLASS_NAMES = ("other", "tumor", "stroma")


def logits_to_predictions(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to hard class predictions."""
    return torch.argmax(logits, dim=1)


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute a confusion matrix from flat prediction and target arrays."""
    valid = (targets >= 0) & (targets < num_classes)
    predictions = predictions[valid].astype(np.int64)
    targets = targets[valid].astype(np.int64)
    indices = targets * num_classes + predictions
    matrix = np.bincount(indices, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes)


def metrics_from_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    eps: float = 1e-8,
) -> Dict[str, object]:
    """Compute segmentation metrics from a confusion matrix."""
    confusion_matrix = confusion_matrix.astype(np.float64)
    true_positives = np.diag(confusion_matrix)
    false_positives = confusion_matrix.sum(axis=0) - true_positives
    false_negatives = confusion_matrix.sum(axis=1) - true_positives
    support = confusion_matrix.sum(axis=1)

    dice = np.full(len(class_names), np.nan, dtype=np.float64)
    iou = np.full(len(class_names), np.nan, dtype=np.float64)
    precision = np.full(len(class_names), np.nan, dtype=np.float64)
    recall = np.full(len(class_names), np.nan, dtype=np.float64)
    f1 = np.full(len(class_names), np.nan, dtype=np.float64)

    for idx in range(len(class_names)):
        dice_denom = 2.0 * true_positives[idx] + false_positives[idx] + false_negatives[idx]
        iou_denom = true_positives[idx] + false_positives[idx] + false_negatives[idx]
        precision_denom = true_positives[idx] + false_positives[idx]
        recall_denom = true_positives[idx] + false_negatives[idx]
        if dice_denom > 0:
            dice[idx] = (2.0 * true_positives[idx] + eps) / (dice_denom + eps)
        if iou_denom > 0:
            iou[idx] = (true_positives[idx] + eps) / (iou_denom + eps)
        if precision_denom > 0:
            precision[idx] = (true_positives[idx] + eps) / (precision_denom + eps)
        if recall_denom > 0:
            recall[idx] = (true_positives[idx] + eps) / (recall_denom + eps)
        if not np.isnan(precision[idx]) and not np.isnan(recall[idx]):
            f1_denom = precision[idx] + recall[idx]
            if f1_denom > 0:
                f1[idx] = (2.0 * precision[idx] * recall[idx] + eps) / (f1_denom + eps)

    pixel_accuracy = float((true_positives.sum() + eps) / (confusion_matrix.sum() + eps))

    per_class = {}
    flat: Dict[str, float] = {}
    for idx, class_name in enumerate(class_names):
        per_class[class_name] = {
            "dice": None if np.isnan(dice[idx]) else float(dice[idx]),
            "iou": None if np.isnan(iou[idx]) else float(iou[idx]),
            "precision": None if np.isnan(precision[idx]) else float(precision[idx]),
            "recall": None if np.isnan(recall[idx]) else float(recall[idx]),
            "f1": None if np.isnan(f1[idx]) else float(f1[idx]),
            "support_pixels": int(support[idx]),
        }
        flat[f"dice_{class_name}"] = float(0.0 if np.isnan(dice[idx]) else dice[idx])
        flat[f"iou_{class_name}"] = float(0.0 if np.isnan(iou[idx]) else iou[idx])
        flat[f"precision_{class_name}"] = float(0.0 if np.isnan(precision[idx]) else precision[idx])
        flat[f"recall_{class_name}"] = float(0.0 if np.isnan(recall[idx]) else recall[idx])
        flat[f"f1_{class_name}"] = float(0.0 if np.isnan(f1[idx]) else f1[idx])

    metrics: Dict[str, object] = {
        "pixel_accuracy": pixel_accuracy,
        "mean_dice": float(np.nanmean(dice)) if not np.isnan(dice).all() else 0.0,
        "mean_iou": float(np.nanmean(iou)) if not np.isnan(iou).all() else 0.0,
        "mean_precision": float(np.nanmean(precision)) if not np.isnan(precision).all() else 0.0,
        "mean_recall": float(np.nanmean(recall)) if not np.isnan(recall).all() else 0.0,
        "mean_f1": float(np.nanmean(f1)) if not np.isnan(f1).all() else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix.astype(np.int64).tolist(),
    }
    metrics.update(flat)
    return metrics


def per_class_rows(metrics: Dict[str, object], class_names: Sequence[str]) -> List[Dict[str, object]]:
    """Convert structured per-class segmentation metrics into CSV-friendly rows."""
    rows: List[Dict[str, object]] = []
    per_class = metrics.get("per_class", {})
    for class_name in class_names:
        payload = per_class.get(class_name, {})
        rows.append(
            {
                "class_name": class_name,
                "dice": payload.get("dice"),
                "iou": payload.get("iou"),
                "precision": payload.get("precision"),
                "recall": payload.get("recall"),
                "f1": payload.get("f1"),
                "support_pixels": payload.get("support_pixels"),
            }
        )
    return rows


@dataclass
class SegmentationMetricAccumulator:
    """Accumulate a confusion matrix across batches."""

    num_classes: int
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the accumulated confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics using batched predictions and targets."""
        preds_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        self.confusion_matrix += compute_confusion_matrix(
            predictions=preds_np.reshape(-1),
            targets=targets_np.reshape(-1),
            num_classes=self.num_classes,
        )

    def update_from_logits(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics from logits."""
        self.update(logits_to_predictions(logits), targets)

    def compute(self) -> Dict[str, object]:
        """Compute final metrics."""
        return metrics_from_confusion_matrix(self.confusion_matrix, self.class_names)
