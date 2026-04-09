"""Classification metrics and confusion-matrix helpers for Task 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


def logits_to_predictions(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits into hard-label predictions."""
    return torch.argmax(logits, dim=1)


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits into class probabilities."""
    return torch.softmax(logits, dim=1)


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute a confusion matrix from flat integer predictions and targets."""
    valid = (targets >= 0) & (targets < num_classes)
    predictions = predictions[valid].astype(np.int64)
    targets = targets[valid].astype(np.int64)
    indices = targets * num_classes + predictions
    matrix = np.bincount(indices, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes)


def classification_metrics_from_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str],
    eps: float = 1e-8,
) -> Dict[str, object]:
    """Compute Task 2 classification metrics from a confusion matrix."""
    confusion_matrix = confusion_matrix.astype(np.float64)
    true_positives = np.diag(confusion_matrix)
    false_positives = confusion_matrix.sum(axis=0) - true_positives
    false_negatives = confusion_matrix.sum(axis=1) - true_positives
    support = confusion_matrix.sum(axis=1)

    precision = np.full(len(class_names), np.nan, dtype=np.float64)
    recall = np.full(len(class_names), np.nan, dtype=np.float64)
    f1 = np.full(len(class_names), np.nan, dtype=np.float64)

    for idx in range(len(class_names)):
        precision_denom = true_positives[idx] + false_positives[idx]
        recall_denom = true_positives[idx] + false_negatives[idx]
        if precision_denom > 0:
            precision[idx] = (true_positives[idx] + eps) / (precision_denom + eps)
        if recall_denom > 0:
            recall[idx] = (true_positives[idx] + eps) / (recall_denom + eps)

        if not np.isnan(precision[idx]) and not np.isnan(recall[idx]):
            f1_denom = precision[idx] + recall[idx]
            if f1_denom > 0:
                f1[idx] = (2.0 * precision[idx] * recall[idx] + eps) / (f1_denom + eps)

    accuracy = float((true_positives.sum() + eps) / (confusion_matrix.sum() + eps))

    per_class = {}
    flat: Dict[str, float] = {}
    for idx, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": None if np.isnan(precision[idx]) else float(precision[idx]),
            "recall": None if np.isnan(recall[idx]) else float(recall[idx]),
            "f1": None if np.isnan(f1[idx]) else float(f1[idx]),
            "support": int(support[idx]),
        }
        flat[f"precision_{class_name}"] = float(0.0 if np.isnan(precision[idx]) else precision[idx])
        flat[f"recall_{class_name}"] = float(0.0 if np.isnan(recall[idx]) else recall[idx])
        flat[f"f1_{class_name}"] = float(0.0 if np.isnan(f1[idx]) else f1[idx])

    metrics: Dict[str, object] = {
        "accuracy": accuracy,
        "macro_precision": float(np.nanmean(precision)) if not np.isnan(precision).all() else 0.0,
        "macro_recall": float(np.nanmean(recall)) if not np.isnan(recall).all() else 0.0,
        "macro_f1": float(np.nanmean(f1)) if not np.isnan(f1).all() else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix.astype(np.int64).tolist(),
    }
    metrics.update(flat)
    return metrics


def per_class_rows(metrics: Dict[str, object], class_names: Sequence[str]) -> List[Dict[str, object]]:
    """Convert structured per-class metrics into CSV-friendly rows."""
    rows: List[Dict[str, object]] = []
    per_class = metrics.get("per_class", {})
    for class_name in class_names:
        payload = per_class.get(class_name, {})
        rows.append(
            {
                "class_name": class_name,
                "precision": payload.get("precision"),
                "recall": payload.get("recall"),
                "f1": payload.get("f1"),
                "support": payload.get("support"),
            }
        )
    return rows


def confusion_matrix_rows(confusion_matrix: np.ndarray, class_names: Sequence[str]) -> List[Dict[str, object]]:
    """Convert a confusion matrix into CSV rows with readable headers."""
    rows: List[Dict[str, object]] = []
    for target_idx, class_name in enumerate(class_names):
        row: Dict[str, object] = {"true_class": class_name}
        for pred_idx, pred_name in enumerate(class_names):
            row[f"pred_{pred_name}"] = int(confusion_matrix[target_idx, pred_idx])
        rows.append(row)
    return rows


def save_confusion_matrix_figure(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
) -> None:
    """Save a labeled confusion matrix heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix.astype(np.int64)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)

    threshold = matrix.max() / 2.0 if matrix.size > 0 else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = int(matrix[row_idx, col_idx])
            axis.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


@dataclass
class ClassificationMetricAccumulator:
    """Accumulate classification predictions across batches."""

    num_classes: int
    class_names: Sequence[str]

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated counts."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update the confusion matrix from hard-label tensors."""
        preds_np = predictions.detach().cpu().numpy().reshape(-1)
        targets_np = targets.detach().cpu().numpy().reshape(-1)
        self.confusion_matrix += compute_confusion_matrix(
            predictions=preds_np,
            targets=targets_np,
            num_classes=self.num_classes,
        )

    def update_from_logits(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Update the confusion matrix from raw logits."""
        self.update(logits_to_predictions(logits), targets)

    def compute(self) -> Dict[str, object]:
        """Return the accumulated metrics payload."""
        return classification_metrics_from_confusion_matrix(self.confusion_matrix, self.class_names)
