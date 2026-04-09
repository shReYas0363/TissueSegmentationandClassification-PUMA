"""Visualization helpers for reconstruction and segmentation results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = ("other", "tumor", "stroma")
CLASS_COLORS = {
    0: (0, 0, 255),
    1: (255, 0, 0),
    2: (0, 200, 0),
}


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {image.shape}")
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0.0, 255.0)
    return image.astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert an integer class-ID mask into an RGB visualization."""
    mask = np.asarray(mask, dtype=np.int64)
    color_mask = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a colorized mask on top of an RGB image."""
    image_uint8 = _to_uint8_image(image)
    color_mask = colorize_mask(mask)
    blended = (1.0 - alpha) * image_uint8.astype(np.float32) + alpha * color_mask.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_segmentation_preview(
    image: np.ndarray,
    target_mask: np.ndarray,
    prediction_mask: np.ndarray | None,
    output_path: str | Path,
    alpha: float = 0.45,
) -> None:
    """Save a 3-panel or 4-panel segmentation preview."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_uint8 = _to_uint8_image(image)
    target_color = colorize_mask(target_mask)
    target_overlay = overlay_mask(image_uint8, target_mask, alpha=alpha)

    panels = [
        ("Image", image_uint8),
        ("Target Mask", target_color),
        ("Target Overlay", target_overlay),
    ]
    if prediction_mask is not None:
        panels.append(("Prediction Overlay", overlay_mask(image_uint8, prediction_mask, alpha=alpha)))

    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for axis, (title, panel) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_reconstruction_preview(
    inputs: np.ndarray,
    targets: np.ndarray,
    reconstructions: np.ndarray,
    output_path: str | Path,
    max_items: int = 4,
) -> None:
    """Save a grid of reconstruction previews."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_items = min(max_items, len(inputs))
    figure, axes = plt.subplots(max_items, 3, figsize=(9, 3 * max_items))
    if max_items == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx in range(max_items):
        axes[row_idx, 0].imshow(_to_uint8_image(inputs[row_idx]))
        axes[row_idx, 0].set_title("Input")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(_to_uint8_image(targets[row_idx]))
        axes[row_idx, 1].set_title("Target")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(_to_uint8_image(reconstructions[row_idx]))
        axes[row_idx, 2].set_title("Reconstruction")
        axes[row_idx, 2].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
