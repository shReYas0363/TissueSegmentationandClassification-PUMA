"""Patch sampling helpers for reconstruction and segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class SegmentationPatchSamplingConfig:
    """Configuration for soft class-aware segmentation patch acceptance."""

    max_retries: int = 5
    stroma_min_fraction: float = 0.05
    other_min_fraction: float = 0.05
    tumor_max_fraction: float = 0.95
    extreme_tumor_accept_prob: float = 0.25


def _normalize_patch_size(patch_size: int | Tuple[int, int]) -> Tuple[int, int]:
    """Convert a scalar or tuple patch size into `(height, width)` form."""
    if isinstance(patch_size, int):
        return patch_size, patch_size
    return int(patch_size[0]), int(patch_size[1])


def pad_image_to_patch_size(image: np.ndarray, patch_size: int | Tuple[int, int]) -> np.ndarray:
    """Pad an image so at least one valid patch of the desired size can be sampled."""
    patch_height, patch_width = _normalize_patch_size(patch_size)
    height, width = image.shape[:2]
    pad_height = max(0, patch_height - height)
    pad_width = max(0, patch_width - width)
    if pad_height == 0 and pad_width == 0:
        return image

    pad_spec = ((0, pad_height), (0, pad_width))
    if image.ndim == 3:
        pad_spec = pad_spec + ((0, 0),)
    return np.pad(image, pad_spec, mode="reflect")


def pad_mask_to_patch_size(mask: np.ndarray, patch_size: int | Tuple[int, int]) -> np.ndarray:
    """Pad a label mask using the background/other class."""
    patch_height, patch_width = _normalize_patch_size(patch_size)
    height, width = mask.shape[:2]
    pad_height = max(0, patch_height - height)
    pad_width = max(0, patch_width - width)
    if pad_height == 0 and pad_width == 0:
        return mask
    return np.pad(mask, ((0, pad_height), (0, pad_width)), mode="constant", constant_values=0)


def sample_random_patch_coordinates(
    height: int,
    width: int,
    patch_size: int | Tuple[int, int],
) -> Tuple[int, int]:
    """Sample top-left patch coordinates."""
    patch_height, patch_width = _normalize_patch_size(patch_size)
    max_top = max(height - patch_height, 0)
    max_left = max(width - patch_width, 0)
    top = int(np.random.randint(0, max_top + 1))
    left = int(np.random.randint(0, max_left + 1))
    return top, left


def crop_patch(array: np.ndarray, top: int, left: int, patch_size: int | Tuple[int, int]) -> np.ndarray:
    """Crop a spatial patch from an image or mask."""
    patch_height, patch_width = _normalize_patch_size(patch_size)
    return array[top : top + patch_height, left : left + patch_width, ...]


def compute_patch_class_fractions(mask_patch: np.ndarray) -> Dict[str, float]:
    """Compute the fraction of pixels belonging to each class."""
    total = float(mask_patch.size)
    return {
        "other_frac": float((mask_patch == 0).sum() / total),
        "tumor_frac": float((mask_patch == 1).sum() / total),
        "stroma_frac": float((mask_patch == 2).sum() / total),
    }


def sample_reconstruction_patch(
    image: np.ndarray,
    patch_size: int | Tuple[int, int],
) -> Dict[str, object]:
    """Sample a random image patch for autoencoder reconstruction."""
    image = pad_image_to_patch_size(image, patch_size)
    top, left = sample_random_patch_coordinates(image.shape[0], image.shape[1], patch_size)
    patch = crop_patch(image, top, left, patch_size)
    return {
        "image_patch": patch,
        "top": top,
        "left": left,
    }


def accept_segmentation_patch(
    fractions: Dict[str, float],
    config: SegmentationPatchSamplingConfig,
) -> bool:
    """Apply the soft acceptance rule for class-imbalanced segmentation patches."""
    if fractions["stroma_frac"] >= config.stroma_min_fraction:
        return True
    if fractions["other_frac"] >= config.other_min_fraction:
        return True
    if fractions["tumor_frac"] < config.tumor_max_fraction:
        return True
    return bool(np.random.rand() < config.extreme_tumor_accept_prob)


def sample_segmentation_patch(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int | Tuple[int, int],
    config: SegmentationPatchSamplingConfig,
) -> Dict[str, object]:
    """Sample a segmentation patch using the coursework acceptance rule."""
    image = pad_image_to_patch_size(image, patch_size)
    mask = pad_mask_to_patch_size(mask, patch_size)

    last_result: Dict[str, object] | None = None
    for _ in range(max(int(config.max_retries), 1)):
        top, left = sample_random_patch_coordinates(image.shape[0], image.shape[1], patch_size)
        image_patch = crop_patch(image, top, left, patch_size)
        mask_patch = crop_patch(mask, top, left, patch_size)
        fractions = compute_patch_class_fractions(mask_patch)
        last_result = {
            "image_patch": image_patch,
            "mask_patch": mask_patch,
            "top": top,
            "left": left,
            **fractions,
        }
        if accept_segmentation_patch(fractions, config):
            return last_result

    if last_result is None:
        raise RuntimeError("Failed to sample a segmentation patch.")
    return last_result
