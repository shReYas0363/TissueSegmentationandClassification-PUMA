"""Transforms for Task 2 nuclei classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def _validate_rgb_patch(image: np.ndarray, expected_size: int | None = None) -> np.ndarray:
    """Validate and normalize one RGB nuclei patch array."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB patch, got shape {image.shape}")
    if expected_size is not None and image.shape[:2] != (expected_size, expected_size):
        raise ValueError(
            f"Expected a {expected_size}x{expected_size} patch, got {image.shape[:2]}"
        )
    return image


def _maybe_hflip(image: np.ndarray, enabled: bool) -> np.ndarray:
    if enabled and np.random.rand() < 0.5:
        image = np.flip(image, axis=1).copy()
    return image


def _maybe_vflip(image: np.ndarray, enabled: bool) -> np.ndarray:
    if enabled and np.random.rand() < 0.5:
        image = np.flip(image, axis=0).copy()
    return image


def _maybe_rotate90(image: np.ndarray, enabled: bool) -> np.ndarray:
    if enabled:
        k = int(np.random.randint(0, 4))
        if k > 0:
            image = np.rot90(image, k=k, axes=(0, 1)).copy()
    return image


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert one HWC image array into a float CHW tensor in [0, 1]."""
    image = np.asarray(image)
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    image = np.clip(image, 0.0, 1.0)
    return torch.from_numpy(np.transpose(image, (2, 0, 1))).float()


def _sample_jitter_factor(amount: float) -> float:
    if amount <= 0:
        return 1.0
    low = max(0.0, 1.0 - amount)
    high = 1.0 + amount
    return float(np.random.uniform(low, high))


def _maybe_color_jitter(
    image_tensor: torch.Tensor,
    brightness: float,
    contrast: float,
    saturation: float,
) -> torch.Tensor:
    if brightness > 0:
        image_tensor = TF.adjust_brightness(image_tensor, _sample_jitter_factor(brightness))
    if contrast > 0:
        image_tensor = TF.adjust_contrast(image_tensor, _sample_jitter_factor(contrast))
    if saturation > 0:
        image_tensor = TF.adjust_saturation(image_tensor, _sample_jitter_factor(saturation))
    return image_tensor.clamp(0.0, 1.0)


@dataclass
class NucleiClassificationTransform:
    """Transform pipeline for generated and official Task 2 nuclei patches."""

    input_size: int
    expected_patch_size: int | None
    mean: Sequence[float]
    std: Sequence[float]
    train_hflip: bool = False
    train_vflip: bool = False
    train_rotate90: bool = False
    train_color_jitter_brightness: float = 0.0
    train_color_jitter_contrast: float = 0.0
    train_color_jitter_saturation: float = 0.0
    is_train: bool = True

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Apply Task 2 preprocessing and augmentation to one patch."""
        image = _validate_rgb_patch(image, expected_size=self.expected_patch_size)
        if self.is_train:
            image = _maybe_hflip(image, self.train_hflip)
            image = _maybe_vflip(image, self.train_vflip)
            image = _maybe_rotate90(image, self.train_rotate90)

        image_tensor = _image_to_tensor(image)
        image_tensor = TF.resize(
            image_tensor,
            [self.input_size, self.input_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if self.is_train:
            image_tensor = _maybe_color_jitter(
                image_tensor,
                brightness=self.train_color_jitter_brightness,
                contrast=self.train_color_jitter_contrast,
                saturation=self.train_color_jitter_saturation,
            )
        image_tensor = TF.normalize(image_tensor, mean=list(self.mean), std=list(self.std))
        return image_tensor


def build_nuclei_classification_transform(
    config: dict,
    is_train: bool,
) -> NucleiClassificationTransform:
    """Build the Task 2 transform pipeline from config."""
    dataset_cfg = config["dataset"]
    aug_cfg = config.get("augmentation", {})
    norm_cfg = config.get("normalization", {})
    return NucleiClassificationTransform(
        input_size=int(dataset_cfg.get("input_size", 224)),
        expected_patch_size=int(dataset_cfg.get("saved_patch_size", 100)),
        mean=norm_cfg.get("mean", [0.485, 0.456, 0.406]),
        std=norm_cfg.get("std", [0.229, 0.224, 0.225]),
        train_hflip=bool(aug_cfg.get("train_hflip", False)),
        train_vflip=bool(aug_cfg.get("train_vflip", False)),
        train_rotate90=bool(aug_cfg.get("train_rotate90", False)),
        train_color_jitter_brightness=float(aug_cfg.get("train_color_jitter_brightness", 0.0)),
        train_color_jitter_contrast=float(aug_cfg.get("train_color_jitter_contrast", 0.0)),
        train_color_jitter_saturation=float(aug_cfg.get("train_color_jitter_saturation", 0.0)),
        is_train=is_train,
    )
