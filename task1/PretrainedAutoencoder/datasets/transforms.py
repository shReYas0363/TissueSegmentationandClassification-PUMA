"""Transforms for reconstruction and segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


def _normalize_image_array(image: np.ndarray) -> np.ndarray:
    """Convert an image to float32 in [0, 1]."""
    image = np.asarray(image)
    original_dtype = image.dtype
    image = image.astype(np.float32)

    if np.issubdtype(original_dtype, np.integer):
        image /= float(np.iinfo(original_dtype).max)
    elif image.max() > 1.0:
        image /= float(image.max())

    return np.clip(image, 0.0, 1.0)


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert HWC float image to CHW tensor."""
    image = _normalize_image_array(image)
    return torch.from_numpy(np.transpose(image, (2, 0, 1))).float()


def _mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """Convert a mask to an integer tensor."""
    return torch.from_numpy(mask.astype(np.int64))


def _maybe_hflip(image: np.ndarray, mask: np.ndarray | None, enabled: bool) -> Tuple[np.ndarray, np.ndarray | None]:
    if enabled and np.random.rand() < 0.5:
        image = np.flip(image, axis=1).copy()
        if mask is not None:
            mask = np.flip(mask, axis=1).copy()
    return image, mask


def _maybe_vflip(image: np.ndarray, mask: np.ndarray | None, enabled: bool) -> Tuple[np.ndarray, np.ndarray | None]:
    if enabled and np.random.rand() < 0.5:
        image = np.flip(image, axis=0).copy()
        if mask is not None:
            mask = np.flip(mask, axis=0).copy()
    return image, mask


def _maybe_rotate90(image: np.ndarray, mask: np.ndarray | None, enabled: bool) -> Tuple[np.ndarray, np.ndarray | None]:
    if enabled:
        k = int(np.random.randint(0, 4))
        if k > 0:
            image = np.rot90(image, k=k, axes=(0, 1)).copy()
            if mask is not None:
                mask = np.rot90(mask, k=k, axes=(0, 1)).copy()
    return image, mask


@dataclass
class ReconstructionTransform:
    """Autoencoder transform pipeline."""

    train_hflip: bool = False
    train_vflip: bool = False
    train_rotate90: bool = False
    is_train: bool = True

    def __call__(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform one reconstruction patch."""
        if self.is_train:
            image, _ = _maybe_hflip(image, None, self.train_hflip)
            image, _ = _maybe_vflip(image, None, self.train_vflip)
            image, _ = _maybe_rotate90(image, None, self.train_rotate90)

        image_tensor = _image_to_tensor(image)
        return image_tensor, image_tensor.clone()


@dataclass
class SegmentationTransform:
    """Joint image/mask segmentation transform pipeline."""

    train_hflip: bool = False
    train_vflip: bool = False
    train_rotate90: bool = False
    is_train: bool = True

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform one segmentation image/mask pair."""
        if self.is_train:
            image, mask = _maybe_hflip(image, mask, self.train_hflip)
            image, mask = _maybe_vflip(image, mask, self.train_vflip)
            image, mask = _maybe_rotate90(image, mask, self.train_rotate90)

        return _image_to_tensor(image), _mask_to_tensor(mask)


def build_reconstruction_transform(config: dict, is_train: bool) -> ReconstructionTransform:
    """Build a reconstruction transform from config."""
    aug = config.get("augmentation", {})
    return ReconstructionTransform(
        train_hflip=bool(aug.get("train_hflip", False)),
        train_vflip=bool(aug.get("train_vflip", False)),
        train_rotate90=bool(aug.get("train_rotate90", False)),
        is_train=is_train,
    )


def build_segmentation_transform(config: dict, is_train: bool) -> SegmentationTransform:
    """Build a segmentation transform from config."""
    aug = config.get("augmentation", {})
    return SegmentationTransform(
        train_hflip=bool(aug.get("train_hflip", False)),
        train_vflip=bool(aug.get("train_vflip", False)),
        train_rotate90=bool(aug.get("train_rotate90", False)),
        is_train=is_train,
    )
