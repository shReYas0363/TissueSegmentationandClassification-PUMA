"""Reusable tissue dataset for reconstruction and segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from task1.PretrainedAutoencoder.datasets.patch_sampling import (
    SegmentationPatchSamplingConfig,
    sample_reconstruction_patch,
    sample_segmentation_patch,
)
from task1.PretrainedAutoencoder.utils.io import discover_split_records, read_mask_image, read_tiff_image


class TissueDataset(Dataset):
    """Dataset that supports both reconstruction and segmentation modes."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        mode: str,
        transform=None,
        mask_root: str | Path | None = None,
        patch_size: int = 512,
        full_image_mode: bool = False,
        samples_per_epoch: Optional[int] = None,
        segmentation_sampling_config: Optional[SegmentationPatchSamplingConfig] = None,
    ) -> None:
        if mode not in {"reconstruction", "segmentation"}:
            raise ValueError(f"Unsupported dataset mode: {mode}")

        self.mode = mode
        self.transform = transform
        self.patch_size = int(patch_size)
        self.full_image_mode = bool(full_image_mode)
        self.segmentation_sampling_config = segmentation_sampling_config

        require_mask = mode == "segmentation"
        self.records = discover_split_records(
            dataset_root=dataset_root,
            split=split,
            mask_root=mask_root,
            require_tissue_geojson=True,
            require_mask=require_mask,
        )
        self.samples_per_epoch = samples_per_epoch or len(self.records)

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.records) if self.full_image_mode else int(self.samples_per_epoch)

    def _get_record(self, index: int):
        """Select a source image record."""
        if self.full_image_mode:
            return self.records[index]
        return self.records[index % len(self.records)]

    def __getitem__(self, index: int) -> Dict[str, object]:
        """Read either a full image or one random patch."""
        record = self._get_record(index)
        image = read_tiff_image(record.image_path)

        if self.mode == "reconstruction":
            if self.full_image_mode:
                image_tensor, target_tensor = self.transform(image) if self.transform else (image, image)
                return {
                    "image": image_tensor,
                    "target": target_tensor,
                    "stem": record.stem,
                    "split": record.split,
                    "subgroup": record.subgroup,
                }

            sampled = sample_reconstruction_patch(image, patch_size=self.patch_size)
            image_tensor, target_tensor = self.transform(sampled["image_patch"]) if self.transform else (
                sampled["image_patch"],
                sampled["image_patch"],
            )
            return {
                "image": image_tensor,
                "target": target_tensor,
                "stem": record.stem,
                "split": record.split,
                "subgroup": record.subgroup,
                "top": sampled["top"],
                "left": sampled["left"],
            }

        mask = read_mask_image(record.mask_path)
        if self.full_image_mode:
            image_tensor, mask_tensor = self.transform(image, mask) if self.transform else (image, mask)
            return {
                "image": image_tensor,
                "mask": mask_tensor,
                "stem": record.stem,
                "split": record.split,
                "subgroup": record.subgroup,
            }

        if self.segmentation_sampling_config is None:
            raise ValueError("Segmentation patch mode requires a sampling config.")

        sampled = sample_segmentation_patch(
            image=image,
            mask=mask,
            patch_size=self.patch_size,
            config=self.segmentation_sampling_config,
        )
        image_tensor, mask_tensor = self.transform(sampled["image_patch"], sampled["mask_patch"]) if self.transform else (
            sampled["image_patch"],
            sampled["mask_patch"],
        )
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "stem": record.stem,
            "split": record.split,
            "subgroup": record.subgroup,
            "top": sampled["top"],
            "left": sampled["left"],
            "tumor_frac": sampled["tumor_frac"],
            "stroma_frac": sampled["stroma_frac"],
            "other_frac": sampled["other_frac"],
        }
