"""Sliding-window inference for full-resolution tissue segmentation."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def compute_window_starts(full_size: int, window_size: int, stride: int) -> List[int]:
    """Compute starting indices that fully cover a dimension."""
    if full_size <= window_size:
        return [0]

    starts = list(range(0, full_size - window_size + 1, stride))
    if starts[-1] != full_size - window_size:
        starts.append(full_size - window_size)
    return starts


def sliding_window_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    patch_size: int = 512,
    overlap: float = 0.25,
    batch_size: int = 2,
) -> torch.Tensor:
    """Run sliding-window inference and average logits across overlaps."""
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError(f"Expected image tensor of shape [1, C, H, W], got {tuple(image_tensor.shape)}")

    _, _, original_height, original_width = image_tensor.shape
    patch_size = int(patch_size)
    stride = max(int(round(patch_size * (1.0 - overlap))), 1)

    pad_height = max(0, patch_size - original_height)
    pad_width = max(0, patch_size - original_width)
    if pad_height > 0 or pad_width > 0:
        image_tensor = F.pad(image_tensor, (0, pad_width, 0, pad_height), mode="reflect")

    _, _, padded_height, padded_width = image_tensor.shape
    row_starts = compute_window_starts(padded_height, patch_size, stride)
    col_starts = compute_window_starts(padded_width, patch_size, stride)

    patch_coordinates: List[Tuple[int, int]] = [(top, left) for top in row_starts for left in col_starts]
    logits_sum = None
    count_map = None

    model.eval()
    with torch.no_grad():
        for start_idx in range(0, len(patch_coordinates), batch_size):
            batch_coords = patch_coordinates[start_idx : start_idx + batch_size]
            patches = []
            for top, left in batch_coords:
                patch = image_tensor[:, :, top : top + patch_size, left : left + patch_size]
                patches.append(patch)
            batch_tensor = torch.cat(patches, dim=0)
            batch_logits = model(batch_tensor)

            if logits_sum is None:
                num_classes = batch_logits.shape[1]
                logits_sum = torch.zeros(
                    (1, num_classes, padded_height, padded_width),
                    dtype=batch_logits.dtype,
                    device=batch_logits.device,
                )
                count_map = torch.zeros(
                    (1, 1, padded_height, padded_width),
                    dtype=batch_logits.dtype,
                    device=batch_logits.device,
                )

            for patch_idx, (top, left) in enumerate(batch_coords):
                logits_sum[:, :, top : top + patch_size, left : left + patch_size] += batch_logits[patch_idx : patch_idx + 1]
                count_map[:, :, top : top + patch_size, left : left + patch_size] += 1.0

    averaged_logits = logits_sum / count_map.clamp_min(1.0)
    return averaged_logits[:, :, :original_height, :original_width]
