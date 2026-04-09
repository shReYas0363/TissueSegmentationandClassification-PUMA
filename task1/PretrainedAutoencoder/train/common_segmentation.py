"""Shared segmentation training implementation used by both train entrypoints."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from task1.PretrainedAutoencoder.datasets.patch_sampling import SegmentationPatchSamplingConfig
from task1.PretrainedAutoencoder.datasets.tissue_dataset import TissueDataset
from task1.PretrainedAutoencoder.datasets.transforms import build_segmentation_transform
from task1.PretrainedAutoencoder.losses.combined import CombinedCrossEntropyDiceLoss
from task1.PretrainedAutoencoder.models.builders import build_segmentation_model
from task1.PretrainedAutoencoder.utils.config import apply_overrides, load_config, prepare_task_runtime, save_config_snapshot
from task1.PretrainedAutoencoder.utils.logging_utils import append_jsonl, append_metrics, prepare_output_dirs, write_metrics_json
from task1.PretrainedAutoencoder.utils.metrics import SegmentationMetricAccumulator
from task1.PretrainedAutoencoder.utils.model_utils import (
    build_optimizer,
    build_scheduler,
    count_total_parameters,
    count_trainable_parameters,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    step_scheduler,
)
from task1.PretrainedAutoencoder.utils.seed import set_seed
from task1.PretrainedAutoencoder.utils.visualization import save_segmentation_preview


def build_common_arg_parser(description: str) -> argparse.ArgumentParser:
    """Build a shared argument parser for segmentation training scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--data-root", default=None, help="Optional dataset root override.")
    parser.add_argument("--output-root", default=None, help="Optional output root override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional DataLoader worker override.")
    parser.add_argument("--device", default=None, help="Optional device override.")
    parser.add_argument("--patch-size", type=int, default=None, help="Optional patch size override.")
    return parser


def _autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def _tensor_to_image_batch(batch_tensor: torch.Tensor) -> np.ndarray:
    images = batch_tensor.detach().cpu().numpy()
    return np.transpose(images, (0, 2, 3, 1))


def _save_preview_batch(
    batch: Dict[str, object],
    logits: torch.Tensor,
    output_path: Path,
    max_items: int,
    alpha: float,
) -> None:
    images = _tensor_to_image_batch(batch["image"])
    masks = batch["mask"].detach().cpu().numpy()
    predictions = torch.argmax(logits.detach().cpu(), dim=1).numpy()

    max_items = min(max_items, len(images))
    for idx in range(max_items):
        save_segmentation_preview(
            image=images[idx],
            target_mask=masks[idx],
            prediction_mask=predictions[idx],
            output_path=output_path.parent / f"{output_path.stem}_{idx}.png",
            alpha=alpha,
        )


def create_segmentation_dataloaders(config: Dict[str, object], device: torch.device) -> Tuple[DataLoader, DataLoader]:
    """Construct train and validation DataLoaders."""
    dataset_cfg = config["dataset"]
    dataloader_cfg = config["dataloader"]
    sampling_cfg = config["sampling"]

    sampling_config = SegmentationPatchSamplingConfig(
        max_retries=int(sampling_cfg.get("max_retries", 5)),
        stroma_min_fraction=float(sampling_cfg.get("stroma_min_fraction", 0.05)),
        other_min_fraction=float(sampling_cfg.get("other_min_fraction", 0.05)),
        tumor_max_fraction=float(sampling_cfg.get("tumor_max_fraction", 0.95)),
        extreme_tumor_accept_prob=float(sampling_cfg.get("extreme_tumor_accept_prob", 0.25)),
    )

    train_dataset = TissueDataset(
        dataset_root=dataset_cfg["root"],
        split=dataset_cfg.get("train_split", "train"),
        mode="segmentation",
        transform=build_segmentation_transform(config, is_train=True),
        mask_root=dataset_cfg["processed_mask_root"],
        patch_size=int(dataset_cfg.get("patch_size", 512)),
        full_image_mode=False,
        samples_per_epoch=int(dataset_cfg.get("train_samples_per_epoch", 1024)),
        segmentation_sampling_config=sampling_config,
    )
    val_dataset = TissueDataset(
        dataset_root=dataset_cfg["root"],
        split=dataset_cfg.get("val_split", "validation"),
        mode="segmentation",
        transform=build_segmentation_transform(config, is_train=False),
        mask_root=dataset_cfg["processed_mask_root"],
        patch_size=int(dataset_cfg.get("patch_size", 512)),
        full_image_mode=False,
        samples_per_epoch=int(dataset_cfg.get("val_samples_per_epoch", 256)),
        segmentation_sampling_config=sampling_config,
    )

    num_workers = int(dataloader_cfg.get("num_workers", 0))
    pin_memory = bool(dataloader_cfg.get("pin_memory", False)) and device.type == "cuda"
    persistent_workers = bool(dataloader_cfg.get("persistent_workers", False)) and num_workers > 0
    batch_size = int(dataloader_cfg.get("batch_size", 4))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader


def run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CombinedCrossEntropyDiceLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
    grad_clip_norm: float | None,
) -> Dict[str, float]:
    """Run one segmentation training epoch."""
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_dice = 0.0

    progress = tqdm(loader, desc="Train", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_enabled):
            logits = model(images)
            loss, components = criterion(logits, masks)

        scaler.scale(loss).backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.detach().item())
        running_ce += float(components["ce_loss"].item())
        running_dice += float(components["dice_loss"].item())
        progress.set_postfix(loss=f"{running_loss / max(len(progress), 1):.4f}")

    num_batches = max(len(loader), 1)
    return {
        "train_loss": running_loss / num_batches,
        "train_ce_loss": running_ce / num_batches,
        "train_dice_loss": running_dice / num_batches,
    }


def run_validation_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CombinedCrossEntropyDiceLoss,
    device: torch.device,
    amp_enabled: bool,
    class_names,
) -> Tuple[Dict[str, float], Dict[str, object], Dict[str, object], torch.Tensor]:
    """Run one validation epoch and return metrics plus the last logits batch."""
    model.eval()
    running_loss = 0.0
    running_ce = 0.0
    running_dice = 0.0
    accumulator = SegmentationMetricAccumulator(num_classes=len(class_names), class_names=class_names)

    preview_batch = None
    preview_logits = None

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with _autocast_context(device, amp_enabled):
                logits = model(images)
                loss, components = criterion(logits, masks)

            running_loss += float(loss.detach().item())
            running_ce += float(components["ce_loss"].item())
            running_dice += float(components["dice_loss"].item())
            accumulator.update_from_logits(logits, masks)

            if preview_batch is None:
                preview_batch = {
                    "image": batch["image"].clone(),
                    "mask": batch["mask"].clone(),
                }
                preview_logits = logits.detach().cpu()

    num_batches = max(len(loader), 1)
    mean_losses = {
        "val_loss": running_loss / num_batches,
        "val_ce_loss": running_ce / num_batches,
        "val_dice_loss": running_dice / num_batches,
    }
    metric_payload = accumulator.compute()
    return mean_losses, metric_payload, preview_batch, preview_logits


def train_segmentation(config: Dict[str, object]) -> None:
    """Shared implementation for both segmentation training scripts."""
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device"))
    amp_enabled = bool(config.get("mixed_precision", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    output_dirs = prepare_output_dirs(config["output"]["root"])
    save_config_snapshot(config, output_dirs["logs"] / "resolved_config.yaml")

    train_loader, val_loader = create_segmentation_dataloaders(config, device)
    class_names = tuple(config["dataset"].get("class_names", ["other", "tumor", "stroma"]))

    model = build_segmentation_model(config).to(device)
    criterion = CombinedCrossEntropyDiceLoss(
        num_classes=int(config["dataset"].get("num_classes", 3)),
        ce_weight=float(config["loss"].get("ce_weight", 1.0)),
        dice_weight=float(config["loss"].get("dice_weight", 1.0)),
        class_weights=config["loss"].get("class_weights"),
        smooth=float(config["loss"].get("smooth", 1.0)),
    )
    optimizer = build_optimizer(model, config["optimizer"])
    scheduler = build_scheduler(optimizer, config.get("scheduler", {}))

    model_info = {
        "trainable_parameters": count_trainable_parameters(model),
        "total_parameters": count_total_parameters(model),
        "device": str(device),
        "mixed_precision": amp_enabled,
    }
    write_metrics_json(output_dirs["metrics"] / "model_info.json", model_info)
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"Total parameters: {model_info['total_parameters']:,}")

    num_epochs = int(config["train"].get("epochs", 1))
    grad_clip_norm = config["train"].get("grad_clip_norm")
    early_stopping_patience = int(config["train"].get("early_stopping_patience", 0))
    best_mean_dice = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0

    metrics_csv_path = output_dirs["metrics"] / "training_metrics.csv"
    metrics_jsonl_path = output_dirs["logs"] / "training_metrics.jsonl"
    alpha = float(config.get("preprocessing", {}).get("overlay_alpha", 0.45))
    preview_count = int(config["train"].get("sample_visualizations", 4))

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            grad_clip_norm=grad_clip_norm,
        )
        val_losses, val_seg_metrics, preview_batch, preview_logits = run_validation_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            class_names=class_names,
        )

        epoch_metrics = {"epoch": epoch, **train_metrics, **val_losses}
        epoch_metrics["val_mean_dice"] = val_seg_metrics["mean_dice"]
        epoch_metrics["val_mean_iou"] = val_seg_metrics["mean_iou"]
        epoch_metrics["val_mean_precision"] = val_seg_metrics["mean_precision"]
        epoch_metrics["val_mean_recall"] = val_seg_metrics["mean_recall"]
        epoch_metrics["val_mean_f1"] = val_seg_metrics["mean_f1"]
        epoch_metrics["val_pixel_accuracy"] = val_seg_metrics["pixel_accuracy"]
        for class_name in class_names:
            epoch_metrics[f"val_dice_{class_name}"] = val_seg_metrics.get(f"dice_{class_name}", 0.0)
            epoch_metrics[f"val_iou_{class_name}"] = val_seg_metrics.get(f"iou_{class_name}", 0.0)
            epoch_metrics[f"val_precision_{class_name}"] = val_seg_metrics.get(f"precision_{class_name}", 0.0)
            epoch_metrics[f"val_recall_{class_name}"] = val_seg_metrics.get(f"recall_{class_name}", 0.0)
            epoch_metrics[f"val_f1_{class_name}"] = val_seg_metrics.get(f"f1_{class_name}", 0.0)

        append_metrics(metrics_csv_path, epoch_metrics)
        append_jsonl(metrics_jsonl_path, epoch_metrics)

        scheduler_monitor = val_seg_metrics["mean_dice"] if str(config.get("scheduler", {}).get("mode", "max")).lower() == "max" else val_losses["val_loss"]
        step_scheduler(scheduler, monitor_value=scheduler_monitor if scheduler is not None else None)

        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "metrics": epoch_metrics,
            "config": config,
        }
        save_checkpoint(output_dirs["checkpoints"] / "last.pt", checkpoint_payload)

        if preview_batch is not None and preview_logits is not None:
            _save_preview_batch(
                preview_batch,
                preview_logits,
                output_dirs["visualizations"] / f"epoch_{epoch:03d}.png",
                max_items=preview_count,
                alpha=alpha,
            )

        if val_seg_metrics["mean_dice"] > best_mean_dice:
            best_mean_dice = float(val_seg_metrics["mean_dice"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_dirs["checkpoints"] / "best.pt", checkpoint_payload)
            write_metrics_json(output_dirs["metrics"] / "best_val_metrics.json", val_seg_metrics)
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"val_loss={val_losses['val_loss']:.4f} "
            f"val_mean_dice={val_seg_metrics['mean_dice']:.4f} "
            f"val_mean_iou={val_seg_metrics['mean_iou']:.4f} "
            f"val_mean_f1={val_seg_metrics['mean_f1']:.4f}"
        )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_mean_dice": best_mean_dice,
        "trainable_parameters": model_info["trainable_parameters"],
        "total_parameters": model_info["total_parameters"],
    }
    write_metrics_json(output_dirs["metrics"] / "training_summary.json", summary)

    best_checkpoint = load_checkpoint(output_dirs["checkpoints"] / "best.pt", map_location="cpu")
    best_model_state = best_checkpoint["model_state"]
    best_encoder_state = {
        key.replace("encoder.", "", 1): value
        for key, value in best_model_state.items()
        if key.startswith("encoder.")
    }
    save_checkpoint(output_dirs["checkpoints"] / "best_encoder_state.pt", {"encoder_state": best_encoder_state})
    print(f"Best validation mean Dice: {best_mean_dice:.4f} at epoch {best_epoch}")


def load_config_from_args(args: argparse.Namespace) -> Dict[str, object]:
    """Load a segmentation config and apply common CLI overrides."""
    config_path = prepare_task_runtime(args.config)
    config = load_config(config_path)
    return apply_overrides(
        config,
        {
            "dataset.root": args.data_root,
            "output.root": args.output_root,
            "dataloader.batch_size": args.batch_size,
            "dataloader.num_workers": args.num_workers,
            "device": args.device,
            "dataset.patch_size": args.patch_size,
            "evaluation.sliding_window.patch_size": args.patch_size,
        },
    )
