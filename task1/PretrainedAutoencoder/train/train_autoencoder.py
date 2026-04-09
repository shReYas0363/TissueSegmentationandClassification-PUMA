"""Train an autoencoder on raw tissue image patches."""

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

from task1.PretrainedAutoencoder.datasets.tissue_dataset import TissueDataset
from task1.PretrainedAutoencoder.datasets.transforms import build_reconstruction_transform
from task1.PretrainedAutoencoder.models.builders import build_autoencoder
from task1.PretrainedAutoencoder.utils.config import apply_overrides, load_config, prepare_task_runtime, save_config_snapshot
from task1.PretrainedAutoencoder.utils.logging_utils import append_jsonl, append_metrics, prepare_output_dirs, write_metrics_json
from task1.PretrainedAutoencoder.utils.model_utils import (
    build_optimizer,
    build_scheduler,
    count_total_parameters,
    count_trainable_parameters,
    resolve_device,
    save_checkpoint,
    step_scheduler,
)
from task1.PretrainedAutoencoder.utils.seed import set_seed
from task1.PretrainedAutoencoder.utils.visualization import save_reconstruction_preview


def parse_args() -> argparse.Namespace:
    """CLI arguments for autoencoder training."""
    parser = argparse.ArgumentParser(description="Train the convolutional autoencoder.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--data-root", default=None, help="Optional dataset root override.")
    parser.add_argument("--output-root", default=None, help="Optional output root override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional DataLoader worker override.")
    parser.add_argument("--device", default=None, help="Optional device override.")
    parser.add_argument("--patch-size", type=int, default=None, help="Optional patch size override.")
    return parser.parse_args()


def _autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def _make_dataloaders(config: Dict[str, object], device: torch.device) -> Tuple[DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    dataloader_cfg = config["dataloader"]

    train_dataset = TissueDataset(
        dataset_root=dataset_cfg["root"],
        split=dataset_cfg.get("train_split", "train"),
        mode="reconstruction",
        transform=build_reconstruction_transform(config, is_train=True),
        mask_root=None,
        patch_size=int(dataset_cfg.get("patch_size", 512)),
        full_image_mode=False,
        samples_per_epoch=int(dataset_cfg.get("train_samples_per_epoch", 1024)),
    )
    val_dataset = TissueDataset(
        dataset_root=dataset_cfg["root"],
        split=dataset_cfg.get("val_split", "validation"),
        mode="reconstruction",
        transform=build_reconstruction_transform(config, is_train=False),
        mask_root=None,
        patch_size=int(dataset_cfg.get("patch_size", 512)),
        full_image_mode=False,
        samples_per_epoch=int(dataset_cfg.get("val_samples_per_epoch", 256)),
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


def _get_reconstruction_loss(loss_name: str):
    loss_name = loss_name.lower()
    if loss_name == "l1":
        return torch.nn.L1Loss()
    if loss_name == "mse":
        return torch.nn.MSELoss()
    raise ValueError(f"Unsupported reconstruction loss: {loss_name}")


def _tensor_batch_to_images(batch_tensor: torch.Tensor) -> np.ndarray:
    images = batch_tensor.detach().cpu().numpy()
    return np.transpose(images, (0, 2, 3, 1))


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    amp_enabled: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    grad_clip_norm: float | None = None,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """Run one training or validation epoch."""
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    preview = {}

    context = torch.enable_grad if is_train else torch.no_grad
    with context():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Train" if is_train else "Validation", leave=False)):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with _autocast_context(device, amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, targets)

            if is_train:
                scaler.scale(loss).backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

            running_loss += float(loss.detach().item())

            if batch_idx == 0:
                preview = {
                    "inputs": batch["image"].clone(),
                    "targets": batch["target"].clone(),
                    "outputs": outputs.detach().cpu(),
                }

    num_batches = max(len(loader), 1)
    return running_loss / num_batches, preview


def main() -> None:
    """Train the autoencoder."""
    args = parse_args()
    config_path = prepare_task_runtime(args.config)
    config = apply_overrides(
        load_config(config_path),
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

    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device"))
    amp_enabled = bool(config.get("mixed_precision", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    output_dirs = prepare_output_dirs(config["output"]["root"])
    save_config_snapshot(config, output_dirs["logs"] / "resolved_config.yaml")

    train_loader, val_loader = _make_dataloaders(config, device)
    model = build_autoencoder(config).to(device)
    criterion = _get_reconstruction_loss(str(config["loss"].get("name", "l1")))
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
    preview_count = int(config["train"].get("sample_visualizations", 4))
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    metrics_csv_path = output_dirs["metrics"] / "training_metrics.csv"
    metrics_jsonl_path = output_dirs["logs"] / "training_metrics.jsonl"

    for epoch in range(1, num_epochs + 1):
        train_loss, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
        )
        val_loss, preview = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            optimizer=None,
            scaler=None,
        )

        if preview:
            save_reconstruction_preview(
                inputs=_tensor_batch_to_images(preview["inputs"]),
                targets=_tensor_batch_to_images(preview["targets"]),
                reconstructions=_tensor_batch_to_images(preview["outputs"]),
                output_path=output_dirs["visualizations"] / f"epoch_{epoch:03d}.png",
                max_items=preview_count,
            )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        append_metrics(metrics_csv_path, epoch_metrics)
        append_jsonl(metrics_jsonl_path, epoch_metrics)

        scheduler_monitor = val_loss if scheduler is not None else None
        step_scheduler(scheduler, monitor_value=scheduler_monitor)

        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "encoder_state": model.encoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "metrics": epoch_metrics,
            "config": config,
        }
        save_checkpoint(output_dirs["checkpoints"] / "last.pt", checkpoint_payload)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_dirs["checkpoints"] / "best.pt", checkpoint_payload)
            save_checkpoint(output_dirs["checkpoints"] / "best_encoder.pt", {"encoder_state": model.encoder.state_dict()})
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    save_checkpoint(output_dirs["checkpoints"] / "final_encoder.pt", {"encoder_state": model.encoder.state_dict()})
    write_metrics_json(
        output_dirs["metrics"] / "training_summary.json",
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "trainable_parameters": model_info["trainable_parameters"],
            "total_parameters": model_info["total_parameters"],
        },
    )
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
