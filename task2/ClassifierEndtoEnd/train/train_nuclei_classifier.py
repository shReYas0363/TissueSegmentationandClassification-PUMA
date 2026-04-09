"""Train the Task 2(a) ConvNeXt-Tiny nuclei classifier."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from typing import Dict, Tuple

import torch
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from task2.ClassifierEndtoEnd.datasets.nuclei_classification_dataset import NucleiClassificationDataset
from task2.ClassifierEndtoEnd.datasets.nuclei_classification_transforms import build_nuclei_classification_transform
from task2.ClassifierEndtoEnd.models.convnext_tiny_classifier import ConvNeXtTinyClassifier
from task2.ClassifierEndtoEnd.utils.classification_metrics import ClassificationMetricAccumulator
from task2.ClassifierEndtoEnd.utils.config import apply_overrides, load_config, prepare_task_runtime, save_config_snapshot
from task2.ClassifierEndtoEnd.utils.logging_utils import append_jsonl, append_metrics, prepare_output_dirs, write_metrics_json
from task2.ClassifierEndtoEnd.utils.model_utils import (
    build_optimizer,
    build_scheduler,
    count_total_parameters,
    count_trainable_parameters,
    resolve_device,
    save_checkpoint,
    step_scheduler,
)
from task2.ClassifierEndtoEnd.utils.nuclei_classification import get_class_names, get_label_mapping
from task2.ClassifierEndtoEnd.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """CLI arguments for Task 2(a) training."""
    parser = argparse.ArgumentParser(description="Train the Task 2(a) ConvNeXt-Tiny nuclei classifier.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--generated-root", default=None, help="Optional generated nuclei dataset root override.")
    parser.add_argument("--output-root", default=None, help="Optional output root override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional DataLoader worker override.")
    parser.add_argument("--device", default=None, help="Optional device override.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional smoke-test train batch limit.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional smoke-test validation batch limit.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional smoke-test train sample limit.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional smoke-test validation sample limit.")
    return parser.parse_args()


def _autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def _limit_from_config(debug_cfg: Dict[str, object], key: str) -> int | None:
    value = debug_cfg.get(key)
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def _resolve_monitor_metric(train_cfg: Dict[str, object]) -> str:
    monitor_metric = str(train_cfg.get("monitor_metric", "val_acc")).strip().lower()
    if monitor_metric not in {"val_acc", "val_macro_f1"}:
        raise ValueError(
            "Unsupported train.monitor_metric: "
            f"{monitor_metric}. Expected one of: val_acc, val_macro_f1."
        )
    return monitor_metric


def _get_monitor_value(monitor_metric: str, val_metrics: Dict[str, object]) -> float:
    if monitor_metric == "val_acc":
        return float(val_metrics["accuracy"])
    if monitor_metric == "val_macro_f1":
        return float(val_metrics["macro_f1"])
    raise ValueError(f"Unsupported monitor metric: {monitor_metric}")


def _make_dataloaders(config: Dict[str, object], device: torch.device) -> Tuple[DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    dataloader_cfg = config["dataloader"]
    debug_cfg = config.get("debug", {})
    class_names = get_class_names(dataset_cfg.get("class_names"))
    label_mapping = get_label_mapping(dataset_cfg.get("label_mapping"))

    train_dataset = NucleiClassificationDataset(
        mode="generated_split",
        split=str(dataset_cfg.get("train_split", "train")),
        generated_root=dataset_cfg["generated_root"],
        transform=build_nuclei_classification_transform(config, is_train=True),
        class_names=class_names,
        label_mapping=label_mapping,
        max_samples=_limit_from_config(debug_cfg, "max_train_samples"),
    )
    val_dataset = NucleiClassificationDataset(
        mode="generated_split",
        split=str(dataset_cfg.get("val_split", "val")),
        generated_root=dataset_cfg["generated_root"],
        transform=build_nuclei_classification_transform(config, is_train=False),
        class_names=class_names,
        label_mapping=label_mapping,
        max_samples=_limit_from_config(debug_cfg, "max_val_samples"),
    )

    num_workers = int(dataloader_cfg.get("num_workers", 0))
    pin_memory = bool(dataloader_cfg.get("pin_memory", False)) and device.type == "cuda"
    persistent_workers = bool(dataloader_cfg.get("persistent_workers", False)) and num_workers > 0
    batch_size = int(dataloader_cfg.get("batch_size", 32))

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


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool,
    class_names,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    grad_clip_norm: float | None = None,
    max_batches: int | None = None,
) -> Tuple[float, Dict[str, object]]:
    """Run one train or validation epoch."""
    is_train = optimizer is not None
    model.train(is_train)
    accumulator = ClassificationMetricAccumulator(
        num_classes=len(class_names),
        class_names=class_names,
    )
    running_loss = 0.0
    processed_batches = 0

    context = torch.enable_grad if is_train else torch.no_grad
    with context():
        progress = tqdm(loader, desc="Train" if is_train else "Validation", leave=False)
        for batch_idx, batch in enumerate(progress, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with _autocast_context(device, amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_train:
                scaler.scale(loss).backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

            running_loss += float(loss.detach().item())
            processed_batches += 1
            accumulator.update_from_logits(logits, targets)

            metrics = accumulator.compute()
            progress.set_postfix(
                loss=f"{running_loss / processed_batches:.4f}",
                acc=f"{metrics['accuracy']:.4f}",
            )

    if processed_batches == 0:
        raise ValueError("No batches were processed. Check dataset paths or debug limits.")

    return running_loss / processed_batches, accumulator.compute()


def main() -> None:
    """Train the Task 2(a) ConvNeXt-Tiny classifier."""
    args = parse_args()
    config_path = prepare_task_runtime(args.config)
    config = apply_overrides(
        load_config(config_path),
        {
            "dataset.generated_root": args.generated_root,
            "output.root": args.output_root,
            "dataloader.batch_size": args.batch_size,
            "dataloader.num_workers": args.num_workers,
            "device": args.device,
            "debug.max_train_batches_per_epoch": args.max_train_batches,
            "debug.max_val_batches_per_epoch": args.max_val_batches,
            "debug.max_train_samples": args.max_train_samples,
            "debug.max_val_samples": args.max_val_samples,
        },
    )

    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device"))
    amp_enabled = bool(config.get("mixed_precision", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    class_names = get_class_names(config["dataset"].get("class_names"))
    output_dirs = prepare_output_dirs(config["output"]["root"])
    save_config_snapshot(config, output_dirs["logs"] / "resolved_config.yaml")

    train_loader, val_loader = _make_dataloaders(config, device)
    model = ConvNeXtTinyClassifier(
        num_classes=len(class_names),
        weights=str(config["model"].get("weights", "default")),
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config["optimizer"])
    scheduler = build_scheduler(optimizer, config.get("scheduler", {}))

    model_info = {
        "trainable_parameters": count_trainable_parameters(model),
        "total_parameters": count_total_parameters(model),
        "device": str(device),
        "mixed_precision": amp_enabled,
        "class_names": list(class_names),
    }
    write_metrics_json(output_dirs["metrics"] / "model_info.json", model_info)
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"Total parameters: {model_info['total_parameters']:,}")

    num_epochs = int(config["train"].get("epochs", 1))
    grad_clip_norm = config["train"].get("grad_clip_norm")
    early_stopping_patience = int(config["train"].get("early_stopping_patience", 0))
    monitor_metric = _resolve_monitor_metric(config["train"])
    debug_cfg = config.get("debug", {})
    max_train_batches = _limit_from_config(debug_cfg, "max_train_batches_per_epoch")
    max_val_batches = _limit_from_config(debug_cfg, "max_val_batches_per_epoch")
    best_monitor_value = float("-inf")
    best_epoch = -1
    best_val_accuracy = float("-inf")
    best_val_macro_f1 = float("-inf")
    epochs_without_improvement = 0

    metrics_csv_path = output_dirs["metrics"] / "training_metrics.csv"
    metrics_jsonl_path = output_dirs["logs"] / "training_metrics.jsonl"

    for epoch in range(1, num_epochs + 1):
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            class_names=class_names,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
            max_batches=max_train_batches,
        )
        val_loss, val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            class_names=class_names,
            optimizer=None,
            scaler=None,
            grad_clip_norm=None,
            max_batches=max_val_batches,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_precision": train_metrics["macro_precision"],
            "train_macro_recall": train_metrics["macro_recall"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        for class_name in class_names:
            epoch_metrics[f"val_precision_{class_name}"] = val_metrics.get(f"precision_{class_name}", 0.0)
            epoch_metrics[f"val_recall_{class_name}"] = val_metrics.get(f"recall_{class_name}", 0.0)
            epoch_metrics[f"val_f1_{class_name}"] = val_metrics.get(f"f1_{class_name}", 0.0)

        append_metrics(metrics_csv_path, epoch_metrics)
        append_jsonl(metrics_jsonl_path, epoch_metrics)

        monitor_value = _get_monitor_value(monitor_metric, val_metrics)
        step_scheduler(scheduler, monitor_value=monitor_value if scheduler is not None else None)

        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "metrics": epoch_metrics,
            "config": config,
        }
        save_checkpoint(output_dirs["checkpoints"] / "last.pt", checkpoint_payload)

        if monitor_value > best_monitor_value:
            best_monitor_value = monitor_value
            best_epoch = epoch
            best_val_accuracy = float(val_metrics["accuracy"])
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            epochs_without_improvement = 0
            save_checkpoint(output_dirs["checkpoints"] / "best.pt", checkpoint_payload)
            best_val_metrics = dict(val_metrics)
            best_val_metrics["monitor_metric"] = monitor_metric
            best_val_metrics["monitor_value"] = monitor_value
            write_metrics_json(output_dirs["metrics"] / "best_val_metrics.json", best_val_metrics)
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    write_metrics_json(
        output_dirs["metrics"] / "training_summary.json",
        {
            "monitor_metric": monitor_metric,
            "best_monitor_value": best_monitor_value,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "best_val_macro_f1": best_val_macro_f1,
            "trainable_parameters": model_info["trainable_parameters"],
            "total_parameters": model_info["total_parameters"],
        },
    )
    print(
        f"Best checkpoint selected by {monitor_metric}: "
        f"{best_monitor_value:.4f} at epoch {best_epoch} "
        f"(val_acc={best_val_accuracy:.4f}, val_macro_f1={best_val_macro_f1:.4f})"
    )


if __name__ == "__main__":
    main()
