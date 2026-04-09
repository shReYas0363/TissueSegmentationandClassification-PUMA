"""Evaluate the Task 2(a) ConvNeXt-Tiny nuclei classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from task2.ClassifierEndtoEnd.datasets.nuclei_classification_dataset import NucleiClassificationDataset
from task2.ClassifierEndtoEnd.datasets.nuclei_classification_transforms import build_nuclei_classification_transform
from task2.ClassifierEndtoEnd.models.convnext_tiny_classifier import ConvNeXtTinyClassifier
from task2.ClassifierEndtoEnd.utils.classification_metrics import (
    ClassificationMetricAccumulator,
    confusion_matrix_rows,
    logits_to_predictions,
    logits_to_probabilities,
    per_class_rows,
    save_confusion_matrix_figure,
)
from task2.ClassifierEndtoEnd.utils.config import apply_overrides, load_config, prepare_task_runtime, save_config_snapshot
from task2.ClassifierEndtoEnd.utils.io import ensure_dir, write_csv_rows
from task2.ClassifierEndtoEnd.utils.logging_utils import prepare_output_dirs, write_metrics_json
from task2.ClassifierEndtoEnd.utils.model_utils import count_total_parameters, count_trainable_parameters, load_checkpoint, resolve_device
from task2.ClassifierEndtoEnd.utils.nuclei_classification import get_class_names, get_label_mapping
from task2.ClassifierEndtoEnd.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """CLI arguments for Task 2(a) evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the Task 2(a) ConvNeXt-Tiny nuclei classifier.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--split", default="task2_test", choices=("val", "task2_test"), help="Which split to evaluate.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override.")
    parser.add_argument("--generated-root", default=None, help="Optional generated nuclei dataset root override.")
    parser.add_argument("--task2-test-root", default=None, help="Optional Task2 official test root override.")
    parser.add_argument("--output-root", default=None, help="Optional output root override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional DataLoader worker override.")
    parser.add_argument("--device", default=None, help="Optional device override.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional smoke-test val sample limit.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional smoke-test test sample limit.")
    return parser.parse_args()


def _limit_from_config(debug_cfg: Dict[str, object], key: str) -> int | None:
    value = debug_cfg.get(key)
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def _build_eval_loader(config: Dict[str, object], split: str, device: torch.device) -> DataLoader:
    dataset_cfg = config["dataset"]
    dataloader_cfg = config["dataloader"]
    debug_cfg = config.get("debug", {})
    class_names = get_class_names(dataset_cfg.get("class_names"))
    label_mapping = get_label_mapping(dataset_cfg.get("label_mapping"))

    if split == "val":
        dataset = NucleiClassificationDataset(
            mode="generated_split",
            split=str(dataset_cfg.get("val_split", "val")),
            generated_root=dataset_cfg["generated_root"],
            transform=build_nuclei_classification_transform(config, is_train=False),
            class_names=class_names,
            label_mapping=label_mapping,
            max_samples=_limit_from_config(debug_cfg, "max_val_samples"),
        )
    else:
        dataset = NucleiClassificationDataset(
            mode="official_task2_test",
            split="task2_test",
            task2_test_root=dataset_cfg["task2_test_root"],
            transform=build_nuclei_classification_transform(config, is_train=False),
            class_names=class_names,
            label_mapping=label_mapping,
            max_samples=_limit_from_config(debug_cfg, "max_test_samples"),
        )

    num_workers = int(dataloader_cfg.get("num_workers", 0))
    pin_memory = bool(dataloader_cfg.get("pin_memory", False)) and device.type == "cuda"
    persistent_workers = bool(dataloader_cfg.get("persistent_workers", False)) and num_workers > 0
    batch_size = int(dataloader_cfg.get("batch_size", 32))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def main() -> None:
    """Evaluate one Task 2(a) experiment checkpoint."""
    args = parse_args()
    config_path = prepare_task_runtime(args.config)
    config = apply_overrides(
        load_config(config_path),
        {
            "dataset.generated_root": args.generated_root,
            "dataset.task2_test_root": args.task2_test_root,
            "output.root": args.output_root,
            "dataloader.batch_size": args.batch_size,
            "dataloader.num_workers": args.num_workers,
            "device": args.device,
            "debug.max_val_samples": args.max_val_samples,
            "debug.max_test_samples": args.max_test_samples,
        },
    )

    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device"))
    class_names = get_class_names(config["dataset"].get("class_names"))

    output_dirs = prepare_output_dirs(config["output"]["root"])
    eval_root = ensure_dir(Path(output_dirs["root"]) / "evaluation" / args.split)
    save_config_snapshot(config, eval_root / "resolved_config.yaml")

    model = ConvNeXtTinyClassifier(num_classes=len(class_names), weights="none").to(device)
    default_checkpoint = Path(config["evaluation"].get("default_checkpoint", "checkpoints/best.pt"))
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        default_checkpoint if default_checkpoint.is_absolute() else Path(output_dirs["root"]) / default_checkpoint
    )
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    model_info = {
        "checkpoint_path": str(checkpoint_path),
        "trainable_parameters": count_trainable_parameters(model),
        "total_parameters": count_total_parameters(model),
        "device": str(device),
        "class_names": list(class_names),
    }
    write_metrics_json(eval_root / "model_info.json", model_info)

    loader = _build_eval_loader(config, args.split, device)
    accumulator = ClassificationMetricAccumulator(num_classes=len(class_names), class_names=class_names)
    prediction_rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)

            logits = model(images)
            probabilities = logits_to_probabilities(logits)
            predictions = logits_to_predictions(logits)
            accumulator.update(predictions, targets)

            probs_np = probabilities.detach().cpu().numpy()
            preds_np = predictions.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            batch_size = len(preds_np)

            for idx in range(batch_size):
                row = {
                    "sample_id": batch["sample_id"][idx],
                    "path": batch["path"][idx],
                    "split": batch["split"][idx],
                    "source_stem": batch["source_stem"][idx],
                    "subgroup": batch["subgroup"][idx],
                    "true_label": int(targets_np[idx]),
                    "true_class_name": batch["class_name"][idx],
                    "pred_label": int(preds_np[idx]),
                    "pred_class_name": class_names[int(preds_np[idx])],
                }
                for class_idx, class_name in enumerate(class_names):
                    row[f"prob_{class_name}"] = float(probs_np[idx, class_idx])
                prediction_rows.append(row)

    metrics = accumulator.compute()
    metrics["split"] = args.split
    metrics["model_name"] = config["model"]["name"]
    metrics["model_info"] = model_info
    write_metrics_json(eval_root / "metrics.json", metrics)
    write_csv_rows(per_class_rows(metrics, class_names), eval_root / "per_class_metrics.csv")
    write_csv_rows(prediction_rows, eval_root / "predictions.csv")

    confusion_matrix = accumulator.confusion_matrix
    write_csv_rows(
        confusion_matrix_rows(confusion_matrix, class_names),
        eval_root / "confusion_matrix.csv",
    )
    save_confusion_matrix_figure(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
        output_path=eval_root / "confusion_matrix.png",
    )

    print(f"Evaluation complete for split={args.split}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
