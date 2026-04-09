"""Evaluate a trained segmentation model on full images using sliding-window inference."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from task1.PretrainedAutoencoder.datasets.transforms import build_segmentation_transform
from task1.PretrainedAutoencoder.evaluate.inference_sliding_window import sliding_window_inference
from task1.PretrainedAutoencoder.models.builders import build_segmentation_model
from task1.PretrainedAutoencoder.utils.config import apply_overrides, load_config, prepare_task_runtime, save_config_snapshot
from task1.PretrainedAutoencoder.utils.io import discover_split_records, ensure_dir, read_mask_image, read_tiff_image, upsert_csv_row, write_csv_rows
from task1.PretrainedAutoencoder.utils.logging_utils import prepare_output_dirs, write_metrics_json
from task1.PretrainedAutoencoder.utils.metrics import SegmentationMetricAccumulator, logits_to_predictions, per_class_rows
from task1.PretrainedAutoencoder.utils.model_utils import count_total_parameters, count_trainable_parameters, load_checkpoint, resolve_device
from task1.PretrainedAutoencoder.utils.seed import set_seed
from task1.PretrainedAutoencoder.utils.visualization import save_segmentation_preview


def parse_args() -> argparse.Namespace:
    """CLI arguments for segmentation evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model on full images.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override.")
    parser.add_argument("--data-root", default=None, help="Optional dataset root override.")
    parser.add_argument("--output-root", default=None, help="Optional output root override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional sliding-window batch size override.")
    parser.add_argument("--device", default=None, help="Optional device override.")
    parser.add_argument("--patch-size", type=int, default=None, help="Optional sliding-window patch size override.")
    return parser.parse_args()


def _to_cpu_tensor(batch_tensor: torch.Tensor) -> torch.Tensor:
    return batch_tensor.detach().cpu()


def main() -> None:
    """Evaluate one segmentation experiment and update the comparison CSV."""
    args = parse_args()
    config_path = prepare_task_runtime(args.config)
    config = apply_overrides(
        load_config(config_path),
        {
            "dataset.root": args.data_root,
            "output.root": args.output_root,
            "device": args.device,
            "evaluation.sliding_window.batch_size": args.batch_size,
            "evaluation.sliding_window.patch_size": args.patch_size,
        },
    )

    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device"))
    class_names = tuple(config["dataset"].get("class_names", ["other", "tumor", "stroma"]))

    output_dirs = prepare_output_dirs(config["output"]["root"])
    eval_root = ensure_dir(output_dirs["root"] / "evaluation" / args.split)
    save_config_snapshot(config, eval_root / "resolved_config.yaml")

    model_build_config = deepcopy(config)
    model_build_config["model"]["pretrained_encoder_path"] = None
    model = build_segmentation_model(model_build_config).to(device)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dirs["checkpoints"] / "best.pt"
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    model_info = {
        "checkpoint_path": str(checkpoint_path),
        "trainable_parameters": count_trainable_parameters(model),
        "total_parameters": count_total_parameters(model),
        "device": str(device),
    }
    write_metrics_json(eval_root / "model_info.json", model_info)

    records = discover_split_records(
        dataset_root=config["dataset"]["root"],
        split=args.split,
        mask_root=config["dataset"]["processed_mask_root"],
        require_tissue_geojson=True,
        require_mask=True,
    )
    transform = build_segmentation_transform(config, is_train=False)
    sliding_cfg = config["evaluation"]["sliding_window"]
    patch_size = int(sliding_cfg.get("patch_size", config["dataset"].get("patch_size", 512)))
    overlap = float(sliding_cfg.get("overlap", 0.25))
    sw_batch_size = int(sliding_cfg.get("batch_size", 2))
    save_overlay_count = int(config["evaluation"].get("save_overlay_count", 8))
    alpha = float(config.get("preprocessing", {}).get("overlay_alpha", 0.45))

    accumulator = SegmentationMetricAccumulator(num_classes=len(class_names), class_names=class_names)
    per_case_rows: List[Dict[str, object]] = []

    for case_idx, record in enumerate(tqdm(records, desc=f"Evaluating {args.split}")):
        image = read_tiff_image(record.image_path)
        mask = read_mask_image(record.mask_path)
        image_tensor, mask_tensor = transform(image, mask)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        mask_tensor = mask_tensor.unsqueeze(0).to(device)

        logits = sliding_window_inference(
            model=model,
            image_tensor=image_tensor,
            patch_size=patch_size,
            overlap=overlap,
            batch_size=sw_batch_size,
        )
        predictions = logits_to_predictions(logits)
        accumulator.update(predictions, mask_tensor)

        case_accumulator = SegmentationMetricAccumulator(num_classes=len(class_names), class_names=class_names)
        case_accumulator.update(predictions, mask_tensor)
        case_metrics = case_accumulator.compute()
        case_row = {
            "split": args.split,
            "stem": record.stem,
            "subgroup": record.subgroup,
            "mean_dice": case_metrics["mean_dice"],
            "mean_iou": case_metrics["mean_iou"],
            "mean_precision": case_metrics["mean_precision"],
            "mean_recall": case_metrics["mean_recall"],
            "mean_f1": case_metrics["mean_f1"],
            "pixel_accuracy": case_metrics["pixel_accuracy"],
        }
        for class_name in class_names:
            case_row[f"dice_{class_name}"] = case_metrics.get(f"dice_{class_name}", 0.0)
            case_row[f"iou_{class_name}"] = case_metrics.get(f"iou_{class_name}", 0.0)
            case_row[f"precision_{class_name}"] = case_metrics.get(f"precision_{class_name}", 0.0)
            case_row[f"recall_{class_name}"] = case_metrics.get(f"recall_{class_name}", 0.0)
            case_row[f"f1_{class_name}"] = case_metrics.get(f"f1_{class_name}", 0.0)
        per_case_rows.append(case_row)

        if case_idx < save_overlay_count:
            save_segmentation_preview(
                image=image,
                target_mask=mask,
                prediction_mask=predictions.squeeze(0).detach().cpu().numpy(),
                output_path=eval_root / "overlays" / f"{record.stem}.png",
                alpha=alpha,
            )

    overall_metrics = accumulator.compute()
    overall_metrics["model_info"] = model_info
    overall_metrics["split"] = args.split
    overall_metrics["model_name"] = config["model"]["name"]
    write_metrics_json(eval_root / "metrics.json", overall_metrics)
    write_csv_rows(per_class_rows(overall_metrics, class_names), eval_root / "per_class_metrics.csv")
    write_csv_rows(per_case_rows, eval_root / "per_case_metrics.csv")

    comparison_row = {
        "model": config["model"]["name"],
        "trainable_params": model_info["trainable_parameters"],
        "mean_dice": overall_metrics["mean_dice"],
        "dice_other": overall_metrics.get("dice_other", 0.0),
        "dice_tumor": overall_metrics.get("dice_tumor", 0.0),
        "dice_stroma": overall_metrics.get("dice_stroma", 0.0),
        "mean_iou": overall_metrics["mean_iou"],
        "mean_precision": overall_metrics["mean_precision"],
        "mean_recall": overall_metrics["mean_recall"],
        "mean_f1": overall_metrics["mean_f1"],
        "pixel_accuracy": overall_metrics["pixel_accuracy"],
    }
    comparison_path = Path(output_dirs["root"]).parent / "comparison" / "segmentation_comparison.csv"
    upsert_csv_row(comparison_row, comparison_path, key_field="model")

    print(f"Evaluation complete for model={config['model']['name']} split={args.split}")
    print(f"Mean Dice: {overall_metrics['mean_dice']:.4f}")
    print(f"Mean IoU: {overall_metrics['mean_iou']:.4f}")
    print(f"Mean Precision: {overall_metrics['mean_precision']:.4f}")
    print(f"Mean Recall: {overall_metrics['mean_recall']:.4f}")
    print(f"Mean F1: {overall_metrics['mean_f1']:.4f}")
    print(f"Pixel Accuracy: {overall_metrics['pixel_accuracy']:.4f}")


if __name__ == "__main__":
    main()
