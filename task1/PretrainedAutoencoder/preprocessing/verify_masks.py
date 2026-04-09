"""Verify generated tissue masks against source images and summarize class balance."""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from task1.PretrainedAutoencoder.utils.config import apply_overrides, load_config, prepare_task_runtime
from task1.PretrainedAutoencoder.utils.io import (
    DEFAULT_SPLITS,
    discover_split_records,
    ensure_dir,
    read_mask_image,
    read_tiff_image,
    read_tiff_shape,
    write_csv_rows,
    write_json,
)
from task1.PretrainedAutoencoder.utils.visualization import save_segmentation_preview


VALID_CLASS_IDS = {0, 1, 2}


def summarize_mask_split(records: List[dict], overlay_dir: Path, sample_count: int, alpha: float) -> Dict[str, object]:
    """Validate and summarize one mask split."""
    split_counter = Counter()
    subgroup_counter = defaultdict(Counter)
    errors: List[str] = []
    rows: List[Dict[str, object]] = []

    sampled_records = random.sample(records, k=min(sample_count, len(records))) if records else []

    for record in tqdm(records, desc=f"Verifying {records[0]['split'] if records else 'split'}", leave=False):
        height, width = read_tiff_shape(record["image_path"])
        mask = read_mask_image(record["mask_path"])

        if mask.shape != (height, width):
            errors.append(
                f"Shape mismatch for {record['stem']}: image={(height, width)} mask={mask.shape}"
            )
            continue

        unique_ids = set(np.unique(mask).tolist())
        if not unique_ids.issubset(VALID_CLASS_IDS):
            errors.append(
                f"Invalid class IDs for {record['stem']}: found {sorted(unique_ids)}"
            )
            continue

        counts = {
            "other_pixels": int((mask == 0).sum()),
            "tumor_pixels": int((mask == 1).sum()),
            "stroma_pixels": int((mask == 2).sum()),
        }
        total_pixels = height * width
        split_counter.update(counts)
        subgroup_counter[record["subgroup"]].update(counts)

        rows.append(
            {
                "split": record["split"],
                "stem": record["stem"],
                "subgroup": record["subgroup"],
                "image_height": height,
                "image_width": width,
                "other_pixels": counts["other_pixels"],
                "tumor_pixels": counts["tumor_pixels"],
                "stroma_pixels": counts["stroma_pixels"],
                "other_fraction": counts["other_pixels"] / total_pixels,
                "tumor_fraction": counts["tumor_pixels"] / total_pixels,
                "stroma_fraction": counts["stroma_pixels"] / total_pixels,
            }
        )

    for record in sampled_records:
        image = read_tiff_image(record["image_path"])
        mask = read_mask_image(record["mask_path"])
        save_segmentation_preview(
            image=image,
            target_mask=mask,
            prediction_mask=None,
            output_path=overlay_dir / f"{record['split']}_{record['stem']}.png",
            alpha=alpha,
        )

    total_pixels = max(sum(split_counter.values()), 1)
    subgroup_summary = {}
    for subgroup, counts in subgroup_counter.items():
        subgroup_total = max(sum(counts.values()), 1)
        subgroup_summary[subgroup] = {
            "pixel_counts": dict(counts),
            "percentages": {
                "other": counts["other_pixels"] / subgroup_total,
                "tumor": counts["tumor_pixels"] / subgroup_total,
                "stroma": counts["stroma_pixels"] / subgroup_total,
            },
        }

    summary = {
        "num_cases": len(records),
        "errors": errors,
        "pixel_counts": dict(split_counter),
        "percentages": {
            "other": split_counter["other_pixels"] / total_pixels,
            "tumor": split_counter["tumor_pixels"] / total_pixels,
            "stroma": split_counter["stroma_pixels"] / total_pixels,
        },
        "subgroups": subgroup_summary,
        "rows": rows,
    }
    return summary


def parse_args() -> argparse.Namespace:
    """CLI arguments for mask verification."""
    parser = argparse.ArgumentParser(description="Verify generated tissue segmentation masks.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--data-root", default=None, help="Optional dataset root override.")
    parser.add_argument("--mask-root", default=None, help="Optional processed mask root override.")
    parser.add_argument("--output-root", default=None, help="Optional output root override.")
    parser.add_argument("--sample-count", type=int, default=8, help="Number of overlays to save per run.")
    return parser.parse_args()


def main() -> None:
    """Verify masks and save summary outputs."""
    args = parse_args()
    config_path = prepare_task_runtime(args.config)
    config = load_config(config_path)
    config = apply_overrides(
        config,
        {
            "dataset.root": args.data_root,
            "dataset.processed_mask_root": args.mask_root,
            "output.root": args.output_root,
        },
    )

    dataset_root = Path(config["dataset"]["root"])
    mask_root = Path(config["dataset"]["processed_mask_root"])
    output_root = Path(config["output"]["root"]) / "mask_verification"
    overlay_dir = ensure_dir(output_root / "overlays")
    alpha = float(config.get("preprocessing", {}).get("overlay_alpha", 0.45))

    splits = [
        config["dataset"].get("train_split", DEFAULT_SPLITS[0]),
        config["dataset"].get("val_split", DEFAULT_SPLITS[1]),
        config["dataset"].get("test_split", DEFAULT_SPLITS[2]),
    ]

    overall_summary: Dict[str, object] = {"splits": {}}
    all_rows: List[Dict[str, object]] = []

    for split in splits:
        records = [
            record.to_dict()
            for record in discover_split_records(
                dataset_root=dataset_root,
                split=split,
                mask_root=mask_root,
                require_tissue_geojson=True,
                require_mask=True,
            )
        ]
        summary = summarize_mask_split(records, overlay_dir, sample_count=args.sample_count, alpha=alpha)
        overall_summary["splits"][split] = {key: value for key, value in summary.items() if key != "rows"}
        all_rows.extend(summary["rows"])

    write_json(overall_summary, output_root / "mask_verification_summary.json")
    write_csv_rows(all_rows, output_root / "mask_pixel_summary.csv")

    for split, split_summary in overall_summary["splits"].items():
        print(f"[{split}] cases={split_summary['num_cases']} errors={len(split_summary['errors'])}")
        print(f"  percentages={split_summary['percentages']}")
        print(f"  subgroups={split_summary['subgroups']}")

    print(f"Saved verification summary to {output_root / 'mask_verification_summary.json'}")
    print(f"Saved pixel summary CSV to {output_root / 'mask_pixel_summary.csv'}")


if __name__ == "__main__":
    main()
