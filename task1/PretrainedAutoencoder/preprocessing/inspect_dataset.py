"""Inspect dataset structure, filename matching, and annotation statistics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from task1.PretrainedAutoencoder.utils.config import apply_overrides, load_config, prepare_task_runtime
from task1.PretrainedAutoencoder.utils.io import (
    DEFAULT_SPLITS,
    SUPPORTED_TISSUE_GEOMETRIES,
    discover_split_records,
    ensure_dir,
    load_geojson,
    read_tiff_shape,
    write_csv_rows,
    write_json,
)


def summarize_split(dataset_root: Path, split: str) -> Dict[str, object]:
    """Collect split-level and case-level dataset statistics."""
    records = discover_split_records(
        dataset_root=dataset_root,
        split=split,
        mask_root=None,
        require_tissue_geojson=True,
        require_mask=False,
    )

    shape_counter = Counter()
    subgroup_counter = Counter()
    class_counter = Counter()
    geometry_counter = Counter()
    case_rows: List[Dict[str, object]] = []

    image_dir = dataset_root / split / "image"
    tissue_dir = dataset_root / split / "tissue"
    nuclei_dir = dataset_root / split / "nuclei"

    for record in tqdm(records, desc=f"Inspecting {split}", leave=False):
        height, width = read_tiff_shape(record.image_path)
        shape_counter[f"{height}x{width}"] += 1
        subgroup_counter[record.subgroup] += 1

        geojson = load_geojson(record.tissue_geojson_path)
        feature_count = len(geojson.get("features", []))
        for feature in geojson.get("features", []):
            properties = feature.get("properties") or {}
            classification = properties.get("classification") or {}
            class_name = classification.get("name", "<missing>")
            class_counter[class_name] += 1

            geometry = feature.get("geometry") or {}
            geometry_type = geometry.get("type", "<missing>")
            geometry_counter[geometry_type] += 1

        case_rows.append(
            {
                "split": split,
                "stem": record.stem,
                "subgroup": record.subgroup,
                "image_path": record.image_path,
                "tissue_geojson_path": record.tissue_geojson_path,
                "image_height": height,
                "image_width": width,
                "annotation_features": feature_count,
            }
        )

    summary = {
        "split": split,
        "num_images": len(records),
        "num_tissue_geojson": len([path for path in tissue_dir.glob("*.geojson")]),
        "num_nuclei_geojson": len([path for path in nuclei_dir.glob("*.geojson")]),
        "image_dir": str(image_dir),
        "tissue_dir": str(tissue_dir),
        "nuclei_dir": str(nuclei_dir),
        "shape_counts": dict(shape_counter),
        "subgroup_counts": dict(subgroup_counter),
        "tissue_class_feature_counts": dict(class_counter),
        "geometry_type_counts": dict(geometry_counter),
        "supported_geometries": list(SUPPORTED_TISSUE_GEOMETRIES),
        "case_rows": case_rows,
    }
    return summary


def build_dataset_summary(dataset_root: Path, splits: List[str]) -> Dict[str, object]:
    """Summarize every dataset split."""
    split_summaries = {}
    all_case_rows: List[Dict[str, object]] = []
    all_class_names = set()
    filename_examples = defaultdict(list)

    for split in splits:
        summary = summarize_split(dataset_root, split)
        split_summaries[split] = {key: value for key, value in summary.items() if key != "case_rows"}
        all_case_rows.extend(summary["case_rows"])
        all_class_names.update(summary["tissue_class_feature_counts"].keys())

        for row in summary["case_rows"][:3]:
            filename_examples[split].append(Path(row["image_path"]).name)

    return {
        "dataset_root": str(dataset_root),
        "splits": split_summaries,
        "all_class_names_found": sorted(all_class_names),
        "filename_examples": dict(filename_examples),
        "total_cases": len(all_case_rows),
        "case_rows": all_case_rows,
    }


def parse_args() -> argparse.Namespace:
    """CLI arguments for dataset inspection."""
    parser = argparse.ArgumentParser(description="Inspect the tissue dataset split structure.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--data-root", default=None, help="Optional dataset root override.")
    parser.add_argument("--output-root", default=None, help="Optional output root override.")
    return parser.parse_args()


def main() -> None:
    """Run dataset inspection and save summaries."""
    args = parse_args()
    config_path = prepare_task_runtime(args.config)
    config = load_config(config_path)
    config = apply_overrides(
        config,
        {
            "dataset.root": args.data_root,
            "output.root": args.output_root,
        },
    )

    dataset_root = Path(config["dataset"]["root"])
    output_root = Path(config["output"]["root"]) / "dataset_inspection"
    ensure_dir(output_root)

    splits = [
        config["dataset"].get("train_split", DEFAULT_SPLITS[0]),
        config["dataset"].get("val_split", DEFAULT_SPLITS[1]),
        config["dataset"].get("test_split", DEFAULT_SPLITS[2]),
    ]
    summary = build_dataset_summary(dataset_root, splits)

    write_json(summary, output_root / "dataset_summary.json")
    write_csv_rows(summary["case_rows"], output_root / "dataset_cases.csv")

    print(f"Dataset root: {dataset_root}")
    for split in splits:
        split_summary = summary["splits"][split]
        print(
            f"[{split}] images={split_summary['num_images']} "
            f"tissue_geojson={split_summary['num_tissue_geojson']} "
            f"nuclei_geojson={split_summary['num_nuclei_geojson']}"
        )
        print(f"  image shapes: {split_summary['shape_counts']}")
        print(f"  subgroups: {split_summary['subgroup_counts']}")
        print(f"  tissue classes: {split_summary['tissue_class_feature_counts']}")
        print(f"  geometries: {split_summary['geometry_type_counts']}")

    print(f"Saved summary JSON to {output_root / 'dataset_summary.json'}")
    print(f"Saved case CSV to {output_root / 'dataset_cases.csv'}")


if __name__ == "__main__":
    main()
