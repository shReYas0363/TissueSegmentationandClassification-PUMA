"""Generate offline tissue segmentation masks from GeoJSON annotations."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from task1.PretrainedAutoencoder.utils.config import apply_overrides, load_config, prepare_task_runtime
from task1.PretrainedAutoencoder.utils.io import (
    DEFAULT_SPLITS,
    discover_split_records,
    ensure_dir,
    load_geojson,
    read_tiff_image,
    read_tiff_shape,
    save_mask_png,
    write_csv_rows,
    write_json,
)
from task1.PretrainedAutoencoder.utils.visualization import save_segmentation_preview


# Refactored from the original repository-level masks.py.
CLASS_TO_ID = {
    "other": 0,
    "tissue_tumor": 1,
    "tissue_stroma": 2,
}
CLASS_APPLICATION_ORDER = ("other", "tissue_stroma", "tissue_tumor")


def map_original_label(original_label: str) -> str:
    """Map original annotation labels to the 3-class coursework setup."""
    if original_label == "tissue_tumor":
        return "tissue_tumor"
    if original_label == "tissue_stroma":
        return "tissue_stroma"
    return "other"


def ring_to_int_points(ring: Iterable[Iterable[float]]) -> np.ndarray:
    """Convert a polygon ring to rounded integer points for rasterization."""
    points = np.asarray(ring, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Invalid ring shape: {points.shape}")
    return np.round(points).astype(np.int32)


def fill_polygon_with_holes(mask: np.ndarray, polygon_coords: List[List[List[float]]]) -> None:
    """Rasterize a polygon while preserving any interior holes."""
    if not polygon_coords:
        return

    outer_ring = ring_to_int_points(polygon_coords[0])
    cv2.fillPoly(mask, [outer_ring], 255)

    for hole_ring in polygon_coords[1:]:
        hole_points = ring_to_int_points(hole_ring)
        cv2.fillPoly(mask, [hole_points], 0)


def generate_class_masks_from_geojson(
    geojson_payload: Dict[str, object],
    image_shape: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """Create one binary mask per target class from a GeoJSON payload."""
    class_masks = {
        "other": np.zeros(image_shape, dtype=np.uint8),
        "tissue_stroma": np.zeros(image_shape, dtype=np.uint8),
        "tissue_tumor": np.zeros(image_shape, dtype=np.uint8),
    }

    for feature_idx, feature in enumerate(geojson_payload.get("features", [])):
        properties = feature.get("properties") or {}
        classification = properties.get("classification") or {}
        original_label = classification.get("name")
        if original_label is None:
            continue

        label = map_original_label(original_label)
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if coordinates is None:
            continue

        try:
            if geometry_type == "Polygon":
                fill_polygon_with_holes(class_masks[label], coordinates)
            elif geometry_type == "MultiPolygon":
                for polygon_coords in coordinates:
                    fill_polygon_with_holes(class_masks[label], polygon_coords)
        except Exception as exc:
            raise ValueError(
                f"Failed to rasterize feature {feature_idx} "
                f"(label={original_label}, geometry={geometry_type}): {exc}"
            ) from exc

    return class_masks


def combine_class_masks(class_masks: Dict[str, np.ndarray]) -> np.ndarray:
    """Combine per-class masks using explicit precedence Other < Stroma < Tumor."""
    first_mask = next(iter(class_masks.values()))
    output = np.zeros(first_mask.shape, dtype=np.uint8)
    for class_name in CLASS_APPLICATION_ORDER:
        output[class_masks[class_name] > 0] = CLASS_TO_ID[class_name]
    return output


def process_case(
    record_dict: Dict[str, str],
    output_root: str,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Generate a single single-channel training mask."""
    record_stem = record_dict["stem"]
    split = record_dict["split"]
    image_path = Path(record_dict["image_path"])
    tissue_geojson_path = Path(record_dict["tissue_geojson_path"])
    output_mask_path = Path(output_root) / split / f"{record_stem}.png"

    if output_mask_path.exists() and not overwrite:
        height, width = read_tiff_shape(image_path)
        return {
            "split": split,
            "stem": record_stem,
            "image_path": str(image_path),
            "tissue_geojson_path": str(tissue_geojson_path),
            "mask_path": str(output_mask_path),
            "image_height": height,
            "image_width": width,
            "status": "skipped_existing",
        }

    image_height, image_width = read_tiff_shape(image_path)
    geojson_payload = load_geojson(tissue_geojson_path)
    class_masks = generate_class_masks_from_geojson(geojson_payload, (image_height, image_width))
    output_mask = combine_class_masks(class_masks)

    save_mask_png(output_mask_path, output_mask)
    return {
        "split": split,
        "stem": record_stem,
        "image_path": str(image_path),
        "tissue_geojson_path": str(tissue_geojson_path),
        "mask_path": str(output_mask_path),
        "image_height": image_height,
        "image_width": image_width,
        "status": "generated",
    }


def save_preview_overlays(
    rows: List[Dict[str, object]],
    output_root: Path,
    preview_count: int,
    alpha: float,
) -> None:
    """Save overlay previews for a small subset of generated masks."""
    preview_dir = ensure_dir(output_root / "previews")
    for row in rows[:preview_count]:
        image = read_tiff_image(row["image_path"])
        mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_UNCHANGED)
        save_segmentation_preview(
            image=image,
            target_mask=mask,
            prediction_mask=None,
            output_path=preview_dir / f"{row['split']}_{row['stem']}.png",
            alpha=alpha,
        )


def parse_args() -> argparse.Namespace:
    """CLI arguments for offline mask generation."""
    parser = argparse.ArgumentParser(description="Generate offline tissue segmentation masks.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--input-root", default=None, help="Optional dataset root override.")
    parser.add_argument("--output-root", default=None, help="Optional processed mask root override.")
    parser.add_argument("--split", default=None, choices=list(DEFAULT_SPLITS), help="Optional split to process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated masks.")
    parser.add_argument("--preview-count", type=int, default=None, help="Number of overlay previews to save.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override worker count.")
    return parser.parse_args()


def main() -> None:
    """Entry point for mask generation."""
    args = parse_args()
    config_path = prepare_task_runtime(args.config)
    config = load_config(config_path)
    config = apply_overrides(
        config,
        {
            "dataset.root": args.input_root,
            "preprocessing.output_root": args.output_root,
            "preprocessing.preview_count": args.preview_count,
            "preprocessing.num_workers": args.num_workers,
        },
    )

    dataset_root = Path(config["dataset"]["root"])
    output_root = Path(config.get("preprocessing", {}).get("output_root", config["dataset"]["processed_mask_root"]))
    ensure_dir(output_root)

    preview_count = int(config.get("preprocessing", {}).get("preview_count", 0))
    alpha = float(config.get("preprocessing", {}).get("overlay_alpha", 0.45))
    requested_split = args.split
    splits = [requested_split] if requested_split else [
        config["dataset"].get("train_split", DEFAULT_SPLITS[0]),
        config["dataset"].get("val_split", DEFAULT_SPLITS[1]),
        config["dataset"].get("test_split", DEFAULT_SPLITS[2]),
    ]

    worker_count = config.get("preprocessing", {}).get("num_workers", 0)
    rows: List[Dict[str, object]] = []

    for split in splits:
        split_records = discover_split_records(
            dataset_root=dataset_root,
            split=split,
            mask_root=None,
            require_tissue_geojson=True,
            require_mask=False,
        )
        ensure_dir(output_root / split)
        print(f"Generating masks for split='{split}' with {len(split_records)} cases")

        process_fn = partial(
            process_case,
            output_root=str(output_root),
            overwrite=args.overwrite,
        )
        record_dicts = [record.to_dict() for record in split_records]

        if worker_count and int(worker_count) > 0:
            with ProcessPoolExecutor(max_workers=int(worker_count)) as executor:
                split_rows = list(
                    tqdm(
                        executor.map(process_fn, record_dicts),
                        total=len(record_dicts),
                        desc=f"Mask generation [{split}]",
                    )
                )
        else:
            split_rows = [
                process_fn(record_dict)
                for record_dict in tqdm(record_dicts, total=len(record_dicts), desc=f"Mask generation [{split}]")
            ]

        rows.extend(split_rows)

    write_csv_rows(rows, output_root / "mask_manifest.csv")
    write_json({"rows": rows}, output_root / "mask_manifest.json")

    if preview_count > 0:
        save_preview_overlays(rows, output_root, preview_count=preview_count, alpha=alpha)

    status_counter = Counter(row["status"] for row in rows)
    print(f"Mask generation complete. Status counts: {dict(status_counter)}")
    print(f"Saved manifest CSV to {output_root / 'mask_manifest.csv'}")


if __name__ == "__main__":
    main()
