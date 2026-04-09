"""Dataset discovery and file IO helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
import tifffile


DEFAULT_SPLITS: Sequence[str] = ("train", "validation", "test")
IMAGE_EXTENSIONS = (".tif", ".tiff")
MASK_EXTENSIONS = (".png", ".tif", ".tiff")
SUPPORTED_TISSUE_GEOMETRIES = ("Polygon", "MultiPolygon")


@dataclass(frozen=True)
class CaseRecord:
    """Metadata for one dataset item."""

    split: str
    stem: str
    image_path: str
    tissue_geojson_path: Optional[str]
    mask_path: Optional[str]
    subgroup: str

    def to_dict(self) -> Dict[str, str]:
        """Convert the dataclass to a dictionary."""
        return asdict(self)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: object, output_path: str | Path) -> None:
    """Write a JSON file with indentation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_csv_rows(rows: Iterable[Dict[str, object]], output_path: str | Path) -> None:
    """Write a sequence of dictionaries to CSV."""
    rows = list(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(row: Dict[str, object], output_path: str | Path) -> None:
    """Append one row to a CSV file, creating it with a header if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def upsert_csv_row(row: Dict[str, object], output_path: str | Path, key_field: str) -> None:
    """Insert or replace a CSV row based on one key column."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

    updated = False
    normalized_rows: List[Dict[str, object]] = []
    for existing_row in rows:
        if existing_row.get(key_field) == str(row[key_field]):
            normalized_rows.append({key: str(value) for key, value in row.items()})
            updated = True
        else:
            normalized_rows.append(existing_row)

    if not updated:
        normalized_rows.append({key: str(value) for key, value in row.items()})

    write_csv_rows(normalized_rows, output_path)


def image_stem_to_tissue_geojson_name(stem: str) -> str:
    """Map an image stem to the default tissue annotation filename."""
    return f"{stem}_tissue.geojson"


def tissue_geojson_path_to_image_stem(path: str | Path) -> str:
    """Map a tissue GeoJSON path to the corresponding image stem."""
    stem = Path(path).stem
    if stem.endswith("_tissue"):
        return stem[: -len("_tissue")]
    return stem


def infer_subgroup_from_stem(stem: str) -> str:
    """Infer primary vs metastatic subgroup from the filename stem."""
    lowered = stem.lower()
    if "primary" in lowered:
        return "primary"
    if "metastatic" in lowered:
        return "metastatic"
    return "unknown"


def list_non_hidden_files(directory: str | Path, suffixes: Sequence[str]) -> List[Path]:
    """List non-hidden files in a directory matching one of the given suffixes."""
    directory = Path(directory)
    if not directory.exists():
        return []
    files = []
    for path in sorted(directory.iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            files.append(path)
    return files


def resolve_mask_path(mask_dir: str | Path, stem: str) -> Optional[Path]:
    """Find a generated mask path using supported mask extensions."""
    mask_dir = Path(mask_dir)
    for extension in MASK_EXTENSIONS:
        candidate = mask_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def discover_split_records(
    dataset_root: str | Path,
    split: str,
    mask_root: str | Path | None = None,
    require_tissue_geojson: bool = True,
    require_mask: bool = False,
) -> List[CaseRecord]:
    """Discover dataset items for a split using the default naming convention."""
    dataset_root = Path(dataset_root)
    image_dir = dataset_root / split / "image"
    tissue_dir = dataset_root / split / "tissue"
    mask_dir = Path(mask_root) / split if mask_root is not None else None

    records: List[CaseRecord] = []
    for image_path in list_non_hidden_files(image_dir, IMAGE_EXTENSIONS):
        stem = image_path.stem
        tissue_geojson_path = tissue_dir / image_stem_to_tissue_geojson_name(stem)
        if require_tissue_geojson and not tissue_geojson_path.exists():
            raise FileNotFoundError(
                f"Missing tissue GeoJSON for image '{image_path.name}': {tissue_geojson_path}"
            )

        mask_path = None
        if mask_dir is not None:
            mask_path = resolve_mask_path(mask_dir, stem)
            if require_mask and mask_path is None:
                raise FileNotFoundError(
                    f"Missing generated mask for image '{image_path.name}' in {mask_dir}"
                )

        records.append(
            CaseRecord(
                split=split,
                stem=stem,
                image_path=str(image_path),
                tissue_geojson_path=str(tissue_geojson_path) if tissue_geojson_path.exists() else None,
                mask_path=str(mask_path) if mask_path is not None else None,
                subgroup=infer_subgroup_from_stem(stem),
            )
        )
    return records


def discover_all_split_records(
    dataset_root: str | Path,
    mask_root: str | Path | None = None,
    splits: Sequence[str] = DEFAULT_SPLITS,
    require_tissue_geojson: bool = True,
    require_mask: bool = False,
) -> Dict[str, List[CaseRecord]]:
    """Discover dataset records for every configured split."""
    return {
        split: discover_split_records(
            dataset_root=dataset_root,
            split=split,
            mask_root=mask_root,
            require_tissue_geojson=require_tissue_geojson,
            require_mask=require_mask,
        )
        for split in splits
    }


def read_tiff_shape(path: str | Path) -> tuple[int, int]:
    """Read TIFF height and width without assuming a fixed image size."""
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        shape = tif.series[0].shape

    if len(shape) == 2:
        height, width = shape
    elif len(shape) == 3:
        if shape[-1] in (1, 3, 4):
            height, width = shape[0], shape[1]
        else:
            height, width = shape[-2], shape[-1]
    else:
        raise ValueError(f"Unsupported TIFF shape for {path}: {shape}")
    return int(height), int(width)


def read_tiff_image(path: str | Path) -> np.ndarray:
    """Read a TIFF image and normalize it to an HWC RGB-style NumPy array."""
    image = tifffile.imread(str(path))
    image = np.asarray(image)

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3:
        if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] >= 4:
            image = image[..., :3]
    else:
        raise ValueError(f"Unsupported image shape for {path}: {image.shape}")

    return image


def read_mask_image(path: str | Path) -> np.ndarray:
    """Read a generated segmentation mask as a single-channel integer map."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        mask = tifffile.imread(str(path))
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if mask.ndim != 2:
        raise ValueError(f"Mask should be single channel, got shape {mask.shape} for {path}")
    return np.asarray(mask, dtype=np.uint8)


def save_mask_png(path: str | Path, mask: np.ndarray) -> None:
    """Save a single-channel class-ID mask as PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), mask.astype(np.uint8))
    if not success:
        raise IOError(f"Failed to save mask to {path}")


def load_geojson(path: str | Path) -> Dict[str, object]:
    """Load a GeoJSON file from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
