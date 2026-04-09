"""Shared constants and filename helpers for Task 2 nuclei classification."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping


DEFAULT_CLASS_NAMES = ("tumor", "lymphocyte", "histiocyte")
DEFAULT_LABEL_MAPPING = {
    "tumor": 0,
    "lymphocyte": 1,
    "histiocyte": 2,
}
RAW_TO_CANONICAL_CLASS_NAME = {
    "nuclei_tumor": "tumor",
    "nuclei_lymphocyte": "lymphocyte",
    "nuclei_histiocyte": "histiocyte",
}

_OFFICIAL_TEST_PATTERN = re.compile(
    r"^(?P<source_stem>.+)_nuclei_(?P<class_name>tumor|lymphocyte|histiocyte)_(?P<sample_suffix>.+)$"
)


def get_class_names(configured_class_names) -> tuple[str, ...]:
    """Validate and return the fixed Task 2 class order."""
    if configured_class_names is None:
        return DEFAULT_CLASS_NAMES

    class_names = tuple(str(name) for name in configured_class_names)
    if class_names != DEFAULT_CLASS_NAMES:
        raise ValueError(
            "Task 2(a) requires the fixed class order: tumor, lymphocyte, histiocyte."
        )
    return class_names


def get_label_mapping(configured_mapping: Mapping[str, int] | None = None) -> Dict[str, int]:
    """Validate and return the fixed Task 2 label mapping."""
    if configured_mapping is None:
        return dict(DEFAULT_LABEL_MAPPING)

    mapping = {str(key): int(value) for key, value in configured_mapping.items()}
    if mapping != DEFAULT_LABEL_MAPPING:
        raise ValueError(
            "Task 2(a) requires the fixed label mapping tumor=0, lymphocyte=1, histiocyte=2."
        )
    return mapping


def canonicalize_class_name(name: str) -> str:
    """Map a raw or canonical label name onto the canonical Task 2 name."""
    normalized = str(name).strip().lower()
    if normalized in DEFAULT_LABEL_MAPPING:
        return normalized
    if normalized in RAW_TO_CANONICAL_CLASS_NAME:
        return RAW_TO_CANONICAL_CLASS_NAME[normalized]
    raise ValueError(f"Unsupported nuclei class name: {name}")


def parse_subgroup_from_name(name: str) -> str:
    """Infer primary vs metastatic subgroup from a filename or stem."""
    lowered = str(name).lower()
    if "primary" in lowered:
        return "primary"
    if "metastatic" in lowered:
        return "metastatic"
    return "unknown"


def parse_official_test_path(path: str | Path) -> Dict[str, str]:
    """Extract label and metadata from an official Task 2 test filename."""
    path = Path(path)
    match = _OFFICIAL_TEST_PATTERN.match(path.stem)
    if match is None:
        raise ValueError(f"Could not parse Task2 test filename: {path.name}")

    source_stem = match.group("source_stem")
    class_name = match.group("class_name")
    return {
        "sample_id": path.stem,
        "source_stem": source_stem,
        "class_name": class_name,
        "subgroup": parse_subgroup_from_name(source_stem),
    }


def generated_class_dir_candidates(class_name: str) -> tuple[str, ...]:
    """Return canonical and backwards-compatible directory names for one class."""
    canonical_name = canonicalize_class_name(class_name)
    raw_names = tuple(
        raw_name
        for raw_name, mapped_name in RAW_TO_CANONICAL_CLASS_NAME.items()
        if mapped_name == canonical_name
    )
    return (canonical_name, *raw_names)
