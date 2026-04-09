"""Simple JSON/CSV logging helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from task1.PretrainedAutoencoder.utils.io import append_csv_row, ensure_dir, write_json


def prepare_output_dirs(root: str | Path) -> Dict[str, Path]:
    """Create a standard experiment output directory structure."""
    root = ensure_dir(root)
    paths = {
        "root": root,
        "checkpoints": ensure_dir(root / "checkpoints"),
        "logs": ensure_dir(root / "logs"),
        "visualizations": ensure_dir(root / "visualizations"),
        "metrics": ensure_dir(root / "metrics"),
    }
    return paths


def append_metrics(metrics_path: str | Path, row: Dict[str, object]) -> None:
    """Append a flat metrics row to CSV."""
    append_csv_row(row, metrics_path)


def write_metrics_json(metrics_path: str | Path, payload: Dict[str, object]) -> None:
    """Write structured metrics to JSON."""
    write_json(payload, metrics_path)


def append_jsonl(output_path: str | Path, payload: Dict[str, object]) -> None:
    """Append one JSON object per line."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
