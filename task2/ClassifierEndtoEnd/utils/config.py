"""Configuration loading and lightweight CLI override helpers."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


def resolve_config_path(config_path: str | Path) -> Path:
    """Resolve a config path independently of the current working directory."""
    return Path(config_path).expanduser().resolve()


def infer_task_root_from_config(config_path: str | Path) -> Path:
    """Infer the task root from a config stored under <task-root>/configs/."""
    return resolve_config_path(config_path).parent.parent


def prepare_task_runtime(config_path: str | Path) -> Path:
    """Switch the working directory to the task root derived from the config path."""
    resolved_config_path = resolve_config_path(config_path)
    os.chdir(infer_task_root_from_config(resolved_config_path))
    return resolved_config_path


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a nested dictionary."""
    config_path = resolve_config_path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge updates into a base dictionary."""
    result = deepcopy(base)
    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def set_by_dotted_path(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Assign a value into a nested dictionary using dot notation."""
    keys = dotted_key.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def apply_overrides(config: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply non-None CLI overrides to a configuration dictionary."""
    updated = deepcopy(config)
    for dotted_key, value in overrides.items():
        if value is not None:
            set_by_dotted_path(updated, dotted_key, value)
    return updated


def save_config_snapshot(config: Dict[str, Any], output_path: str | Path) -> None:
    """Save a resolved configuration for reproducibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def get_config_value(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a nested value using dot notation."""
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def require_keys(config: Mapping[str, Any], dotted_keys: Iterable[str]) -> None:
    """Raise a KeyError if a required dotted path is missing."""
    missing = [key for key in dotted_keys if get_config_value(config, key) is None]
    if missing:
        raise KeyError(f"Missing required config entries: {', '.join(missing)}")
