"""Dataset loader for generated nuclei patches and the official Task 2 test set."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np
from torch.utils.data import Dataset

from task2.ClassifierEndtoEnd.utils.io import list_non_hidden_files
from task2.ClassifierEndtoEnd.utils.nuclei_classification import (
    DEFAULT_CLASS_NAMES,
    canonicalize_class_name,
    generated_class_dir_candidates,
    get_class_names,
    get_label_mapping,
    parse_official_test_path,
    parse_subgroup_from_name,
)


GENERATED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class NucleiClassificationRecord:
    """One dataset item for Task 2 nuclei classification."""

    split: str
    path: str
    label: int
    class_name: str
    sample_id: str
    source_stem: str
    subgroup: str

    def to_dict(self) -> Dict[str, object]:
        """Convert the dataclass into a flat dictionary."""
        return asdict(self)


def _normalize_max_samples(max_samples: Optional[int]) -> Optional[int]:
    if max_samples is None:
        return None
    limit = int(max_samples)
    return limit if limit > 0 else None


def _read_generated_patch(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read generated nuclei patch: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_official_test_patch(path: Path) -> np.ndarray:
    patch = np.load(path)
    patch = np.asarray(patch)
    if patch.ndim != 3 or patch.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB .npy patch, got shape {patch.shape} for {path}")
    return patch


class NucleiClassificationDataset(Dataset):
    """Dataset for generated train/val nuclei patches and official Task 2 test patches."""

    def __init__(
        self,
        mode: str,
        split: str,
        generated_root: str | Path | None = None,
        task2_test_root: str | Path | None = None,
        transform=None,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        label_mapping: Mapping[str, int] | None = None,
        max_samples: Optional[int] = None,
    ) -> None:
        if mode not in {"generated_split", "official_task2_test"}:
            raise ValueError(f"Unsupported nuclei dataset mode: {mode}")

        self.mode = mode
        self.split = str(split)
        self.transform = transform
        self.class_names = get_class_names(class_names)
        self.label_mapping = get_label_mapping(label_mapping)
        self.max_samples = _normalize_max_samples(max_samples)

        if mode == "generated_split":
            if generated_root is None:
                raise ValueError("generated_root is required for generated_split mode.")
            self.root = Path(generated_root)
            self.records = self._discover_generated_records(self.root / self.split)
        else:
            if task2_test_root is None:
                raise ValueError("task2_test_root is required for official_task2_test mode.")
            self.root = Path(task2_test_root)
            self.records = self._discover_official_test_records(self.root)

        if self.max_samples is not None:
            self.records = self.records[: self.max_samples]
        if not self.records:
            raise ValueError(
                f"No nuclei classification samples found for mode={self.mode} split={self.split} root={self.root}"
            )

    def _discover_generated_records(self, split_root: Path) -> List[NucleiClassificationRecord]:
        if not split_root.exists():
            raise FileNotFoundError(f"Generated nuclei split directory not found: {split_root}")

        records: List[NucleiClassificationRecord] = []
        for class_name in self.class_names:
            candidate_dirs = generated_class_dir_candidates(class_name)
            class_dir = None
            for candidate_name in candidate_dirs:
                candidate_dir = split_root / candidate_name
                if candidate_dir.exists():
                    class_dir = candidate_dir
                    break

            if class_dir is None:
                raise FileNotFoundError(
                    f"Could not find generated class directory for '{class_name}' under {split_root}"
                )

            for image_path in list_non_hidden_files(class_dir, GENERATED_IMAGE_EXTENSIONS):
                sample_id = image_path.stem
                records.append(
                    NucleiClassificationRecord(
                        split=self.split,
                        path=str(image_path),
                        label=self.label_mapping[class_name],
                        class_name=class_name,
                        sample_id=sample_id,
                        source_stem="",
                        subgroup=parse_subgroup_from_name(sample_id),
                    )
                )
        records.sort(key=lambda record: record.path)
        return records

    def _discover_official_test_records(self, test_root: Path) -> List[NucleiClassificationRecord]:
        if not test_root.exists():
            raise FileNotFoundError(f"Task 2 official test directory not found: {test_root}")

        records: List[NucleiClassificationRecord] = []
        for path in sorted(test_root.glob("*.npy")):
            if path.name.startswith("."):
                continue

            parsed = parse_official_test_path(path)
            class_name = canonicalize_class_name(parsed["class_name"])
            records.append(
                NucleiClassificationRecord(
                    split=self.split,
                    path=str(path),
                    label=self.label_mapping[class_name],
                    class_name=class_name,
                    sample_id=parsed["sample_id"],
                    source_stem=parsed["source_stem"],
                    subgroup=parsed["subgroup"],
                )
            )
        return records

    def __len__(self) -> int:
        """Return the dataset size."""
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        """Load one image patch and return it with Task 2 metadata."""
        record = self.records[index]
        path = Path(record.path)
        if self.mode == "generated_split":
            image = _read_generated_patch(path)
        else:
            image = _read_official_test_patch(path)

        image_tensor = self.transform(image) if self.transform is not None else image
        payload = record.to_dict()
        payload["image"] = image_tensor
        return payload
