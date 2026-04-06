import json
import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from shapely.geometry import shape


DATASET_ROOT = Path("Dataset_Splits")
OUTPUT_ROOT = Path("nuclei_dataset")

PATCH_HALF = 50          
TRAIN_PER_CLASS = 2500
VAL_PER_CLASS = 700
CONTRASTIVE_PER_CLASS = 1000

CLASSES_TO_EXTRACT = {"nuclei_tumor", "nuclei_lymphocyte", "nuclei_histiocyte"}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_geojson_entries(geojson_path: Path):
    """Give (centroid_x, centroid_y, class_name) for every feature."""
    with open(geojson_path, "r") as f:
        data = json.load(f)

    for feat in data["features"]:
        geom = shape(feat["geometry"])
        cx, cy = geom.centroid.x, geom.centroid.y
        class_name = feat["properties"]["classification"]["name"]
        yield int(round(cx)), int(round(cy)), class_name


def extract_patch(image: np.ndarray, cx: int, cy: int, half: int = PATCH_HALF):
    """
    Extract a (2*half by 2*half) patch centred at (cx, cy).
    Zeropadif the bounding box exceeds image boundaries.
    """
    h, w = image.shape[:2]
    patch_size = 2 * half

    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half

    sx1 = max(x1, 0)
    sy1 = max(y1, 0)
    sx2 = min(x2, w)
    sy2 = min(y2, h)

    dx1 = sx1 - x1
    dy1 = sy1 - y1
    dx2 = dx1 + (sx2 - sx1)
    dy2 = dy1 + (sy2 - sy1)

    if image.ndim == 3:
        patch = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
    else:
        patch = np.zeros((patch_size, patch_size), dtype=image.dtype)

    patch[dy1:dy2, dx1:dx2] = image[sy1:sy2, sx1:sx2]
    return patch


def collect_nuclei_by_image(split: str):
    """
    Return
        {class_name: {image_stem: [(image_path, cx, cy), …]}}
    Grouped by source image so we can split at the image level.
    """
    nuclei_dir = DATASET_ROOT / split / "nuclei"
    image_dir = DATASET_ROOT / split / "image"


    class_entries = defaultdict(lambda: defaultdict(list))

    for gj_file in sorted(nuclei_dir.glob("*.geojson")):
        stem = gj_file.stem.replace("_nuclei", "")
        img_path = image_dir / (stem + ".tif")
        if not img_path.exists():
            print(f"[WARN] Image not found for {gj_file.name}, skipping.")
            continue

        for cx, cy, cls in load_geojson_entries(gj_file):
            if cls not in CLASSES_TO_EXTRACT:
                continue
            class_entries[cls][stem].append((str(img_path), cx, cy))

    return class_entries


def save_patches(entries, out_dir: Path, labels_dict: dict | None = None):
    """
    Given a list of (image_path, cx, cy, class_name) tuples,
    save patches to out_dir/<class_name>/ and populate labels_dict.
    """
    
    image_cache = {}

    for img_path, cx, cy, cls in entries:
        if img_path not in image_cache:
            image_cache[img_path] = np.array(Image.open(img_path))

        image = image_cache[img_path]
        patch = extract_patch(image, cx, cy)

        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        idx = len(list(cls_dir.glob("*.png")))
        fname = f"{cls}_{idx:05d}.png"
        save_path = cls_dir / fname
        Image.fromarray(patch).save(save_path)

        if labels_dict is not None:
            labels_dict[fname] = cls


# ── Main ───────────────────────────────────────────────────────────────────
def split_images_for_class(image_stems: list[str],
                           train_ratio: float = 0.6,
                           val_ratio: float = 0.17):
    """
    Split a list of image stems into train / val / contrastive groups.
    Remaining images go to contrastive.
    """
    random.shuffle(image_stems)
    n = len(image_stems)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    return (
        image_stems[:n_train],
        image_stems[n_train:n_train + n_val],
        image_stems[n_train + n_val:],
    )


def sample_from_image_group(image_dict: dict, stems: list[str], n: int):
    """
    Sample up to `n` nuclei from a set of images.
    Falls back to sampling with replacement if the pool is too small.
    """
    pool = []
    for s in stems:
        pool.extend(image_dict.get(s, []))
    random.shuffle(pool)

    if len(pool) >= n:
        return pool[:n]
    print(f"  [WARN] only {len(pool)} nuclei in group, need {n}. "
          f"Sampling with replacement.")
    return random.choices(pool, k=n)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


    print("Collecting nuclei from TRAIN split …")
    train_by_img = collect_nuclei_by_image("train")
    print("Collecting nuclei from VALIDATION split …")
    val_by_img = collect_nuclei_by_image("validation")

    all_by_img: dict[str, dict[str, list]] = defaultdict(dict)
    for cls in CLASSES_TO_EXTRACT:
        for stem, entries in train_by_img.get(cls, {}).items():
            all_by_img[cls][stem] = entries
        for stem, entries in val_by_img.get(cls, {}).items():
            all_by_img[cls][stem] = entries

    print("\n── Available nuclei per class ──")
    for cls in sorted(all_by_img):
        total = sum(len(v) for v in all_by_img[cls].values())
        n_images = len(all_by_img[cls])
        print(f"  {cls}: {total} nuclei across {n_images} images")


    train_entries = []
    val_entries = []
    contrastive_entries = []

    for cls in sorted(all_by_img):
        image_dict = all_by_img[cls]
        stems = list(image_dict.keys())

        train_stems, val_stems, contr_stems = split_images_for_class(stems)
        print(f"\n  {cls}: {len(train_stems)} train images, "
              f"{len(val_stems)} val images, {len(contr_stems)} contrastive images")

        t_sel = sample_from_image_group(image_dict, train_stems, TRAIN_PER_CLASS)
        v_sel = sample_from_image_group(image_dict, val_stems, VAL_PER_CLASS)
        c_sel = sample_from_image_group(image_dict, contr_stems, CONTRASTIVE_PER_CLASS)

        train_entries.extend([(p, x, y, cls) for p, x, y in t_sel])
        val_entries.extend([(p, x, y, cls) for p, x, y in v_sel])
        contrastive_entries.extend([(p, x, y, cls) for p, x, y in c_sel])


    train_labels = {}
    val_labels = {}

    print(f"\nExtracting {len(train_entries)} training patches …")
    save_patches(train_entries, OUTPUT_ROOT / "train", train_labels)

    print(f"Extracting {len(val_entries)} validation patches …")
    save_patches(val_entries, OUTPUT_ROOT / "val", val_labels)

    print(f"Extracting {len(contrastive_entries)} contrastive patches …")
    save_patches(contrastive_entries, OUTPUT_ROOT / "contrastive_set", labels_dict=None)

    with open(OUTPUT_ROOT / "train_labels.json", "w") as f:
        json.dump(train_labels, f, indent=2)
    with open(OUTPUT_ROOT / "val_labels.json", "w") as f:
        json.dump(val_labels, f, indent=2)

    print("\n✓ Dataset created at:", OUTPUT_ROOT.resolve())
    print(f"  Train labels: {len(train_labels)} entries")
    print(f"  Val labels:   {len(val_labels)} entries")
    print(f"  Contrastive:  {len(contrastive_entries)} patches (no labels)")


if __name__ == "__main__":
    main()
