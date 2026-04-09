from collections import defaultdict
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import shape


TASK_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = TASK_ROOT / "Dataset_Splits"
OUTPUT_ROOT = TASK_ROOT / "task2_nuclei_patches"

PATCH_HALF = 50          
TRAIN_PER_CLASS = 2500
VAL_PER_CLASS = 700
CONTRASTIVE_PER_CLASS = 1000

RAW_TO_TARGET_CLASS = {
    "nuclei_tumor": "tumor",
    "nuclei_lymphocyte": "lymphocyte",
    "nuclei_histiocyte": "histiocyte",
}
TARGET_CLASS_NAMES = ("tumor", "lymphocyte", "histiocyte")

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
    Zeropad if the bounding box exceeds image boundaries.
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
            if cls not in RAW_TO_TARGET_CLASS:
                continue
            class_entries[RAW_TO_TARGET_CLASS[cls]][stem].append((str(img_path), cx, cy))

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

    print("Collecting nuclei from TRAIN and VALIDATION splits …")
    train_by_img = collect_nuclei_by_image("train")
    val_by_img = collect_nuclei_by_image("validation")

    # 1. Identify ALL unique train image stems and validation image stems globally
    train_unique_stems = set()
    for cls in TARGET_CLASS_NAMES:
        for stem in train_by_img.get(cls, {}).keys():
            train_unique_stems.add(stem)
            
    val_unique_stems = set()
    for cls in TARGET_CLASS_NAMES:
        for stem in val_by_img.get(cls, {}).keys():
            val_unique_stems.add(stem)

    # 2. Perform a GLOBAL split on the train stems into train vs contrastive (no leakage!)
    train_stems_list = sorted(list(train_unique_stems))
    val_stems = sorted(list(val_unique_stems))
    
    random.shuffle(train_stems_list)
    n_train = max(1, int(len(train_stems_list) * 0.7))
    train_stems = train_stems_list[:n_train]
    contr_stems = train_stems_list[n_train:]

    print(f"\n── Global Split Stats ──")
    print(f"  Training images:    {len(train_stems)} (from {DATASET_ROOT / 'train'})")
    print(f"  Contrastive images: {len(contr_stems)} (from {DATASET_ROOT / 'train'})")
    print(f"  Validation images:  {len(val_stems)} (from {DATASET_ROOT / 'validation'})")

    train_entries = []
    val_entries = []
    contrastive_entries = []

    # 3. For each class, sample only from the images assigned to that split
    for cls in TARGET_CLASS_NAMES:
        train_class_data = train_by_img.get(cls, {})
        val_class_data = val_by_img.get(cls, {})
        
        # Filter the available nuclei by the global split
        t_pool = [e for s in train_stems if s in train_class_data for e in train_class_data[s]]
        c_pool = [e for s in contr_stems if s in train_class_data for e in train_class_data[s]]
        v_pool = [e for s in val_stems if s in val_class_data for e in val_class_data[s]]

        print(f"\n  Processing {cls}:")
        print(f"    Available: {len(t_pool)} train, {len(v_pool)} val, {len(c_pool)} contr")

        # Use a helper to sample or replace if pool is too small
        def get_samples(pool, count, name):
            if not pool:
                print(f"    [ERROR] No nuclei for {cls} in {name} split!")
                return []
            if len(pool) < count:
                print(f"    [WARN] Only {len(pool)} nuclei for {cls} in {name}. Sampling with replacement.")
                return random.choices(pool, k=count)
            return random.sample(pool, k=count)

        t_sel = get_samples(t_pool, TRAIN_PER_CLASS, "train")
        c_sel = get_samples(c_pool, CONTRASTIVE_PER_CLASS, "contrastive")
        v_sel = get_samples(v_pool, VAL_PER_CLASS, "validation")

        train_entries.extend([(p, x, y, cls) for p, x, y in t_sel])
        contrastive_entries.extend([(p, x, y, cls) for p, x, y in c_sel])
        val_entries.extend([(p, x, y, cls) for p, x, y in v_sel])

    # 4. Save and export labels
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

    print("\n Dataset created successfully")

if __name__ == "__main__":
    main()
