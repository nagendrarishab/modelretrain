"""
    python src/prepare_detect_dataset.py
"""
import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CLASSES = ["closed", "open"]  # must match annotate_bboxes.py's class-id order


def average_hash(path, hash_size=8):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.asarray(img, dtype=np.float64)
    return (pixels > pixels.mean()).flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def cluster_scenes(paths, threshold):
    hashes = [average_hash(p) for p in paths]
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if hamming(hashes[i], hashes[j]) <= threshold:
                union(i, j)

    return [find(i) for i in range(len(paths))]


def assign_groups_to_splits(group_ids, group_sizes, val_frac, test_frac, rng):
    n_total = sum(group_sizes.values())
    targets = {
        "val": round(n_total * val_frac),
        "test": round(n_total * test_frac),
    }
    targets["train"] = n_total - targets["val"] - targets["test"]

    order = list(group_ids)
    rng.shuffle(order)
    order.sort(key=lambda g: -group_sizes[g])

    counts = {"train": 0, "val": 0, "test": 0}
    assignment = {}
    for g in order:
        if counts["train"] < targets["train"]:
            split = "train"
        else:
            split = max(("val", "test"), key=lambda s: targets[s] - counts[s])
        assignment[g] = split
        counts[split] += group_sizes[g]

    return assignment, counts


def save_exif_corrected(src, dst):
    img = Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.save(dst, quality=95)


def prepare(raw_dir, labels_dir, out_dir, val_frac, test_frac, seed, dup_threshold, background_dir=None):
    import random
    rng = random.Random(seed)
    raw_dir, labels_dir, out_dir = Path(raw_dir), Path(labels_dir), Path(out_dir)
    background_dir = Path(background_dir) if background_dir else raw_dir / "background"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {cls: {"train": 0, "val": 0, "test": 0} for cls in CLASSES + ["extra"]}
    skipped_unlabeled = 0

    labeled = []  # (path, cls)
    for cls in CLASSES + ["extra"]:
        all_files = sorted((raw_dir / cls).glob("*.jpg")) + sorted((raw_dir / cls).glob("*.jpeg"))
        files = [f for f in all_files if (labels_dir / cls / (f.stem + ".txt")).exists()]
        skipped_unlabeled += len(all_files) - len(files)
        labeled.extend((f, cls) for f in files)

    if labeled:
        group_ids = cluster_scenes([f for f, _ in labeled], dup_threshold)
        group_to_items = defaultdict(list)
        for (f, cls), g in zip(labeled, group_ids):
            group_to_items[g].append((f, cls))
        group_sizes = {g: len(items) for g, items in group_to_items.items()}

        assignment, _ = assign_groups_to_splits(
            list(group_to_items.keys()), group_sizes, val_frac, test_frac, rng)

        for g, split in assignment.items():
            for f, cls in group_to_items[g]:
                counts[cls][split] += 1
                save_exif_corrected(f, out_dir / split / "images" / f.name)
                shutil.copy2(labels_dir / cls / (f.stem + ".txt"), out_dir / split / "labels" / (f.stem + ".txt"))

    bg_files = (sorted(background_dir.glob("*.jpg")) + sorted(background_dir.glob("*.jpeg"))
                if background_dir.exists() else [])
    if bg_files:
        group_ids = cluster_scenes(bg_files, dup_threshold)
        group_to_files = defaultdict(list)
        for f, g in zip(bg_files, group_ids):
            group_to_files[g].append(f)
        group_sizes = {g: len(fs) for g, fs in group_to_files.items()}

        assignment, split_counts = assign_groups_to_splits(
            list(group_to_files.keys()), group_sizes, val_frac, test_frac, rng)
        counts["background"] = split_counts

        for g, split in assignment.items():
            for f in group_to_files[g]:
                save_exif_corrected(f, out_dir / split / "images" / f.name)
                (out_dir / split / "labels" / (f.stem + ".txt")).write_text("")  # zero boxes = negative example
    else:
        counts["background"] = {"train": 0, "val": 0, "test": 0}

    print(f"Split counts (scene-group-aware, dedup across closed/open/extra combined, "
          f"dup-threshold={dup_threshold}):")
    for cls, c in counts.items():
        print(f"  {cls}: {c}")
    if skipped_unlabeled:
        print(f"\nSkipped {skipped_unlabeled} raw images with no label yet - "
              f"run src/annotate_bboxes.py to cover them.")
    if not bg_files:
        print(f"\nNo background (no-box) images found in {background_dir} - the model has never seen "
              f"'nothing here', so it may false-positive on non-box content. Consider adding some.")

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
    )
    print(f"\nWrote {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="raw")
    parser.add_argument("--labels-dir", default="raw_labels")
    parser.add_argument("--background-dir", default=None,
                         help="folder of no-box photos to include as negative examples (default: <raw-dir>/background)")
    parser.add_argument("--out-dir", default="data_detect")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dup-threshold", type=int, default=10,
                         help="Hamming distance (out of 64) below which two raw photos are the same scene")
    args = parser.parse_args()

    prepare(args.raw_dir, args.labels_dir, args.out_dir, args.val_frac, args.test_frac, args.seed,
            args.dup_threshold, args.background_dir)
