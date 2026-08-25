"""
    python src/train_yolo_detect.py --model yolo26n.pt --epochs 50
"""
import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import certifi
import yaml

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import torch
from ultralytics import YOLO


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


MODEL_CHOICES = [
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
]


def analyze_class_distribution(data_dir, names):
    """Count annotations per class across train/val/test and report imbalance."""
    class_counts = Counter()
    for split in ("train", "val", "test"):
        labels_dir = data_dir / split / "labels"
        if not labels_dir.exists():
            continue
        for label_file in labels_dir.glob("*.txt"):
            for line in label_file.read_text().splitlines():
                parts = line.split()
                if parts:
                    class_counts[int(parts[0])] += 1

    total = sum(class_counts.values())
    print("\nClass distribution:")
    for cls_id in sorted(names):
        count = class_counts.get(cls_id, 0)
        pct = (count / total * 100) if total else 0
        print(f"  class {cls_id} ({names[cls_id]:12s}): {count:6d} annotations ({pct:5.2f}%)")

    valid_counts = [c for c in class_counts.values() if c > 0]
    if len(valid_counts) >= 2:
        print(f"Class imbalance ratio: {max(valid_counts) / min(valid_counts):.2f}x")

    return class_counts


def filter_low_sample_classes(data_yaml_path, min_samples_per_class):
    """Drop classes with too few annotations and remap the remaining ones,
    writing a filtered copy of the dataset alongside the original. Data split
    (train/val/test) and per-image content are otherwise left untouched.
    Returns the path to the data.yaml to train on (original if nothing needed excluding).
    """
    data_yaml_path = Path(data_yaml_path)
    data_dir = data_yaml_path.parent
    cfg = yaml.safe_load(data_yaml_path.read_text())
    names = cfg["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))

    class_counts = analyze_class_distribution(data_dir, names)
    keep_ids = sorted(cls_id for cls_id in names if class_counts.get(cls_id, 0) >= min_samples_per_class)

    if len(keep_ids) == len(names):
        print("All classes meet --min-samples-per-class - training on the full dataset.")
        return data_yaml_path

    excluded = [names[i] for i in sorted(names) if i not in keep_ids]
    if not keep_ids:
        raise RuntimeError(
            f"No class has >= {min_samples_per_class} annotations; lower --min-samples-per-class or add data."
        )
    print(f"Excluding class(es) below threshold ({min_samples_per_class}): {excluded}")

    id_remap = {old: new for new, old in enumerate(keep_ids)}
    filtered_dir = data_dir.parent / f"{data_dir.name}_filtered"
    if filtered_dir.exists():
        shutil.rmtree(filtered_dir)

    for split in ("train", "val", "test"):
        images_dir = data_dir / split / "images"
        labels_dir = data_dir / split / "labels"
        if not labels_dir.exists():
            continue
        out_images = filtered_dir / split / "images"
        out_labels = filtered_dir / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        for label_file in labels_dir.glob("*.txt"):
            kept_lines = []
            for line in label_file.read_text().splitlines():
                parts = line.split()
                if parts and int(parts[0]) in id_remap:
                    kept_lines.append(" ".join([str(id_remap[int(parts[0])])] + parts[1:]))
            (out_labels / label_file.name).write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))

            for ext in (".jpg", ".jpeg", ".png"):
                src_img = images_dir / f"{label_file.stem}{ext}"
                if src_img.exists():
                    os.symlink(src_img.resolve(), out_images / src_img.name)
                    break

    (filtered_dir / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\n"
        "names:\n" + "".join(f"  {new_id}: {names[old_id]}\n" for old_id, new_id in id_remap.items())
    )
    print(f"Wrote filtered dataset to {filtered_dir} ({len(keep_ids)}/{len(names)} classes kept)")
    return filtered_dir / "data.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_detect/data.yaml")
    parser.add_argument("--model", default="yolo26n.pt", choices=MODEL_CHOICES)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--scale", type=float, default=0.9,
                         help="random zoom in/out range (Ultralytics 'scale' aug) - higher means more "
                              "aggressive zoom-out, i.e. more training views of the box looking small/far away")
    parser.add_argument("--perspective", type=float, default=0.0005,
                         help="random perspective warp strength (Ultralytics 'perspective' aug)")
    parser.add_argument("--output", default=None, help="Output path. Defaults to models/<model_name>_best.pt")
    parser.add_argument("--min-samples-per-class", type=int, default=50,
                         help="classes with fewer annotations than this are dropped (and remaining "
                              "classes remapped) before training; data split and other settings are unaffected")
    parser.add_argument("--cache", default="ram", choices=["ram", "disk", "none"],
                         help="cache decoded images after the first epoch to avoid re-reading/decoding "
                              "from disk every epoch (keeps the GPU fed); 'none' disables caching")
    parser.add_argument("--patience", type=int, default=0,
                         help="stop training early if val mAP doesn't improve for this many epochs; "
                              "0 disables early stopping and always runs the full --epochs")
    parser.add_argument("--freeze", type=int, default=0,
                         help="freeze (skip backprop through) the first N backbone layers, so only the "
                              "later layers/head are fine-tuned - faster per epoch, useful when the "
                              "pretrained backbone already extracts good enough features for this task; "
                              "0 trains every layer")
    args = parser.parse_args()
    args.cache = False if args.cache == "none" else args.cache

    if args.output is None:
        model_basename = args.model.replace(".pt", "")
        args.output = f"models/{model_basename}_best.pt"

    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"No dataset at '{args.data}'. Run src/annotate_bboxes.py then "
            f"src/prepare_detect_dataset.py first."
        )

    args.data = str(filter_low_sample_classes(args.data, args.min_samples_per_class))

    device = str(get_device())
    print(f"Using device: {device}")

    model = YOLO(args.model)
    train_results = model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=device,
        scale=args.scale,
        perspective=args.perspective,
        cache=args.cache,
        patience=args.patience,
        freeze=args.freeze,
        name="box_open_closed_yolo_detect",
        exist_ok=True,
    )

    save_dir = Path(getattr(train_results, "save_dir", None) or model.trainer.save_dir)
    best_ckpt = save_dir / "weights" / "best.pt"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_ckpt, args.output)
    print(f"Saved best model to {args.output} (full run artifacts in {save_dir})")

    best_model = YOLO(str(best_ckpt))
    metrics = best_model.val(data=str(Path(args.data).resolve()), split="test", imgsz=args.img_size, device=device)
    print(f"\nTest mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    print(f"Full report saved under: {metrics.save_dir}")


if __name__ == "__main__":
    main()
