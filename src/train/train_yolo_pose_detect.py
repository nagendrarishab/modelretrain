"""
    python src/train_yolo_pose_detect.py --model yolo26n-pose.pt --epochs 50

    # resume a run that got killed partway through (points at its last.pt)
    python src/train_yolo_pose_detect.py --model yolo26n-pose.pt --epochs 50 \
        --resume runs/pose/box_open_closed_yolo_pose/weights/last.pt

Same training-settings surface as train_yolo_detect.py - this is still a
plain Ultralytics YOLO().train()/.val() call, just on a pose model instead
of a detect model, so every BaseTrainer-level setting (augmentation,
patience/freeze/time/save-period/resume/cos-lr/fraction/min-samples-per-class)
applies identically. See train_yolo_detect.py's docstring/--help for the
per-setting rationale; not repeated here.
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


KPT_NAMES = ["top_left", "top_right", "bottom_right", "bottom_left"]
KPT_FLIP_IDX = [1, 0, 3, 2]

MODEL_CHOICES = [
    "yolo26n-pose.pt", "yolo26s-pose.pt", "yolo26m-pose.pt", "yolo26l-pose.pt", "yolo26x-pose.pt",
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def convert_label_line_to_pose(line):
    cls_token, cx_token, cy_token, w_token, h_token = line.split()[:5]
    cx, cy, w, h = float(cx_token), float(cy_token), float(w_token), float(h_token)
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # matches KPT_NAMES/KPT_FLIP_IDX order
    kpt_fields = " ".join(f"{x:.6f} {y:.6f} 2" for x, y in corners)
    return f"{cls_token} {cx_token} {cy_token} {w_token} {h_token} {kpt_fields}"


def build_pose_dataset(detect_yaml_path, pose_data_dir):
    detect_root = detect_yaml_path.parent
    class_names = yaml.safe_load(detect_yaml_path.read_text())["names"]

    for split in ("train", "val", "test"):
        src_images = detect_root / split / "images"
        src_labels = detect_root / split / "labels"
        dst_images = pose_data_dir / split / "images"
        dst_labels = pose_data_dir / split / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for label_path in src_labels.glob("*.txt"):
            pose_lines = [
                convert_label_line_to_pose(line)
                for line in label_path.read_text().splitlines() if line.strip()
            ]
            (dst_labels / label_path.name).write_text("\n".join(pose_lines))

        for image_path in src_images.iterdir():
            dst_image = dst_images / image_path.name
            if dst_image.exists() or dst_image.is_symlink():
                continue
            try:
                dst_image.symlink_to(image_path.resolve())
            except OSError:
                shutil.copy2(image_path, dst_image)

    pose_yaml_path = pose_data_dir / "data.yaml"
    pose_yaml_path.write_text(yaml.safe_dump({
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "kpt_shape": [4, 3],
        "flip_idx": KPT_FLIP_IDX,
        "names": class_names,
        "kpt_names": {cls_id: KPT_NAMES for cls_id in class_names},
    }, sort_keys=False))
    return pose_yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data_detect/data.yaml",
                         help="box-detection dataset (from prepare_detect_dataset.py) to derive corner keypoints from")
    parser.add_argument("--pose-data-dir", default="data_pose",
                         help="where the derived pose-format dataset lives; reused as-is if it already "
                              "has a data.yaml, otherwise (re)built from --data")
    parser.add_argument("--model", default="yolo26n-pose.pt", choices=MODEL_CHOICES)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--scale", type=float, default=0.9,
                         help="same reasoning as train_yolo_detect.py - raw photos are mostly close-up on the box")
    parser.add_argument("--perspective", type=float, default=0.0005,
                         help="random perspective warp strength (Ultralytics 'perspective' aug)")
    parser.add_argument("--hsv-h", type=float, default=0.015,
                         help="random hue jitter (Ultralytics 'hsv_h' aug) - see train_yolo_detect.py's --help")
    parser.add_argument("--hsv-s", type=float, default=0.7,
                         help="random saturation jitter (Ultralytics 'hsv_s' aug) - see train_yolo_detect.py's --help")
    parser.add_argument("--hsv-v", type=float, default=0.4,
                         help="random brightness/value jitter (Ultralytics 'hsv_v' aug) - see train_yolo_detect.py's --help")
    parser.add_argument("--mixup", type=float, default=0.0,
                         help="probability of blending two training images together (Ultralytics 'mixup' "
                              "aug); 0 disables it")
    parser.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=False,
                         help="randomly resize images between 0.5x-1.5x --img-size each batch. Off by default - "
                              "same MPS recompilation cost as train_yolo_detect.py's --multi-scale; "
                              "fine to enable on CUDA")
    parser.add_argument("--output", default=None, help="Output path. Defaults to models/<model_name>_best.pt")
    parser.add_argument("--min-samples-per-class", type=int, default=50,
                         help="classes with fewer annotations than this are dropped (and remaining classes "
                              "remapped) in the underlying box-detection dataset before deriving the pose "
                              "dataset from it")
    parser.add_argument("--cache", default="ram", choices=["ram", "disk", "none"],
                         help="cache decoded images after the first epoch to avoid re-reading/decoding "
                              "from disk every epoch; 'none' disables caching")
    parser.add_argument("--patience", type=int, default=0,
                         help="stop training early if val pose mAP doesn't improve for this many epochs; "
                              "0 disables early stopping and always runs the full --epochs")
    parser.add_argument("--freeze", type=int, default=0,
                         help="freeze (skip backprop through) the first N backbone layers; 0 trains every layer")
    parser.add_argument("--time", type=float, default=None,
                         help="max training time in hours; overrides --epochs once reached")
    parser.add_argument("--save-period", type=int, default=-1,
                         help="also save a numbered checkpoint (weights/epochN.pt) every N epochs; "
                              "-1 disables this")
    parser.add_argument("--resume", default=None,
                         help="path to a last.pt checkpoint to resume an interrupted run from "
                              "(model weights, optimizer state, and epoch count are restored)")
    parser.add_argument("--name", default="box_open_closed_yolo_pose",
                         help="run name; outputs go to runs/pose/<name>")
    parser.add_argument("--exist-ok", action="store_true",
                         help="overwrite an existing run directory with this --name instead of "
                              "keeping each run's args/weights/logs separate")
    parser.add_argument("--no-plots", action="store_true",
                         help="skip generating training curve/confusion-matrix plots at the end of the run")
    parser.add_argument("--fraction", type=float, default=1.0,
                         help="train on only this fraction of the dataset - use a small value "
                              "(e.g. 0.05) for a quick smoke test before committing to a full run")
    parser.add_argument("--cos-lr", action="store_true",
                         help="use a cosine learning-rate schedule instead of the default linear one")
    parser.add_argument("--tta", action="store_true",
                         help="use test-time augmentation for the final test-set eval")
    parser.add_argument("--conf", type=float, default=0.001,
                         help="confidence threshold for the final test-set eval")
    parser.add_argument("--iou", type=float, default=0.7,
                         help="NMS IoU threshold for the final test-set eval")
    args = parser.parse_args()
    args.cache = False if args.cache == "none" else args.cache

    if args.output is None:
        model_basename = args.model.replace(".pt", "")
        args.output = f"models/{model_basename}_best.pt"

    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"No dataset at '{args.data}'. Run src/annotate_bboxes.py then "
            f"src/prepare_detect_dataset.py first."
        )

    data_yaml_path = filter_low_sample_classes(data_yaml_path, args.min_samples_per_class)

    pose_data_dir = Path(args.pose_data_dir)
    existing_pose_yaml = pose_data_dir / "data.yaml"
    if existing_pose_yaml.exists():
        pose_yaml_path = existing_pose_yaml
        print(f"Reusing existing pose dataset at {pose_data_dir} (data.yaml already present) - "
              f"delete this directory first to rebuild it from --data")
    else:
        pose_yaml_path = build_pose_dataset(data_yaml_path, pose_data_dir)
        print(f"Derived pose dataset at {pose_data_dir} (4 box-corner keypoints per box)")

    device = str(get_device())
    print(f"Using device: {device}")

    model = YOLO(args.resume or args.model)
    train_kwargs = dict(
        data=str(pose_yaml_path.resolve()),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=device,
        scale=args.scale,
        perspective=args.perspective,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        mixup=args.mixup,
        multi_scale=args.multi_scale,
        cache=args.cache,
        patience=args.patience,
        freeze=args.freeze,
        time=args.time,
        save_period=args.save_period,
        plots=not args.no_plots,
        fraction=args.fraction,
        cos_lr=args.cos_lr,
        name=args.name,
        exist_ok=args.exist_ok,
    )
    if args.resume:
        train_kwargs["resume"] = True
    train_results = model.train(**train_kwargs)

    save_dir = Path(getattr(train_results, "save_dir", None) or model.trainer.save_dir)
    best_ckpt = save_dir / "weights" / "best.pt"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_ckpt, args.output)
    print(f"Saved best model to {args.output} (full run artifacts in {save_dir})")

    best_model = YOLO(str(best_ckpt))
    metrics = best_model.val(
        data=str(pose_yaml_path.resolve()),
        split="test",
        imgsz=args.img_size,
        device=device,
        augment=args.tta,
        conf=args.conf,
        iou=args.iou,
        plots=True,  # needed for metrics.confusion_matrix.matrix to actually be populated
    )
    print(f"\nTest box mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    print(f"Test pose mAP50: {metrics.pose.map50:.4f}  mAP50-95: {metrics.pose.map:.4f}")
    print(f"Full report saved under: {metrics.save_dir}")


if __name__ == "__main__":
    main()
