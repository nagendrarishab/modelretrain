"""
    python src/train_yolo_pose_detect.py --model yolo26n-pose.pt --epochs 50
"""
import argparse
import os
import shutil
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import torch
import yaml
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_detect/data.yaml",
                         help="box-detection dataset (from prepare_detect_dataset.py) to derive corner keypoints from")
    parser.add_argument("--pose-data-dir", default="data_pose",
                         help="where the derived pose-format dataset is (re)built")
    parser.add_argument("--model", default="yolo26n-pose.pt", choices=MODEL_CHOICES)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--scale", type=float, default=0.9,
                         help="same reasoning as train_yolo_detect.py - raw photos are mostly close-up on the box")
    parser.add_argument("--perspective", type=float, default=0.0005,
                         help="random perspective warp strength (Ultralytics 'perspective' aug)")
    parser.add_argument("--output", default=None, help="Output path. Defaults to models/<model_name>_best.pt")
    args = parser.parse_args()

    if args.output is None:
        model_basename = args.model.replace(".pt", "")
        args.output = f"models/{model_basename}_best.pt"

    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"No dataset at '{args.data}'. Run src/annotate_bboxes.py then "
            f"src/prepare_detect_dataset.py first."
        )

    pose_data_dir = Path(args.pose_data_dir)
    pose_yaml_path = build_pose_dataset(data_yaml_path, pose_data_dir)
    print(f"Derived pose dataset at {pose_data_dir} (4 box-corner keypoints per box)")

    device = str(get_device())
    print(f"Using device: {device}")

    model = YOLO(args.model)
    train_results = model.train(
        data=str(pose_yaml_path.resolve()),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=device,
        scale=args.scale,
        perspective=args.perspective,
        name="box_open_closed_yolo_pose",
        exist_ok=True,
    )

    save_dir = Path(getattr(train_results, "save_dir", None) or model.trainer.save_dir)
    best_ckpt = save_dir / "weights" / "best.pt"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_ckpt, args.output)
    print(f"Saved best model to {args.output} (full run artifacts in {save_dir})")

    best_model = YOLO(str(best_ckpt))
    metrics = best_model.val(data=str(pose_yaml_path.resolve()), split="test", imgsz=args.img_size, device=device)
    print(f"\nTest box mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    print(f"Test pose mAP50: {metrics.pose.map50:.4f}  mAP50-95: {metrics.pose.map:.4f}")
    print(f"Full report saved under: {metrics.save_dir}")


if __name__ == "__main__":
    main()
