"""
    python src/train_yolo_detect.py --model yolo26n.pt --epochs 50
"""
import argparse
import os
import shutil
from pathlib import Path

import certifi

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_detect/data.yaml")
    parser.add_argument("--model", default="yolo26n.pt", choices=MODEL_CHOICES)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--scale", type=float, default=0.9,
                         help="random zoom in/out range (Ultralytics 'scale' aug) - higher means more "
                              "aggressive zoom-out, i.e. more training views of the box looking small/far away")
    parser.add_argument("--perspective", type=float, default=0.0005,
                         help="random perspective warp strength (Ultralytics 'perspective' aug)")
    parser.add_argument("--output", default=None, help="Output path. Defaults to models/<model_name>_best.pt")
    args = parser.parse_args()

    if args.output is None:
        model_basename = args.model.replace(".pt", "")
        args.output = f"models/{model_basename}_best.pt"

    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"No dataset at '{args.data}'. Run src/annotate_bboxes.py then "
            f"src/prepare_detect_dataset.py first."
        )

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
