"""
    python src/train/train_mobilenetv3_fasterrcnn_detect.py --epochs 30

    # GPU/MPS run (default) - only one process gets the GPU at a time, so use the largest
    # --batch-size that fits instead of splitting it across concurrent GPU runs:
    ./.venv/bin/python src/train/train_mobilenetv3_fasterrcnn_detect.py --epochs 2 --batch-size 16

    # CPU run - pair with a GPU run of another backbone for real parallelism (see --device):
    ./.venv/bin/python src/train/train_mobilenetv3_fasterrcnn_detect.py --epochs 2 --batch-size 4 --device cpu
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import numpy as np
import torch
import yaml
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.transforms.functional import to_tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

VARIANT_CHOICES = ["large_fpn", "large_320_fpn"]


def make_emitter(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"train_mobilenetv3_fasterrcnn_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = open(log_path, "w")

    def emit(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    return emit, log_path


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class YoloFormatDataset(Dataset):
    """--cache mirrors train_yolo_detect.py's --cache and the train_*_torch_detect.py
    RetinaNet ports' --cache: avoids re-decoding the same JPEG every epoch. Unlike those
    RetinaNet ports, Faster R-CNN's own GeneralizedRCNNTransform resizes internally (no
    fixed height/width here), so what's cached is the image at its native decoded
    resolution. 'disk' writes it once to a <stem>.npy file under --cache-dir (survives
    across runs, and across --workers > 0's separate worker processes, since they share
    the filesystem - writes go through a temp file + atomic rename so concurrent workers
    racing to cache the same image can't corrupt it). 'ram' keeps a dict in this process
    only - with --workers > 0 each worker process gets its own private dict, and shuffled
    sampling means a given worker rarely sees the same image twice, so it's a much weaker
    cache here than in the (worker-less) RetinaNet ports; 'disk' is the one that matters.
    'none' decodes fresh every access."""

    def __init__(self, split_dir, cache="disk", cache_dir=None):
        self.images_dir = split_dir / "images"
        self.labels_dir = split_dir / "labels"
        self.cache = cache
        self.cache_dir = Path(cache_dir) / split_dir.name if cache_dir else None
        if self.cache == "disk":
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ram_cache = {} if cache == "ram" else None
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )

    def __len__(self):
        return len(self.image_paths)

    def _load_image_array(self, image_path):
        if self.ram_cache is not None and image_path in self.ram_cache:
            return self.ram_cache[image_path]

        npy_path = None
        if self.cache == "disk":
            npy_path = self.cache_dir / f"{image_path.stem}.npy"
            if npy_path.exists():
                arr = np.load(npy_path)
                if self.ram_cache is not None:
                    self.ram_cache[image_path] = arr
                return arr

        arr = np.array(Image.open(image_path).convert("RGB"))  # np.array (not np.asarray) copies -> writable

        if npy_path is not None:
            tmp_path = npy_path.with_suffix(f".tmp-{os.getpid()}.npy")
            np.save(tmp_path, arr)
            os.replace(tmp_path, npy_path)  # atomic - safe if another --workers process races us here
        if self.ram_cache is not None:
            self.ram_cache[image_path] = arr
        return arr

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        arr = self._load_image_array(image_path)
        height, width = arr.shape[:2]

        boxes, labels = [], []
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                cls_id, cx, cy, bw, bh = map(float, line.split())
                x1 = (cx - bw / 2) * width
                y1 = (cy - bh / 2) * height
                x2 = (cx + bw / 2) * width
                y2 = (cy + bh / 2) * height
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls_id) + 1)  # +1: 0 is background

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return to_tensor(arr), target


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(step):
        if step >= warmup_iters:
            return 1.0
        alpha = step / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def build_model(num_classes, variant):
    ctor = fasterrcnn_mobilenet_v3_large_320_fpn if variant == "large_320_fpn" else fasterrcnn_mobilenet_v3_large_fpn
    return ctor(weights=None, num_classes=num_classes)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    use_amp = device.type == "cuda"
    for images, targets in data_loader:
        images = [img.to(device, non_blocking=True) for img in images]
        with autocast(device_type=device.type, enabled=use_amp):
            preds = model(images)
        preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
        metric.update(preds, targets)
    result = metric.compute()
    return float(result["map_50"]), float(result["map"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data_detect/data.yaml")
    parser.add_argument("--variant", default="large_fpn", choices=VARIANT_CHOICES,
                         help="large_fpn (higher accuracy, default input resolution) or large_320_fpn "
                              "(lower internal resolution, faster) - both ImageNet-backbone by default")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--output", default="models/mobilenetv3_detect_best.pt")
    parser.add_argument("--log-dir", default="logs", help="Directory for this run's log file")
    parser.add_argument("--cache", default="disk", choices=["disk", "ram", "none"],
                         help="cache decoded images to avoid re-reading+re-decoding the same JPEG every "
                              "epoch (same idea as train_yolo_detect.py's --cache). 'disk' (default) writes "
                              "each image once under --cache-dir and reuses it on later epochs *and* later "
                              "runs. 'ram' skips the disk write but only lasts this process. 'none' decodes "
                              "fresh every access.")
    parser.add_argument("--cache-dir", default="cache/detect_ds_fasterrcnn_torch",
                         help="Directory for --cache disk files. Shared by default with the other "
                              "train_*_fasterrcnn_detect.py scripts, since the cached array is just the "
                              "decoded image at native resolution - identical regardless of backbone. "
                              "Delete it if the dataset changes.")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu"],
                         help="'cpu' forces CPU-only training (e.g. to run alongside another script's "
                              "GPU/MPS run, or work around a GPU-specific issue). 'auto' (default) trains "
                              "on GPU if one is visible (CUDA, or Apple Silicon via MPS), else CPU")
    args = parser.parse_args()

    emit, log_path = make_emitter(args.log_dir)
    emit(f"Logging to {log_path}")

    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"No dataset at '{args.data}'. Run src/annotate_bboxes.py then "
            f"src/prepare_detect_dataset.py first."
        )
    data_root = data_yaml_path.parent
    class_names = yaml.safe_load(data_yaml_path.read_text())["names"]
    num_classes = len(class_names) + 1  # +1 for background

    device = torch.device("cpu") if args.device == "cpu" else get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    emit(f"Using device: {device}")

    loader_kwargs = dict(
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    dataset_kwargs = dict(cache=args.cache, cache_dir=args.cache_dir if args.cache == "disk" else None)
    train_loader = DataLoader(
        YoloFormatDataset(data_root / "train", **dataset_kwargs), batch_size=args.batch_size,
        shuffle=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        YoloFormatDataset(data_root / "val", **dataset_kwargs), batch_size=args.batch_size,
        shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        YoloFormatDataset(data_root / "test", **dataset_kwargs), batch_size=args.batch_size,
        shuffle=False, **loader_kwargs,
    )
    emit(f"Train: {len(train_loader.dataset)} images ({len(train_loader)} batches/epoch)  "
         f"Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")

    model = build_model(num_classes, args.variant).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.1)

    best_map50 = -1.0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        epoch_warmup = None
        if epoch == 1:
            warmup_iters = min(1000, len(train_loader) - 1)
            if warmup_iters > 0:
                epoch_warmup = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor=1.0 / 1000)

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for images, targets in progress:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            with autocast(device_type=device.type, enabled=use_amp):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                breakdown = ", ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
                raise RuntimeError(
                    f"Loss went non-finite ({breakdown}) on device={device}."
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if epoch_warmup is not None:
                epoch_warmup.step()
            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        if device.type == "mps":
            # MPS's caching allocator doesn't release memory between epochs the way CUDA's
            # does, so long runs slowly accumulate until the OS OOM-kills the process
            torch.mps.empty_cache()
        map50, map5095 = evaluate(model, val_loader, device)
        emit(f"Epoch {epoch}/{args.epochs}  loss={epoch_loss / len(train_loader):.4f}  "
             f"val mAP50={map50:.4f}  val mAP50-95={map5095:.4f}")

        if map50 > best_map50:
            best_map50 = map50
            torch.save({"model_state_dict": model.state_dict(),
                        "family": "mobilenetv3-fasterrcnn",
                        "variant": args.variant,
                        "class_names": class_names}, args.output)

    emit(f"\nSaved best model to {args.output} (val mAP50={best_map50:.4f})")

    best = torch.load(args.output, map_location=device, weights_only=False)
    model = build_model(len(best["class_names"]) + 1, best["variant"]).to(device)
    model.load_state_dict(best["model_state_dict"])
    test_map50, test_map5095 = evaluate(model, test_loader, device)
    emit(f"Test mAP50: {test_map50:.4f}  mAP50-95: {test_map5095:.4f}")


if __name__ == "__main__":
    main()
