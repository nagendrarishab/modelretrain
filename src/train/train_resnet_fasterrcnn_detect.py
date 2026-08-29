"""
    python src/train/train_resnet_fasterrcnn_detect.py --epochs 30

Faster R-CNN on a ResNet50 backbone, using torchvision's own ready-made
`fasterrcnn_resnet50_fpn`/`fasterrcnn_resnet50_fpn_v2` constructors instead
of hand-building a backbone+FPN wrapper (unlike train_efficientnet_detect.py
used to, and train_mobilenetv4_detect.py still does) - ResNet is one of the
only two backbones (the other being MobileNetV3) torchvision ships a
ready-made Faster R-CNN constructor for; every other backbone in this repo
needs the hand-built approach because no such constructor exists for it.

Both constructors default to `weights=None` (no COCO-pretrained detection
head to discard) - matching this repo's ImageNet-only convention already,
with zero custom code needed to get there.
"""
import argparse
import os
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import torch
import yaml
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

VARIANT_CHOICES = ["resnet50_fpn", "resnet50_fpn_v2"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class YoloFormatDataset(Dataset):

    def __init__(self, split_dir):
        self.images_dir = split_dir / "images"
        self.labels_dir = split_dir / "labels"
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

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
        return to_tensor(img), target


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
    if variant == "resnet50_fpn_v2":
        # v2's weights_backbone defaults to None (unlike v1's IMAGENET1K_V1
        # default) - pass it explicitly or this silently trains a
        # random-init backbone, which would quietly contradict this
        # project's "don't reduce accuracy" priority. Confirmed this
        # asymmetry directly via inspect.signature(), not assumed.
        return fasterrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V1, num_classes=num_classes,
        )
    return fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)


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
    parser.add_argument("--variant", default="resnet50_fpn", choices=VARIANT_CHOICES,
                         help="resnet50_fpn (v1, ImageNet backbone by default) or resnet50_fpn_v2 "
                              "(improved RPN head/box predictor, same ImageNet backbone)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--output", default="models/resnet_fasterrcnn_detect_best.pt")
    args = parser.parse_args()

    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"No dataset at '{args.data}'. Run src/annotate_bboxes.py then "
            f"src/prepare_detect_dataset.py first."
        )
    data_root = data_yaml_path.parent
    class_names = yaml.safe_load(data_yaml_path.read_text())["names"]
    num_classes = len(class_names) + 1  # +1 for background

    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    loader_kwargs = dict(
        collate_fn=collate_fn,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(
        YoloFormatDataset(data_root / "train"), batch_size=args.batch_size,
        shuffle=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        YoloFormatDataset(data_root / "val"), batch_size=args.batch_size,
        shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        YoloFormatDataset(data_root / "test"), batch_size=args.batch_size,
        shuffle=False, **loader_kwargs,
    )
    print(f"Train: {len(train_loader.dataset)} images ({len(train_loader)} batches/epoch)  "
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
                    f"Loss went non-finite ({breakdown}) on device={device}. "
                    f"This is the known Faster R-CNN/MPS instability - rerun with a CPU-only "
                    f"get_device() (or on CUDA) if it recurs."
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
        map50, map5095 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs}  loss={epoch_loss / len(train_loader):.4f}  "
              f"val mAP50={map50:.4f}  val mAP50-95={map5095:.4f}")

        if map50 > best_map50:
            best_map50 = map50
            torch.save({"model_state_dict": model.state_dict(),
                        "family": "resnet-fasterrcnn",
                        "variant": args.variant,
                        "class_names": class_names}, args.output)

    print(f"\nSaved best model to {args.output} (val mAP50={best_map50:.4f})")

    best = torch.load(args.output, map_location=device, weights_only=False)
    model = build_model(len(best["class_names"]) + 1, best["variant"]).to(device)
    model.load_state_dict(best["model_state_dict"])
    test_map50, test_map5095 = evaluate(model, test_loader, device)
    print(f"Test mAP50: {test_map50:.4f}  mAP50-95: {test_map5095:.4f}")


if __name__ == "__main__":
    main()
