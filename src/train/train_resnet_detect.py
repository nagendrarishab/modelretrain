"""
    python src/train_resnet_detect.py --backbone resnet50 --epochs 30
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
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms.functional import to_tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

BACKBONE_WEIGHTS = {
    "resnet18": "IMAGENET1K_V1",
    "resnet34": "IMAGENET1K_V1",
    "resnet50": "IMAGENET1K_V2",
    "resnet101": "IMAGENET1K_V2",
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
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


def offload_to_device(module, device):
    module.to(device)
    original_forward = module.forward

    def forward(x):
        output = original_forward(x.to(device))
        return {key: value.to("cpu") for key, value in output.items()}

    module.forward = forward
    return module


def build_model(backbone_name, num_classes, backbone_device=None,
                min_size=800, max_size=1333):
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=BACKBONE_WEIGHTS[backbone_name],
        trainable_layers=3,
    )
    if backbone_device is not None:
        offload_to_device(backbone, backbone_device)
    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
    )


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_detect/data.yaml")
    parser.add_argument("--backbone", default="resnet50",
                         choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--min-size", type=int, default=320,
                        help="Shorter image side after resize")
    parser.add_argument("--max-size", type=int, default=569,
                        help="Longer image side cap after resize")
    parser.add_argument("--output", default=None, help="Output path. Defaults to models/resnet_<backbone>_detect_best.pt")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"models/resnet_{args.backbone}_detect_best.pt"

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
    backbone_device = None
    if device.type == "cpu" and torch.backends.mps.is_available():
        backbone_device = torch.device("mps")
        print(f"Using device: {device} (backbone offloaded to {backbone_device}, "
              "RPN/ROI heads stay on CPU)")
    else:
        print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

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

    model = build_model(
        args.backbone,
        num_classes,
        backbone_device=backbone_device,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    if backbone_device is None:
        model = model.to(device)
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
                    f"This can result from Faster R-CNN/MPS instability; rerun with "
                    f"--workers 0 or on CUDA if it recurs."
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
                        "backbone": args.backbone,
                        "class_names": class_names}, args.output)

    print(f"\nSaved best model to {args.output} (val mAP50={best_map50:.4f})")

    best = torch.load(args.output, map_location=device, weights_only=False)
    model = build_model(
        best["backbone"],
        len(best["class_names"]) + 1,
        backbone_device=backbone_device,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    if backbone_device is None:
        model = model.to(device)
    model.load_state_dict(best["model_state_dict"])
    test_map50, test_map5095 = evaluate(model, test_loader, device)
    print(f"Test mAP50: {test_map50:.4f}  mAP50-95: {test_map5095:.4f}")


if __name__ == "__main__":
    main()
