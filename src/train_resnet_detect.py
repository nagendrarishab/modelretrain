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
    # MPS previously made Faster R-CNN's ROIAlign/NMS ops numerically unstable
    # (loss exploded to NaN within a few batches) - re-enabled per request, but
    # the training loop below aborts fast on a non-finite loss so a recurrence
    # is caught in the first few batches instead of burning hours on garbage.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class YoloFormatDataset(Dataset):
    """Reads the data_detect/{split}/{images,labels} layout that
    prepare_detect_dataset.py produces (YOLO-format labels: `class cx cy w h`,
    normalized 0-1) and converts it to the [x1, y1, x2, y2] absolute-pixel
    boxes + 1-indexed labels torchvision detection models expect (label 0 is
    reserved for background)."""

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
    """Linearly ramps the LR up from warmup_factor*base_lr to base_lr over the
    first warmup_iters steps. Faster R-CNN's loss reliably explodes to NaN
    within the first few batches without this - the untrained RPN/ROI heads
    produce huge early gradients that a full-strength LR overshoots on."""
    def f(step):
        if step >= warmup_iters:
            return 1.0
        alpha = step / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def build_model(backbone_name, num_classes):
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=BACKBONE_WEIGHTS[backbone_name],
        trainable_layers=3,
    )
    return FasterRCNN(backbone, num_classes=num_classes)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
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
    print(f"Using device: {device}")

    train_loader = DataLoader(
        YoloFormatDataset(data_root / "train"), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        YoloFormatDataset(data_root / "val"), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        YoloFormatDataset(data_root / "test"), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )
    print(f"Train: {len(train_loader.dataset)} images ({len(train_loader)} batches/epoch)  "
          f"Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")

    model = build_model(args.backbone, num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.1)

    best_map50 = -1.0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

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
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

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
            loss.backward()
            optimizer.step()
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
    model = build_model(best["backbone"], len(best["class_names"]) + 1).to(device)
    model.load_state_dict(best["model_state_dict"])
    test_map50, test_map5095 = evaluate(model, test_loader, device)
    print(f"Test mAP50: {test_map50:.4f}  mAP50-95: {test_map5095:.4f}")


if __name__ == "__main__":
    main()
