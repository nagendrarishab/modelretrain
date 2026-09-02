"""
    python src/train/train_densenet_torch_detect.py --backbone densenet121 --epochs 30

    # GPU/MPS run (default) - only one process gets the GPU at a time, so use the largest
    # --batch-size that fits instead of splitting it across concurrent GPU runs:
    ./.venv/bin/python src/train/train_densenet_torch_detect.py --epochs 2 --batch-size 16

    # CPU run - pair with a GPU run of another backbone for real parallelism (see --device):
    ./.venv/bin/python src/train/train_densenet_torch_detect.py --epochs 2 --batch-size 4 --device cpu
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())  # needed to download pretrained weights

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    DenseNet121_Weights, DenseNet169_Weights, DenseNet201_Weights,
    densenet121, densenet169, densenet201,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork
from torchvision.transforms.functional import to_tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

BACKBONE_BUILDERS = {
    "densenet121": (densenet121, DenseNet121_Weights.IMAGENET1K_V1),
    "densenet169": (densenet169, DenseNet169_Weights.IMAGENET1K_V1),
    "densenet201": (densenet201, DenseNet201_Weights.IMAGENET1K_V1),
}


def make_emitter(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"train_densenet_torch_{datetime.now():%Y%m%d_%H%M%S}.log"
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
    """Same on-disk layout as the TF scripts' load_split() (images/ + YOLO-format labels/),
    but resizes to a fixed (height, width) up front so it matches the no-op resize
    build_model() sets up via RetinaNet's min_size/max_size transform.

    --cache mirrors train_yolo_detect.py's --cache: avoids re-decoding+resizing the same
    JPEG every epoch. 'disk' writes the decoded/resized array to a <stem>_HxW.npy file
    (same idea as Ultralytics' own disk cache) so it also survives across separate runs;
    'ram' keeps it in a dict in this process only (nothing written to disk, but lost when
    the process exits); 'none' matches the old behavior - decode fresh every access."""

    def __init__(self, split_dir, height, width, cache="disk", cache_dir=None):
        self.images_dir = split_dir / "images"
        self.labels_dir = split_dir / "labels"
        self.height = height
        self.width = width
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
            npy_path = self.cache_dir / f"{image_path.stem}_{self.height}x{self.width}.npy"
            if npy_path.exists():
                arr = np.load(npy_path)
                if self.ram_cache is not None:
                    self.ram_cache[image_path] = arr
                return arr

        img = Image.open(image_path).convert("RGB").resize((self.width, self.height), Image.BILINEAR)
        arr = np.array(img)  # (not np.asarray) - copies, so the array owns a writable buffer

        if npy_path is not None:
            np.save(npy_path, arr)
        if self.ram_cache is not None:
            self.ram_cache[image_path] = arr
        return arr

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = self._load_image_array(image_path)

        boxes, labels = [], []
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                cls_id, cx, cy, bw, bh = map(float, line.split())
                x1 = (cx - bw / 2) * self.width
                y1 = (cy - bh / 2) * self.height
                x2 = (cx + bw / 2) * self.width
                y2 = (cy + bh / 2) * self.height
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls_id))  # 0-indexed - torchvision's RetinaNet has no background class

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
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


class DenseNetFPNBackbone(nn.Module):
    """DenseNet body + a plain 3-level FPN (P3/P4/P5 only, out_channels=256). Mirrors
    keras_hub's RetinaNetBackbone(min_level=3, max_level=5) used by the TF version of this
    script - torchvision's BackboneWithFPN helper always adds an extra P6 level
    (LastLevelMaxPool), so this wires the body and FPN directly to skip that."""

    def __init__(self, densenet, height, width, out_channels=256):
        super().__init__()
        self.body = create_feature_extractor(
            densenet, return_nodes={
                "features.denseblock2": "p3",  # stride 8
                "features.denseblock3": "p4",  # stride 16
                "features.norm5": "p5",        # stride 32
            },
        )
        with torch.no_grad():
            dummy = self.body(torch.zeros(1, 3, height, width))
        in_channels_list = [dummy[k].shape[1] for k in ("p3", "p4", "p5")]
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels, extra_blocks=None)
        self.out_channels = out_channels

    def forward(self, x):
        return self.fpn(self.body(x))


def build_model(num_classes, height, width, backbone_name):
    ctor, weights = BACKBONE_BUILDERS[backbone_name]
    backbone = DenseNetFPNBackbone(ctor(weights=weights), height, width)

    # P3/P4/P5 anchor sizes at the RetinaNet paper's standard base scales (32/64/128), 3
    # sizes x 3 aspect ratios per level - same formula torchvision's built-in RetinaNet uses
    # for its 5-level default, just trimmed to the 3 levels DenseNetFPNBackbone produces
    anchor_sizes = tuple((s, int(s * 2 ** (1.0 / 3)), int(s * 2 ** (2.0 / 3))) for s in (32, 64, 128))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    # min_size=height, max_size=width makes RetinaNet's internal GeneralizedRCNNTransform a
    # no-op resize (scale = min(height/height, width/width) == 1) since YoloFormatDataset
    # already resizes every image to exactly (height, width) - keep height < width or this
    # stops being a no-op (see --height's --help text)
    return RetinaNet(backbone, num_classes, min_size=height, max_size=width, anchor_generator=anchor_generator)


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
    parser.add_argument("--backbone", default="densenet121", choices=list(BACKBONE_BUILDERS),
                         help="DenseNet depth - 121 (fastest), 169, or 201 (most accurate, slowest)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4,
                         help="Adam learning rate. Can run higher than the TF version's 5e-5 default: "
                              "torchvision's classification/bbox losses divide by the number of matched "
                              "anchors per image (see RetinaNetClassificationHead.compute_loss), unlike "
                              "keras_hub's raw reduction='sum' over every anchor, so it's less prone to "
                              "the cls_logits_loss blow-up the TF scripts' lr comment warns about")
    parser.add_argument("--height", type=int, default=480,
                         help="Fixed input height to train at - keep this and --width divisible by 32 "
                              "(the P3/P4/P5 pyramid's coarsest stride) and height < width, or the FPN's "
                              "upsample-and-add across levels, and the min_size/max_size no-op resize in "
                              "build_model(), can break")
    parser.add_argument("--width", type=int, default=640, help="Fixed input width to train at - see --height")
    parser.add_argument("--patience", type=int, default=5,
                         help="Epochs with no val mAP50 improvement before ReduceLROnPlateau halves the lr")
    parser.add_argument("--output", default=None,
                         help="Output path. Defaults to models/densenet_torch_<backbone>_detect_best.pt")
    parser.add_argument("--log-dir", default="logs", help="Directory for this run's log file")
    parser.add_argument("--cache", default="disk", choices=["disk", "ram", "none"],
                         help="cache decoded/resized images to avoid re-reading+re-decoding the same JPEG "
                              "every epoch (same idea as train_yolo_detect.py's --cache). 'disk' (default) "
                              "writes each image once under --cache-dir and reuses it on later epochs *and* "
                              "later runs. 'ram' skips the disk write but only lasts this process - useful "
                              "for a small dataset, risky here for the full train split: ~18k images at "
                              "480x640x3 bytes is already close to this Mac's 16GB unified memory on its "
                              "own, before the model/GPU/OS need any. 'none' decodes fresh every access.")
    parser.add_argument("--cache-dir", default="cache/detect_ds_torch",
                         help="Directory for --cache disk files. Shared by default with the other "
                              "train_*_torch_detect.py scripts at the same --height/--width, since the "
                              "cached array is just the decoded/resized image - identical regardless of "
                              "backbone. Delete it if the dataset or --height/--width change (stale cache "
                              "files are per-size-namespaced by filename, so this is only needed to "
                              "reclaim disk space, not for correctness)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu"],
                         help="'cpu' forces CPU-only training (e.g. to run alongside another script's "
                              "GPU/MPS run, or work around a GPU-specific issue). 'auto' (default) trains "
                              "on GPU if one is visible (CUDA, or Apple Silicon via MPS), else CPU")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"models/densenet_torch_{args.backbone}_detect_best.pt"

    emit, log_path = make_emitter(args.log_dir)
    emit(f"Logging to {log_path}")

    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"No dataset at '{args.data}'. Run src/dataset/auto_annotate_bboxes.py then "
            f"src/dataset/prepare_detect_dataset.py first."
        )
    data_root = data_yaml_path.parent
    names_dict = yaml.safe_load(data_yaml_path.read_text())["names"]
    class_names = [names_dict[i] for i in sorted(names_dict)]
    num_classes = len(class_names)  # no +1 for background - see YoloFormatDataset's label comment

    device = torch.device("cpu") if args.device == "cpu" else get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    emit(f"Using device: {device}")

    loader_kwargs = dict(collate_fn=collate_fn, pin_memory=device.type == "cuda")
    dataset_kwargs = dict(cache=args.cache, cache_dir=args.cache_dir if args.cache == "disk" else None)
    train_loader = DataLoader(
        YoloFormatDataset(data_root / "train", args.height, args.width, **dataset_kwargs),
        batch_size=args.batch_size, shuffle=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        YoloFormatDataset(data_root / "val", args.height, args.width, **dataset_kwargs),
        batch_size=args.batch_size, shuffle=False, **loader_kwargs,
    )
    emit(f"Train: {len(train_loader.dataset)} images ({len(train_loader)} batches/epoch)  "
         f"Val: {len(val_loader.dataset)}")

    model = build_model(num_classes, args.height, args.width, args.backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # mode="max" (not the TF version's "min"/val_loss) since torchvision's RetinaNet only
    # exposes losses in train() mode and detections in eval() mode - there's no clean
    # eval-mode val_loss to monitor, so this plateaus on val mAP50 instead, same metric the
    # sibling train_resnet_fasterrcnn_detect.py already checkpoints on
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.patience, min_lr=1e-7,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    best_map50 = -1.0

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
                raise RuntimeError(f"Loss went non-finite ({breakdown}) on device={device}.")

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if epoch_warmup is not None:
                epoch_warmup.step()
            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        if device.type == "mps":
            # MPS's caching allocator doesn't release memory between epochs the way CUDA's
            # does, so long runs slowly accumulate until the OS OOM-kills the process
            torch.mps.empty_cache()

        map50, map5095 = evaluate(model, val_loader, device)
        plateau.step(map50)
        emit(f"Epoch {epoch}/{args.epochs}  loss={epoch_loss / len(train_loader):.4f}  "
             f"val mAP50={map50:.4f}  val mAP50-95={map5095:.4f}  "
             f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if map50 > best_map50:
            best_map50 = map50
            torch.save({
                "model_state_dict": model.state_dict(),
                "family": "densenet-retinanet-torch",
                "backbone": args.backbone,
                "class_names": class_names,
                "height": args.height,
                "width": args.width,
            }, output_path)

    emit(f"\nSaved best model to {output_path} (val mAP50={best_map50:.4f})")
    emit(f"Class names ({num_classes}): {class_names}")
    emit(
        "\nNote: src/evaluate_models.py doesn't recognize the 'densenet-retinanet-torch' "
        "checkpoint family yet - it would need a small loader added, the same way the "
        "*-fasterrcnn families are wired in."
    )


if __name__ == "__main__":
    main()
