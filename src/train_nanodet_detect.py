"""
    python src/train_nanodet_detect.py --epochs 100
"""
import argparse
import os
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ShuffleNet_V2_X1_0_Weights, shufflenet_v2_x1_0
from torchvision.ops import batched_nms, generalized_box_iou_loss, sigmoid_focal_loss
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

STRIDES = (8, 16, 32)
FEAT_CHANNELS = 96
STACKED_CONVS = 2

BACKBONE_CHANNELS = (116, 232, 464)

REGRESS_RANGES = ((0, 64), (64, 128), (128, float("inf")))

REG_LOSS_WEIGHT = 2.0  

EVAL_SCORE_THRESH = 0.05
NMS_IOU_THRESH = 0.6
MAX_DETECTIONS = 100


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NanoDetDataset(Dataset):

    def __init__(self, split_dir, img_size):
        self.images_dir = split_dir / "images"
        self.labels_dir = split_dir / "labels"
        self.img_size = img_size
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()

        boxes, labels = [], []
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                cls_id, cx, cy, bw, bh = map(float, line.split())
                x1 = (cx - bw / 2) * self.img_size
                y1 = (cy - bh / 2) * self.img_size
                x2 = (cx + bw / 2) * self.img_size
                y2 = (cy + bh / 2) * self.img_size
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls_id))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, 0), list(targets)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.scale


class NanoDetNeck(nn.Module):

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.laterals = nn.ModuleList(nn.Conv2d(c, out_channels, 1) for c in in_channels_list)

    def forward(self, feats):
        laterals = [lateral(f) for lateral, f in zip(self.laterals, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        return laterals


class NanoDetHead(nn.Module):

    def __init__(self, num_classes, feat_channels, stacked_convs, strides):
        super().__init__()
        self.cls_convs = nn.ModuleList(DepthwiseSeparableConv(feat_channels) for _ in range(stacked_convs))
        self.reg_convs = nn.ModuleList(DepthwiseSeparableConv(feat_channels) for _ in range(stacked_convs))
        self.cls_pred = nn.Conv2d(feat_channels, num_classes, 3, padding=1)
        self.reg_pred = nn.Conv2d(feat_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList(Scale() for _ in strides)

    def forward(self, feats):
        cls_scores, bbox_preds = [], []
        for feat, scale in zip(feats, self.scales):
            cls_feat = feat
            for conv in self.cls_convs:
                cls_feat = conv(cls_feat)
            reg_feat = feat
            for conv in self.reg_convs:
                reg_feat = conv(reg_feat)
            cls_scores.append(self.cls_pred(cls_feat))
            bbox_preds.append(scale(self.reg_pred(reg_feat)))
        return cls_scores, bbox_preds


class NanoDet(nn.Module):
    def __init__(self, num_classes, feat_channels=FEAT_CHANNELS, stacked_convs=STACKED_CONVS, strides=STRIDES):
        super().__init__()
        backbone = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(backbone.conv1, backbone.maxpool)
        self.stage2 = backbone.stage2
        self.stage3 = backbone.stage3
        self.stage4 = backbone.stage4
        self.neck = NanoDetNeck(BACKBONE_CHANNELS, feat_channels)
        self.head = NanoDetHead(num_classes, feat_channels, stacked_convs, strides)

    def forward(self, x):
        x = self.stem(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        feats = self.neck([c3, c4, c5])
        return self.head(feats)


def generate_points(feat_size, stride, device):
    h, w = feat_size
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # [H*W, 2]


@torch.no_grad()
def assign_targets(points, strides_per_point, ranges_per_point, gt_boxes, gt_labels, num_classes):
    num_points = points.shape[0]
    device = points.device
    cls_targets = torch.zeros((num_points, num_classes), device=device)
    reg_targets = torch.zeros((num_points, 4), device=device)
    pos_mask = torch.zeros(num_points, dtype=torch.bool, device=device)

    if gt_boxes.numel() == 0:
        return cls_targets, reg_targets, pos_mask

    xs, ys = points[:, 0:1], points[:, 1:2]  # [P, 1]
    x1, y1, x2, y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]  # [N]

    ltrb = torch.stack([
        xs - x1[None, :], ys - y1[None, :], x2[None, :] - xs, y2[None, :] - ys,
    ], dim=-1)  # [P, N, 4]

    inside_box = ltrb.min(dim=-1).values > 0  # [P, N]
    max_ltrb = ltrb.max(dim=-1).values  # [P, N]
    low, high = ranges_per_point[:, 0:1], ranges_per_point[:, 1:2]
    inside_range = (max_ltrb >= low) & (max_ltrb <= high)

    areas = (x2 - x1) * (y2 - y1)  # [N]
    candidate_areas = areas[None, :].expand(num_points, -1).clone()
    candidate_areas[~(inside_box & inside_range)] = float("inf")

    min_area, min_idx = candidate_areas.min(dim=1)
    pos_mask = min_area < float("inf")

    pos_idx = torch.nonzero(pos_mask).squeeze(1)
    assigned_gt = min_idx[pos_idx]
    cls_targets[pos_idx, gt_labels[assigned_gt]] = 1.0
    reg_targets[pos_idx] = ltrb[pos_idx, assigned_gt]

    return cls_targets, reg_targets, pos_mask


def compute_loss(cls_scores, bbox_preds, targets, strides, regress_ranges, num_classes, device):
    batch_size = cls_scores[0].shape[0]

    points_per_level, strides_per_level, ranges_per_level = [], [], []
    for cls_score, stride, rng in zip(cls_scores, strides, regress_ranges):
        h, w = cls_score.shape[-2:]
        points = generate_points((h, w), stride, device)
        points_per_level.append(points)
        strides_per_level.append(torch.full((points.shape[0],), float(stride), device=device))
        ranges_per_level.append(torch.tensor(rng, device=device, dtype=torch.float32).expand(points.shape[0], 2))
    all_points = torch.cat(points_per_level, dim=0)
    all_strides = torch.cat(strides_per_level, dim=0)
    all_ranges = torch.cat(ranges_per_level, dim=0)

    flat_cls = torch.cat([
        cs.permute(0, 2, 3, 1).reshape(batch_size, -1, num_classes) for cs in cls_scores
    ], dim=1)  # [B, P, C]
    flat_reg = torch.cat([
        bp.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for bp in bbox_preds
    ], dim=1)  # [B, P, 4] (pre-exp, pre-stride)

    total_cls_loss = flat_cls.new_zeros(())
    total_reg_loss = flat_cls.new_zeros(())
    total_pos = 0
    for b in range(batch_size):
        gt_boxes = targets[b]["boxes"].to(device)
        gt_labels = targets[b]["labels"].to(device)

        cls_target, reg_target, pos_mask = assign_targets(
            all_points, all_strides, all_ranges, gt_boxes, gt_labels, num_classes
        )
        total_cls_loss = total_cls_loss + sigmoid_focal_loss(flat_cls[b], cls_target, reduction="sum")

        num_pos = int(pos_mask.sum().item())
        total_pos += num_pos
        if num_pos == 0:
            continue

        pts = all_points[pos_mask]
        pred_dist = flat_reg[b][pos_mask].exp() * all_strides[pos_mask].unsqueeze(-1)
        pred_boxes = torch.stack([
            pts[:, 0] - pred_dist[:, 0], pts[:, 1] - pred_dist[:, 1],
            pts[:, 0] + pred_dist[:, 2], pts[:, 1] + pred_dist[:, 3],
        ], dim=-1)
        target_dist = reg_target[pos_mask]
        target_boxes = torch.stack([
            pts[:, 0] - target_dist[:, 0], pts[:, 1] - target_dist[:, 1],
            pts[:, 0] + target_dist[:, 2], pts[:, 1] + target_dist[:, 3],
        ], dim=-1)
        total_reg_loss = total_reg_loss + generalized_box_iou_loss(pred_boxes, target_boxes, reduction="sum")

    num_pos = max(total_pos, 1)
    return total_cls_loss / num_pos, total_reg_loss / num_pos


def decode_single(cls_scores, bbox_preds, strides, img_size,
                   score_thresh=EVAL_SCORE_THRESH, nms_iou=NMS_IOU_THRESH, max_dets=MAX_DETECTIONS):
    device = cls_scores[0].device
    all_boxes, all_scores, all_labels = [], [], []
    for cls_score, bbox_pred, stride in zip(cls_scores, bbox_preds, strides):
        h, w = cls_score.shape[-2:]
        points = generate_points((h, w), stride, device)
        scores = cls_score.permute(1, 2, 0).reshape(-1, cls_score.shape[0]).sigmoid()
        dist = bbox_pred.permute(1, 2, 0).reshape(-1, 4).exp() * stride

        max_scores, labels = scores.max(dim=1)
        keep = max_scores > score_thresh
        if not keep.any():
            continue

        pts, dist, max_scores, labels = points[keep], dist[keep], max_scores[keep], labels[keep]
        boxes = torch.stack([
            pts[:, 0] - dist[:, 0], pts[:, 1] - dist[:, 1],
            pts[:, 0] + dist[:, 2], pts[:, 1] + dist[:, 3],
        ], dim=-1).clamp(0, img_size)

        all_boxes.append(boxes)
        all_scores.append(max_scores)
        all_labels.append(labels)

    if not all_boxes:
        return (torch.zeros((0, 4), device=device), torch.zeros(0, device=device),
                torch.zeros(0, dtype=torch.int64, device=device))

    boxes, scores, labels = torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_labels)
    keep = batched_nms(boxes, scores, labels, nms_iou)[:max_dets]
    return boxes[keep], scores[keep], labels[keep]


def decode_batch(cls_scores, bbox_preds, strides, img_size):
    preds = []
    for b in range(cls_scores[0].shape[0]):
        boxes, scores, labels = decode_single(
            [cs[b] for cs in cls_scores], [bp[b] for bp in bbox_preds], strides, img_size,
        )
        preds.append({"boxes": boxes.cpu(), "scores": scores.cpu(), "labels": labels.cpu()})
    return preds


@torch.no_grad()
def evaluate(model, data_loader, device, img_size):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    use_amp = device.type == "cuda"
    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            cls_scores, bbox_preds = model(images)
        preds = decode_batch(cls_scores, bbox_preds, STRIDES, img_size)
        metric.update(preds, targets)
    result = metric.compute()
    return float(result["map_50"]), float(result["map"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_detect/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--output", default="models/nanodet_best.pt")
    args = parser.parse_args()

    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"No dataset at '{args.data}'. Run src/annotate_bboxes.py then "
            f"src/prepare_detect_dataset.py first."
        )
    data_root = data_yaml_path.parent
    class_names = yaml.safe_load(data_yaml_path.read_text())["names"]
    num_classes = len(class_names)

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
        NanoDetDataset(data_root / "train", args.img_size), batch_size=args.batch_size,
        shuffle=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        NanoDetDataset(data_root / "val", args.img_size), batch_size=args.batch_size,
        shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        NanoDetDataset(data_root / "test", args.img_size), batch_size=args.batch_size,
        shuffle=False, **loader_kwargs,
    )
    print(f"Train: {len(train_loader.dataset)} images ({len(train_loader)} batches/epoch)  "
          f"Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")

    model = NanoDet(num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.1)

    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_map50 = -1.0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for images, targets in progress:
            images = images.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                cls_scores, bbox_preds = model(images)
                cls_loss, reg_loss = compute_loss(
                    cls_scores, bbox_preds, targets, STRIDES, REGRESS_RANGES, num_classes, device
                )
                loss = cls_loss + REG_LOSS_WEIGHT * reg_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", cls=f"{cls_loss.item():.4f}", reg=f"{reg_loss.item():.4f}")

        scheduler.step()
        map50, map5095 = evaluate(model, val_loader, device, args.img_size)
        print(f"Epoch {epoch}/{args.epochs}  loss={epoch_loss / len(train_loader):.4f}  "
              f"val mAP50={map50:.4f}  val mAP50-95={map5095:.4f}")

        if map50 > best_map50:
            best_map50 = map50
            torch.save({"model_state_dict": model.state_dict(),
                        "class_names": class_names,
                        "img_size": args.img_size}, args.output)

    print(f"\nSaved best model to {args.output} (val mAP50={best_map50:.4f})")

    best = torch.load(args.output, map_location=device, weights_only=False)
    model = NanoDet(len(best["class_names"])).to(device)
    model.load_state_dict(best["model_state_dict"])
    test_map50, test_map5095 = evaluate(model, test_loader, device, best["img_size"])
    print(f"Test mAP50: {test_map50:.4f}  mAP50-95: {test_map5095:.4f}")


if __name__ == "__main__":
    main()
