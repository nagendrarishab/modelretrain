"""
    python src/train/train_nanodet_torch_detect.py --epochs 100

PyTorch port of train_nanodet_tf_detect.py (NanoDet-Plus: ShuffleNetV2 backbone, PAN neck,
SimOTA dynamic label assignment, Quality Focal Loss + Distribution Focal Loss + GIoU loss,
EMA eval weights). Unlike that file, the backbone here is just torchvision's own pretrained
shufflenet_v2_x1_0 - the TF version had to hand-roll ShuffleNetV2 from Keras layers and
port torchvision's weights into it layer by layer (see its convert_shufflenet_weights()),
since Keras has no such model built in. Loss/assignment math (SimOTA, QFL, DFL) is
translated fairly mechanically from that file's numpy/keras.ops - see its docstrings for
the algorithm-level explanation, not repeated here.
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())  # needed to download pretrained weights

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ShuffleNet_V2_X1_0_Weights, shufflenet_v2_x1_0
from torchvision.ops import box_iou, generalized_box_iou_loss, nms
from torchvision.transforms.functional import normalize, to_tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

STRIDES = (8, 16, 32)  # P3/P4/P5 - shufflenet_v2_x1_0's stage2/stage3/stage4 outputs
STAGE_CHANNELS = (116, 232, 464)  # stage2/3/4 out_channels for the x1_0 width multiplier
FEAT_CHANNELS = 96
STACKED_CONVS = 2
REG_MAX = 7  # DFL: distance regressed as a distribution over 0..REG_MAX bins
REG_LOSS_WEIGHT = 2.0
DFL_LOSS_WEIGHT = 0.25
EMA_DECAY = 0.9998
SIMOTA_CANDIDATE_TOPK = 10

EVAL_SCORE_THRESH = 0.05
NMS_IOU_THRESH = 0.6
MAX_DETECTIONS = 100

BN_EPS = 1e-5  # matches shufflenet_v2's own BatchNorm2d eps, kept the same for the neck/head's BNs too


def make_emitter(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"train_nanodet_torch_{datetime.now():%Y%m%d_%H%M%S}.log"
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
    """Same on-disk layout as the TF script's load_split() (images/ + YOLO-format labels/),
    resized to a fixed square img_size and ImageNet-normalized (shufflenet_v2_x1_0's
    pretrained weights expect that, same as the TF version's manual (img/255-mean)/std).

    Returns (image, boxes, labels) rather than a target dict - matches how NanoDet's own
    per-image SimOTA assignment already wants them (see compute_loss).

    --cache mirrors the other train_*_torch_detect.py scripts' --cache: avoids re-decoding
    +resizing the same JPEG every epoch. The TF version doesn't do this (its own load_split()
    comment explains why: preloading the whole 18k-image split as decoded float32 arrays
    OOMs on this Mac - it lazily decodes per-batch instead) - a DataLoader with workers=0
    already gets the same lazy-per-batch memory profile without a DataLoader-level cache, so
    --cache here is a pure speed add-on, not something needed for the OOM fix like it was
    implicitly built into the TF file's design."""

    def __init__(self, split_dir, img_size, cache="disk", cache_dir=None):
        self.images_dir = split_dir / "images"
        self.labels_dir = split_dir / "labels"
        self.img_size = img_size
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
            npy_path = self.cache_dir / f"{image_path.stem}_{self.img_size}x{self.img_size}.npy"
            if npy_path.exists():
                arr = np.load(npy_path)
                if self.ram_cache is not None:
                    self.ram_cache[image_path] = arr
                return arr

        img = Image.open(image_path).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
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
                x1 = (cx - bw / 2) * self.img_size
                y1 = (cy - bh / 2) * self.img_size
                x2 = (cx + bw / 2) * self.img_size
                y2 = (cy + bh / 2) * self.img_size
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls_id))  # 0-indexed, no background class (matches sigmoid-per-class QFL)

        image_tensor = normalize(to_tensor(img), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        return image_tensor, boxes_tensor, labels_tensor


def collate_fn(batch):
    images, boxes, labels = zip(*batch)
    return torch.stack(images), list(boxes), list(labels)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 + BN + ReLU - matches the TF version's conv_bn-based
    block exactly. Instantiated once per stacked conv and called on all 3 pyramid levels
    (see NanoDetHead), so its weights are shared across levels - same as NanoDetHead in the
    original PyTorch NanoDet-Plus this TF file was itself ported from."""

    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=BN_EPS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class Scale(nn.Module):
    """Learnable per-level scalar multiplier on the regression output - one instance per
    pyramid level (not shared, unlike DepthwiseSeparableConv above)."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.scale


class NanoDetNeck(nn.Module):
    """1x1 lateral convs to a common channel count, then a full PAN pass: top-down
    nearest-upsample-and-add followed by a bottom-up strided-downsample-and-add - same
    bidirectional fusion as the TF version's build_neck()."""

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.downs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels, eps=BN_EPS),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels, eps=BN_EPS),
                nn.ReLU(inplace=True),
            )
            for _ in range(len(in_channels_list) - 1)
        ])

    def forward(self, feats):
        laterals = [lat(f) for lat, f in zip(self.laterals, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + up
        for i in range(len(laterals) - 1):
            laterals[i + 1] = laterals[i + 1] + self.downs[i](laterals[i])
        return laterals


class NanoDetHead(nn.Module):

    def __init__(self, feat_channels, stacked_convs, num_classes, reg_max, num_levels):
        super().__init__()
        self.cls_convs = nn.ModuleList([DepthwiseSeparableConv(feat_channels) for _ in range(stacked_convs)])
        self.reg_convs = nn.ModuleList([DepthwiseSeparableConv(feat_channels) for _ in range(stacked_convs)])
        self.cls_pred = nn.Conv2d(feat_channels, num_classes, 3, padding=1)
        # 4*(reg_max+1): a discrete probability distribution over 0..reg_max per side
        # (left/top/right/bottom), decoded via integral_distribution() - DFL, not a direct
        # scalar regression.
        self.reg_pred = nn.Conv2d(feat_channels, 4 * (reg_max + 1), 3, padding=1)
        self.scales = nn.ModuleList([Scale() for _ in range(num_levels)])

    def forward(self, feats):
        cls_scores, bbox_preds = [], []
        for level, feat in enumerate(feats):
            cls_feat = feat
            for conv in self.cls_convs:
                cls_feat = conv(cls_feat)
            reg_feat = feat
            for conv in self.reg_convs:
                reg_feat = conv(reg_feat)
            cls_scores.append(self.cls_pred(cls_feat))
            bbox_preds.append(self.scales[level](self.reg_pred(reg_feat)))
        return cls_scores, bbox_preds


class NanoDetModel(nn.Module):

    def __init__(self, num_classes, feat_channels=FEAT_CHANNELS, stacked_convs=STACKED_CONVS, reg_max=REG_MAX):
        super().__init__()
        backbone = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(backbone.conv1, backbone.maxpool)
        self.stage2 = backbone.stage2  # stride 8
        self.stage3 = backbone.stage3  # stride 16
        self.stage4 = backbone.stage4  # stride 32
        self.neck = NanoDetNeck(list(STAGE_CHANNELS), feat_channels)
        self.head = NanoDetHead(feat_channels, stacked_convs, num_classes, reg_max, num_levels=len(STRIDES))

    def forward(self, x):
        x = self.stem(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        neck_feats = self.neck([c3, c4, c5])
        return self.head(neck_feats)


def generate_points(feat_size, stride, device):
    h, w = feat_size
    ys = (torch.arange(h, dtype=torch.float32, device=device) + 0.5) * stride
    xs = (torch.arange(w, dtype=torch.float32, device=device) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # [H*W, 2]


def integral_distribution(dist_logits, reg_max):
    """Softmax-weighted integral over the reg_max+1 discrete distance bins (DFL decode):
    converts each side's (left/top/right/bottom) probability distribution over integer
    distances into one continuous distance, in stride units. Differentiable - unlike the TF
    version, which needed separate numpy (assignment-cost, no-grad) and keras.ops
    (loss, differentiable) copies of this because TF graph mode can't mix the two, a single
    torch function covers both: call it under torch.no_grad() for the cost computation and
    normally for the loss (see compute_loss)."""
    dist_logits = dist_logits.reshape(*dist_logits.shape[:-1], 4, reg_max + 1)
    probs = F.softmax(dist_logits, dim=-1)
    bins = torch.arange(reg_max + 1, dtype=torch.float32, device=dist_logits.device)
    return (probs * bins).sum(dim=-1)  # [..., 4]


def simota_assign_targets(points, gt_boxes, gt_labels, pred_scores, pred_boxes, num_classes,
                           candidate_topk=SIMOTA_CANDIDATE_TOPK, iou_cost_weight=3.0):
    """Dynamic soft label assignment (SimOTA-style, as used by NanoDet-Plus/YOLOX) - see the
    TF version's docstring for the full algorithm explanation. pred_scores/pred_boxes must
    already be detached (no grad needed for the assignment itself, matches the TF version's
    numpy-side convention)."""
    num_points = points.shape[0]
    device = points.device
    cls_targets = torch.zeros((num_points, num_classes), dtype=torch.float32, device=device)
    reg_targets = torch.zeros((num_points, 4), dtype=torch.float32, device=device)
    pos_mask = torch.zeros(num_points, dtype=torch.bool, device=device)

    if gt_boxes.shape[0] == 0:
        return cls_targets, reg_targets, pos_mask

    xs, ys = points[:, 0:1], points[:, 1:2]  # [P, 1]
    x1, y1, x2, y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]  # [N]
    ltrb = torch.stack([
        xs - x1[None, :], ys - y1[None, :], x2[None, :] - xs, y2[None, :] - ys,
    ], dim=-1)  # [P, N, 4]
    inside_box = ltrb.amin(dim=-1) > 0  # [P, N]

    if not inside_box.any():
        return cls_targets, reg_targets, pos_mask

    iou_matrix = box_iou(pred_boxes, gt_boxes)  # [P, N]
    cls_cost = -torch.log(pred_scores[:, gt_labels] + 1e-8)  # [P, N]
    iou_cost = -torch.log(iou_matrix + 1e-8)
    cost = cls_cost + iou_cost_weight * iou_cost
    cost = cost.masked_fill(~inside_box, 1e5)

    k = min(candidate_topk, num_points)
    topk_ious = torch.topk(iou_matrix, k, dim=0).values  # [k, N]
    dynamic_ks = topk_ious.sum(dim=0).long().clamp(min=1)  # [N]

    num_gt = gt_boxes.shape[0]
    matching_matrix = torch.zeros((num_points, num_gt), dtype=torch.bool, device=device)
    for n in range(num_gt):
        valid = torch.nonzero(inside_box[:, n], as_tuple=True)[0]
        if valid.numel() == 0:
            continue
        n_select = min(int(dynamic_ks[n]), valid.numel())
        chosen = valid[torch.argsort(cost[valid, n])[:n_select]]
        matching_matrix[chosen, n] = True

    multi_match = matching_matrix.sum(dim=1) > 1
    for p in torch.nonzero(multi_match, as_tuple=True)[0]:
        matched_gts = torch.nonzero(matching_matrix[p], as_tuple=True)[0]
        best = matched_gts[torch.argmin(cost[p, matched_gts])]
        matching_matrix[p, :] = False
        matching_matrix[p, best] = True

    pos_idx = torch.nonzero(matching_matrix.any(dim=1), as_tuple=True)[0]
    if pos_idx.numel() == 0:
        return cls_targets, reg_targets, pos_mask
    assigned_gt = matching_matrix[pos_idx].long().argmax(dim=1)

    pos_mask[pos_idx] = True
    cls_targets[pos_idx, gt_labels[assigned_gt]] = iou_matrix[pos_idx, assigned_gt]
    reg_targets[pos_idx] = ltrb[pos_idx, assigned_gt]

    return cls_targets, reg_targets, pos_mask


def quality_focal_loss(cls_target, cls_logits, beta=2.0):
    """Quality Focal Loss: sigmoid BCE scaled by |target - sigmoid(logit)|**beta.
    `cls_target` is the continuous IoU quality score from simota_assign_targets (0 at
    negatives), not a hard 0/1 label. Sum reduction, matching the TF version."""
    pred_sigmoid = torch.sigmoid(cls_logits)
    scale_factor = (cls_target - pred_sigmoid).abs() ** beta
    bce = F.binary_cross_entropy_with_logits(cls_logits, cls_target, reduction="none")
    return (bce * scale_factor).sum()


def distribution_focal_loss(pred_logits, target_dist_units, reg_max):
    """Cross-entropy against the two integer bins straddling the (fractional) target
    distance, weighted by proximity. `target_dist_units` is in stride units, matching
    integral_distribution()'s output space. Sum reduction, matching the TF version."""
    pred_logits = pred_logits.reshape(-1, 4, reg_max + 1)
    target_dist_units = target_dist_units.clamp(0.0, float(reg_max) - 1e-3)
    left = target_dist_units.floor().long()
    right = left + 1
    weight_left = right.float() - target_dist_units
    weight_right = target_dist_units - left.float()

    log_probs = F.log_softmax(pred_logits, dim=-1)
    ce_left = -torch.gather(log_probs, -1, left.unsqueeze(-1)).squeeze(-1)
    ce_right = -torch.gather(log_probs, -1, right.unsqueeze(-1)).squeeze(-1)
    return (ce_left * weight_left + ce_right * weight_right).sum()


def compute_loss(cls_scores, bbox_preds, batch_boxes, batch_labels, strides, reg_max, num_classes):
    batch_size = cls_scores[0].shape[0]
    device = cls_scores[0].device

    points_per_level = [
        generate_points((cs.shape[2], cs.shape[3]), stride, device)
        for cs, stride in zip(cls_scores, strides)
    ]
    all_points = torch.cat(points_per_level, dim=0)
    all_strides = torch.cat([
        torch.full((p.shape[0],), s, dtype=torch.float32, device=device)
        for p, s in zip(points_per_level, strides)
    ])

    # NCHW -> [B, P, C] per-point vectors
    flat_cls = torch.cat(
        [cs.permute(0, 2, 3, 1).reshape(batch_size, -1, num_classes) for cs in cls_scores], dim=1,
    )
    flat_reg = torch.cat(
        [bp.permute(0, 2, 3, 1).reshape(batch_size, -1, 4 * (reg_max + 1)) for bp in bbox_preds], dim=1,
    )

    total_cls_loss = torch.zeros((), device=device)
    total_reg_loss = torch.zeros((), device=device)
    total_dfl_loss = torch.zeros((), device=device)
    total_pos = 0
    for b in range(batch_size):
        # Detached copies of this image's own current predictions, used only to compute the
        # SimOTA assignment cost - no gradient flows through the assignment itself.
        with torch.no_grad():
            pred_scores = torch.sigmoid(flat_cls[b])
            pred_dist_units = integral_distribution(flat_reg[b], reg_max)
            pred_boxes = torch.stack([
                all_points[:, 0] - pred_dist_units[:, 0] * all_strides,
                all_points[:, 1] - pred_dist_units[:, 1] * all_strides,
                all_points[:, 0] + pred_dist_units[:, 2] * all_strides,
                all_points[:, 1] + pred_dist_units[:, 3] * all_strides,
            ], dim=-1)

            cls_target, reg_target, pos_mask = simota_assign_targets(
                all_points, batch_boxes[b], batch_labels[b], pred_scores, pred_boxes, num_classes,
            )
        total_cls_loss = total_cls_loss + quality_focal_loss(cls_target, flat_cls[b])

        num_pos = int(pos_mask.sum())
        total_pos += num_pos
        if num_pos == 0:
            continue

        pos_idx = torch.nonzero(pos_mask, as_tuple=True)[0]
        pts = all_points[pos_idx]
        strides_pos = all_strides[pos_idx]
        pred_logits_pos = flat_reg[b][pos_idx]  # [num_pos, 4*(reg_max+1)]
        pred_dist_units = integral_distribution(pred_logits_pos, reg_max)  # differentiable here
        pred_dist = pred_dist_units * strides_pos[:, None]
        pred_boxes_pos = torch.stack([
            pts[:, 0] - pred_dist[:, 0], pts[:, 1] - pred_dist[:, 1],
            pts[:, 0] + pred_dist[:, 2], pts[:, 1] + pred_dist[:, 3],
        ], dim=-1)
        target_dist = reg_target[pos_idx]
        target_boxes = torch.stack([
            pts[:, 0] - target_dist[:, 0], pts[:, 1] - target_dist[:, 1],
            pts[:, 0] + target_dist[:, 2], pts[:, 1] + target_dist[:, 3],
        ], dim=-1)
        total_reg_loss = total_reg_loss + generalized_box_iou_loss(pred_boxes_pos, target_boxes, reduction="sum")

        target_dist_units = target_dist / strides_pos[:, None]
        total_dfl_loss = total_dfl_loss + distribution_focal_loss(pred_logits_pos, target_dist_units, reg_max)

    num_pos = max(total_pos, 1)
    return total_cls_loss / num_pos, total_reg_loss / num_pos, total_dfl_loss / num_pos


def decode_single(cls_scores, bbox_preds, strides, img_size, reg_max,
                   score_thresh=EVAL_SCORE_THRESH, nms_iou=NMS_IOU_THRESH, max_dets=MAX_DETECTIONS):
    """Single-image decode: raw per-level (C,H,W)/(4*(reg_max+1),H,W) tensors -> boxes/
    scores/labels. NMS is class-agnostic (plain torchvision.ops.nms, not batched_nms),
    matching the TF version's tf.image.non_max_suppression call across all classes at once."""
    device = cls_scores[0].device
    all_boxes, all_scores, all_labels = [], [], []
    for cls_score, bbox_pred, stride in zip(cls_scores, bbox_preds, strides):
        c, h, w = cls_score.shape
        points = generate_points((h, w), stride, device)
        scores = torch.sigmoid(cls_score.permute(1, 2, 0).reshape(-1, c))
        dist_units = integral_distribution(bbox_pred.permute(1, 2, 0).reshape(-1, 4 * (reg_max + 1)), reg_max)
        dist = dist_units * stride

        max_scores, labels = scores.max(dim=1)
        keep = max_scores > score_thresh
        if not keep.any():
            continue

        pts, dist_k, scores_k, labels_k = points[keep], dist[keep], max_scores[keep], labels[keep]
        boxes = torch.stack([
            pts[:, 0] - dist_k[:, 0], pts[:, 1] - dist_k[:, 1],
            pts[:, 0] + dist_k[:, 2], pts[:, 1] + dist_k[:, 3],
        ], dim=-1).clamp(0, img_size)

        all_boxes.append(boxes)
        all_scores.append(scores_k)
        all_labels.append(labels_k)

    if not all_boxes:
        return (torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.int64))

    boxes, scores, labels = torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_labels)
    keep = nms(boxes, scores, nms_iou)[:max_dets]
    return boxes[keep].cpu(), scores[keep].cpu(), labels[keep].cpu()


@torch.no_grad()
def evaluate(model, data_loader, device, img_size, reg_max):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    for images, batch_boxes, batch_labels in data_loader:
        images = images.to(device, non_blocking=True)
        cls_scores, bbox_preds = model(images)
        preds, targets = [], []
        for i in range(images.shape[0]):
            boxes, scores, labels = decode_single(
                [cs[i] for cs in cls_scores], [bp[i] for bp in bbox_preds], STRIDES, img_size, reg_max,
            )
            preds.append({"boxes": boxes, "scores": scores, "labels": labels})
            targets.append({"boxes": batch_boxes[i].cpu(), "labels": batch_labels[i].cpu()})
        metric.update(preds, targets)
    result = metric.compute()
    return float(result["map_50"]), float(result["map"])


class EMA:
    """Exponential moving average of trainable parameters (NanoDet-Plus's
    ExpMovingAverager). update() runs after every optimizer step; swap_in()/swap_out()
    temporarily install the shadow weights for eval+checkpointing, then restore the live
    (non-EMA) weights so training continues on those, not the average."""

    def __init__(self, parameters, decay):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in parameters]
        self.backup = None

    @torch.no_grad()
    def update(self, parameters):
        for s, p in zip(self.shadow, parameters):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, parameters):
        self.backup = [p.detach().clone() for p in parameters]
        for p, s in zip(parameters, self.shadow):
            p.data.copy_(s)

    @torch.no_grad()
    def swap_out(self, parameters):
        for p, b in zip(parameters, self.backup):
            p.data.copy_(b)
        self.backup = None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data_detect/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg-max", type=int, default=REG_MAX,
                         help="DFL: number of distance bins per side is reg_max+1")
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY,
                         help="exponential moving average decay for eval/checkpoint weights")
    parser.add_argument("--output", default="models/nanodet_torch_best.pt")
    parser.add_argument("--log-dir", default="logs", help="Directory for this run's log file")
    parser.add_argument("--cache", default="disk", choices=["disk", "ram", "none"],
                         help="cache decoded/resized images to avoid re-reading+re-decoding the same JPEG "
                              "every epoch (same idea as train_yolo_detect.py's --cache and the other "
                              "train_*_torch_detect.py scripts). 'disk' (default) writes each image once "
                              "under --cache-dir and reuses it on later epochs *and* later runs. 'ram' "
                              "skips the disk write but only lasts this process. 'none' decodes fresh "
                              "every access - closest to the TF version's own lazy per-batch loading.")
    parser.add_argument("--cache-dir", default="cache/detect_ds_nanodet_torch",
                         help="Directory for --cache disk files. What's cached is the same thing as the "
                              "other train_*_torch_detect.py scripts' cache - a raw resized uint8 RGB "
                              "array; ImageNet normalization happens after loading, not before caching - "
                              "so it's namespaced separately here only because --img-size defaults to a "
                              "different (square) size than their --height/--width; point this at the "
                              "same --cache-dir as one of them and pass a matching --img-size to actually "
                              "share cached files. Delete it if the dataset or --img-size change.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu"],
                         help="'cpu' forces CPU-only training (e.g. to run alongside another script's "
                              "GPU/MPS run, or work around a GPU-specific issue). 'auto' (default) trains "
                              "on GPU if one is visible (CUDA, or Apple Silicon via MPS), else CPU")
    args = parser.parse_args()

    emit, log_path = make_emitter(args.log_dir)
    emit(f"Logging to {log_path}")

    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"No dataset at '{args.data}'. Run src/dataset/auto_annotate_bboxes.py then "
            f"src/dataset/prepare_detect_dataset.py first."
        )
    data_root = data_yaml_path.parent
    class_names_dict = yaml.safe_load(data_yaml_path.read_text())["names"]
    class_names = [class_names_dict[i] for i in sorted(class_names_dict)]
    num_classes = len(class_names)

    device = torch.device("cpu") if args.device == "cpu" else get_device()
    emit(f"Using device: {device}")

    dataset_kwargs = dict(cache=args.cache, cache_dir=args.cache_dir if args.cache == "disk" else None)
    train_loader = DataLoader(
        YoloFormatDataset(data_root / "train", args.img_size, **dataset_kwargs),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        YoloFormatDataset(data_root / "val", args.img_size, **dataset_kwargs),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )
    emit(f"Train: {len(train_loader.dataset)} images ({len(train_loader)} batches/epoch)  "
         f"Val: {len(val_loader.dataset)}")

    model = NanoDetModel(num_classes, reg_max=args.reg_max).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
    ema = EMA(list(model.parameters()), args.ema_decay)
    step_size = max(args.epochs // 3, 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_map50 = -1.0

    for epoch in range(1, args.epochs + 1):
        for group in optimizer.param_groups:
            group["lr"] = args.lr * (0.1 ** ((epoch - 1) // step_size))

        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for images, batch_boxes, batch_labels in progress:
            images = images.to(device, non_blocking=True)
            batch_boxes = [b.to(device, non_blocking=True) for b in batch_boxes]
            batch_labels = [l.to(device, non_blocking=True) for l in batch_labels]

            cls_scores, bbox_preds = model(images)
            cls_loss, reg_loss, dfl_loss = compute_loss(
                cls_scores, bbox_preds, batch_boxes, batch_labels, STRIDES, args.reg_max, num_classes,
            )
            loss = cls_loss + REG_LOSS_WEIGHT * reg_loss + DFL_LOSS_WEIGHT * dfl_loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Loss went non-finite (cls={cls_loss.item():.4f} reg={reg_loss.item():.4f} "
                    f"dfl={dfl_loss.item():.4f}) on device={device}."
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model.parameters())
            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        if device.type == "mps":
            # MPS's caching allocator doesn't release memory between epochs the way CUDA's
            # does, so long runs slowly accumulate until the OS OOM-kills the process
            torch.mps.empty_cache()

        ema.swap_in(model.parameters())
        map50, map5095 = evaluate(model, val_loader, device, args.img_size, args.reg_max)
        emit(f"Epoch {epoch}/{args.epochs}  loss={epoch_loss / len(train_loader):.4f}  "
             f"val mAP50={map50:.4f}  val mAP50-95={map5095:.4f}")

        if map50 > best_map50:
            best_map50 = map50
            torch.save({
                "model_state_dict": model.state_dict(),
                "family": "nanodet-torch",
                "class_names": class_names,
                "img_size": args.img_size,
                "reg_max": args.reg_max,
            }, output_path)
        ema.swap_out(model.parameters())

    emit(f"\nSaved best model to {output_path} (val mAP50={best_map50:.4f})")
    emit(f"Class names ({num_classes}): {class_names}")
    emit(
        "\nNote: src/evaluate_models.py doesn't recognize the 'nanodet-torch' checkpoint "
        "family yet - it would need a small loader added, the same way the *-fasterrcnn "
        "families are wired in."
    )


if __name__ == "__main__":
    main()
