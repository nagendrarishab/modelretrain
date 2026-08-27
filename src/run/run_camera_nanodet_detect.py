"""
    python src/run_camera_nanodet_detect.py --model-path models/nanodet_best.pt --source webcam
    python src/run_camera_nanodet_detect.py --model-path models/nanodet_best.pt \
        --camera-index 1 --save-dir raw_capture --save-interval 2
"""
import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ShuffleNet_V2_X1_0_Weights, shufflenet_v2_x1_0
from torchvision.ops import batched_nms

logger = logging.getLogger("camera_nanodet_detect")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Must match train_nanodet_detect.py's copies of these exactly
STRIDES = (8, 16, 32)
FEAT_CHANNELS = 96
STACKED_CONVS = 2
BACKBONE_CHANNELS = (116, 232, 464)

NMS_IOU_THRESH = 0.6
MAX_DETECTIONS = 100


def setup_logging(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"detect_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging to {log_path}")
    return log_path


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


LABEL_COLORS = {"open": (0, 200, 0), "closed": (0, 0, 220)}  # BGR


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
        backbone = shufflenet_v2_x1_0(weights=None)
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
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  


def decode_single(cls_scores, bbox_preds, strides, img_size, score_thresh, nms_iou=NMS_IOU_THRESH, max_dets=MAX_DETECTIONS):
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


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    model = NanoDet(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_names, img_size


def open_capture(args):
    if args.source == "webcam":
        cap = cv2.VideoCapture(args.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        source_desc = f"webcam index {args.camera_index}"
    else:
        url = args.droidcam_url or f"http://{args.droidcam_ip}:{args.droidcam_port}/video"
        cap = cv2.VideoCapture(url)
        source_desc = f"DroidCam at {url}"

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source_desc}")
    logger.info(
        f"Streaming from {source_desc} "
        f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} negotiated)"
    )
    return cap


@torch.no_grad()
def detect(model, class_names, img_size, frame, device, conf_threshold):
    height, width = frame.shape[:2]
    resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    cls_scores, bbox_preds = model(img_tensor)
    boxes, scores, labels = decode_single(
        [cs[0] for cs in cls_scores], [bp[0] for bp in bbox_preds], STRIDES, img_size, conf_threshold,
    )

    scale_x, scale_y = width / img_size, height / img_size
    detections = []
    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        x1, y1, x2, y2 = box
        detections.append((
            int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y),
            class_names[label], float(score),
        ))
    return detections


def draw_detections(frame, detections):
    for x1, y1, x2, y2, label, confidence in detections:
        color = LABEL_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} ({confidence:.0%})"
        text_y = max(20, y1 - 8)
        cv2.putText(frame, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def log_detections(detections):
    if not detections:
        logger.info("No detections")
        return
    parts = [f"{label} ({confidence:.0%})" for *_, label, confidence in detections]
    logger.info("Detected: " + ", ".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/nanodet_best.pt")
    parser.add_argument("--conf", type=float, default=0.5,
                         help="minimum confidence to consider the box 'visible' and draw anything")
    parser.add_argument("--source", choices=["webcam", "droidcam"], default="webcam")
    parser.add_argument("--camera-index", type=int, default=0, help="webcam device index")
    parser.add_argument("--camera-width", type=int, default=1920, help="requested webcam capture width")
    parser.add_argument("--camera-height", type=int, default=1080, help="requested webcam capture height")
    parser.add_argument("--droidcam-ip", default="192.168.0.107")
    parser.add_argument("--droidcam-port", type=int, default=4747)
    parser.add_argument("--droidcam-url", default=None,
                         help="override the full DroidCam stream URL instead of building it from ip/port")
    parser.add_argument("--reconnect-delay", type=float, default=2.0,
                         help="seconds to wait before retrying a dropped stream")
    parser.add_argument("--log-dir", default="logs", help="directory to write detection logs to")
    parser.add_argument("--log-interval", type=float, default=1.0,
                         help="minimum seconds between logged detection snapshots (avoids per-frame spam)")
    parser.add_argument("--save-dir", default=None,
                         help="if set, periodically save the raw (undetected) frame here for later annotation")
    parser.add_argument("--save-interval", type=float, default=2.0,
                         help="minimum seconds between saved frames when --save-dir is set")
    args = parser.parse_args()

    setup_logging(args.log_dir)

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"No checkpoint at '{args.model_path}'.")

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info(f"Using device: {device}")
    model, class_names, img_size = load_model(args.model_path, device)
    logger.info(f"Loaded model, classes: {class_names}, img_size: {img_size}, confidence threshold: {args.conf:.0%}")

    cap = open_capture(args)
    window = "Box Detector - NanoDet (q to quit)"
    last_log_time = 0.0
    last_save_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Lost the video stream, retrying...")
                cap.release()
                time.sleep(args.reconnect_delay)
                cap = open_capture(args)
                continue

            if args.save_dir:
                now = time.monotonic()
                if now - last_save_time >= args.save_interval:
                    save_path = Path(args.save_dir) / f"frame_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                    cv2.imwrite(str(save_path), frame)
                    last_save_time = now

            detections = detect(model, class_names, img_size, frame, device, args.conf)
            draw_detections(frame, detections)

            now = time.monotonic()
            if now - last_log_time >= args.log_interval:
                log_detections(detections)
                last_log_time = now

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
