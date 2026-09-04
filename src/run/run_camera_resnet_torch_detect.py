"""
    python src/run/run_camera_resnet_torch_detect.py --model-path models/resnet_torch_resnet18_detect_best.pt --source webcam
    python src/run/run_camera_resnet_torch_detect.py --model-path models/resnet_torch_resnet18_detect_best.pt --source droidcam \
        --droidcam-ip 192.168.0.107 --droidcam-port 4747

PyTorch counterpart of run_camera_resnet_tf_detect.py, for checkpoints produced by
train_resnet_torch_detect.py (torchvision ResNet + a 3-level FPN, wrapped in torchvision's
own RetinaNet - see that file's ResNetFPNBackbone/build_model for the architecture).
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
import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork
from torchvision.transforms.functional import to_tensor

logger = logging.getLogger("camera_resnet_torch_detect")

BACKBONE_CTORS = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}


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


class ResNetFPNBackbone(nn.Module):
    """Must match train_resnet_torch_detect.py's ResNetFPNBackbone exactly - same
    layer2/3/4 -> p3/p4/p5 feature extraction feeding a plain 3-level FPN."""

    def __init__(self, resnet, height, width, out_channels=256):
        super().__init__()
        self.body = create_feature_extractor(
            resnet, return_nodes={"layer2": "p3", "layer3": "p4", "layer4": "p5"},
        )
        with torch.no_grad():
            dummy = self.body(torch.zeros(1, 3, height, width))
        in_channels_list = [dummy[k].shape[1] for k in ("p3", "p4", "p5")]
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels, extra_blocks=None)
        self.out_channels = out_channels

    def forward(self, x):
        return self.fpn(self.body(x))


def build_model(num_classes, height, width, backbone_name):
    backbone = ResNetFPNBackbone(BACKBONE_CTORS[backbone_name](weights=None), height, width)
    anchor_sizes = tuple((s, int(s * 2 ** (1.0 / 3)), int(s * 2 ** (2.0 / 3))) for s in (32, 64, 128))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    return RetinaNet(backbone, num_classes, min_size=height, max_size=width, anchor_generator=anchor_generator)


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    height, width = checkpoint["height"], checkpoint["width"]

    model = build_model(len(class_names), height, width, checkpoint["backbone"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_names, height, width


def open_capture(args):
    if args.source == "webcam":
        cap = cv2.VideoCapture(args.camera_index)
        source_desc = f"webcam index {args.camera_index}"
    else:
        url = args.droidcam_url or f"http://{args.droidcam_ip}:{args.droidcam_port}/video"
        cap = cv2.VideoCapture(url)
        source_desc = f"DroidCam at {url}"

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source_desc}")
    logger.info(f"Streaming from {source_desc}")
    return cap


@torch.no_grad()
def detect(model, class_names, frame, device, conf_threshold, height, width):
    orig_height, orig_width = frame.shape[:2]
    resized = cv2.resize(frame, (width, height))
    img_tensor = to_tensor(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).to(device)
    prediction = model([img_tensor])[0]

    x_scale, y_scale = orig_width / width, orig_height / height
    detections = []
    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        x1, x2 = int(x1 * x_scale), int(x2 * x_scale)
        y1, y2 = int(y1 * y_scale), int(y2 * y_scale)
        class_name = class_names[int(label.item())]  # 0-indexed, RetinaNet has no background class
        detections.append((x1, y1, x2, y2, class_name, float(score)))
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="models/resnet_torch_resnet18_detect_best.pt")
    parser.add_argument("--conf", type=float, default=0.5,
                         help="minimum confidence to consider the box 'visible' and draw anything")
    parser.add_argument("--source", choices=["webcam", "droidcam"], default="webcam")
    parser.add_argument("--camera-index", type=int, default=0, help="webcam device index")
    parser.add_argument("--droidcam-ip", default="192.168.0.107")
    parser.add_argument("--droidcam-port", type=int, default=4747)
    parser.add_argument("--droidcam-url", default=None,
                         help="override the full DroidCam stream URL instead of building it from ip/port")
    parser.add_argument("--reconnect-delay", type=float, default=2.0,
                         help="seconds to wait before retrying a dropped stream")
    parser.add_argument("--log-dir", default="logs", help="directory to write detection logs to")
    parser.add_argument("--log-interval", type=float, default=1.0,
                         help="minimum seconds between logged detection snapshots (avoids per-frame spam)")
    args = parser.parse_args()

    setup_logging(args.log_dir)

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"No checkpoint at '{args.model_path}'.")

    device = get_device()
    logger.info(f"Using device: {device}")
    model, class_names, height, width = load_model(args.model_path, device)
    logger.info(f"Loaded model, classes: {class_names}, input size: {height}x{width}, "
                f"confidence threshold: {args.conf:.0%}")

    cap = open_capture(args)
    window = "Box Detector - ResNet RetinaNet Torch (q to quit)"
    last_log_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Lost the video stream, retrying...")
                cap.release()
                time.sleep(args.reconnect_delay)
                cap = open_capture(args)
                continue

            detections = detect(model, class_names, frame, device, args.conf, height, width)
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
