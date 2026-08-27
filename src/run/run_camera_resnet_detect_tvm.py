"""Hybrid inference: ResNet+FPN backbone runs compiled Apache TVM, RPN/ROI
heads/NMS run in PyTorch on the resulting feature maps (see
tvm_backbone.py / convert-tvm.py for why the split is necessary).

The first frame from the camera is used to determine the model's own
resize target size (via FasterRCNN's internal transform), and the TVM
backbone is compiled for that exact fixed shape - cached to disk so
repeat runs at the same camera resolution skip recompilation.

    python src/run_camera_resnet_detect_tvm.py --model-path models/resnet_resnet18_detect_best.pt --source webcam
    python src/run_camera_resnet_detect_tvm.py --model-path models/resnet_resnet18_detect_best.pt --source droidcam \
        --droidcam-ip 192.168.0.107 --droidcam-port 4747
"""
import argparse
import logging
import os
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import cv2
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.transforms.functional import to_tensor

import tvm
from tvm import relax
from tvm_backbone import FEATURE_NAMES, compile_backbone_cached

logger = logging.getLogger("camera_resnet_detect_tvm")


def setup_logging(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"detect_tvm_{datetime.now():%Y%m%d_%H%M%S}.log"

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


LABEL_COLORS = {"open": (0, 200, 0), "closed": (0, 0, 220)}  # BGR


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    backbone = resnet_fpn_backbone(backbone_name=checkpoint["backbone"], weights=None)
    model = FasterRCNN(backbone, num_classes=len(class_names) + 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


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


def resize_target_shape(model, sample_tensor):
    """The exact (height, width) FasterRCNN's own transform resizes an
    image of this shape to. The TVM backbone must be compiled for this
    exact shape - it's fixed for a given camera resolution, since the
    resize is deterministic given the input size."""
    with torch.no_grad():
        image_list, _ = model.transform([sample_tensor])
    _, _, height, width = image_list.tensors.shape
    return height, width


@torch.no_grad()
def detect(model, vm, class_names, frame, conf_threshold):
    img_tensor = to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    original_size = img_tensor.shape[-2:]

    image_list, _ = model.transform([img_tensor])

    tvm_input = tvm.runtime.tensor(image_list.tensors.numpy(), device=tvm.cpu())
    tvm_features = vm["main"](tvm_input)
    features = OrderedDict(
        (name, torch.from_numpy(tvm_features[i].numpy()))
        for i, name in enumerate(FEATURE_NAMES)
    )

    proposals, _ = model.rpn(image_list, features, None)
    detections, _ = model.roi_heads(features, proposals, image_list.image_sizes, None)
    detections = model.transform.postprocess(detections, image_list.image_sizes, [original_size])
    prediction = detections[0]

    results = []
    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        class_name = class_names[int(label.item()) - 1]  # label 0 is background
        results.append((x1, y1, x2, y2, class_name, float(score)))
    return results


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
    parser.add_argument("--model-path", default="models/resnet_resnet18_detect_best.pt")
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
    parser.add_argument("--tvm-cache-dir", default="models/tvm_cache",
                         help="where to cache the compiled TVM backbone (keyed by resolution)")
    args = parser.parse_args()

    setup_logging(args.log_dir)

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"No checkpoint at '{args.model_path}'.")

    logger.info("Using device: cpu (TVM backbone) + cpu (PyTorch RPN/ROI/NMS)")
    model, class_names = load_model(args.model_path)
    logger.info(f"Loaded model, classes: {class_names}, confidence threshold: {args.conf:.0%}")

    cap = open_capture(args)
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read a first frame to determine the resize target shape")

    sample_tensor = to_tensor(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    height, width = resize_target_shape(model, sample_tensor)
    checkpoint_stem = Path(args.model_path).stem
    cache_path = Path(args.tvm_cache_dir) / f"{checkpoint_stem}_backbone_{height}x{width}.so"
    logger.info(f"Camera frame resizes to {height}x{width}; compiling/loading TVM backbone at {cache_path} ...")
    ex = compile_backbone_cached(model.backbone, height, width, cache_path)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    logger.info("TVM backbone ready")

    window = "Box Detector - ResNet Faster R-CNN + TVM backbone (q to quit)"
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

            detections = detect(model, vm, class_names, frame, args.conf)
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
