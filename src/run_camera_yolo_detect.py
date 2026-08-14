"""
    python src/run_camera_yolo_detect.py --model-path models/yolo26n_best.pt --source webcam
    python src/run_camera_yolo_detect.py --model-path models/yolo26n_best.pt --source droidcam \
        --droidcam-ip 192.168.0.107 --droidcam-port 4747
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
from ultralytics import YOLO

logger = logging.getLogger("camera_yolo_detect")


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


def draw_detections(frame, result):
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = result.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        color = LABEL_COLORS.get(label, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} ({confidence:.0%})"
        text_y = max(20, y1 - 8)
        cv2.putText(frame, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def log_detections(result):
    if not len(result.boxes):
        logger.info("No detections")
        return
    parts = []
    for box in result.boxes:
        label = result.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        parts.append(f"{label} ({confidence:.0%})")
    logger.info("Detected: " + ", ".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/yolo26n_best.pt")
    parser.add_argument("--img-size", type=int, default=640)
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
        raise FileNotFoundError(
            f"No checkpoint at '{args.model_path}'."
        )

    device = str(get_device())
    logger.info(f"Using device: {device}")
    model = YOLO(args.model_path)
    logger.info(f"Loaded model, classes: {model.names}, confidence threshold: {args.conf:.0%}")

    cap = open_capture(args)
    window = "Box Detector - YOLO (q to quit)"
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

            result = model.predict(frame, imgsz=args.img_size, conf=args.conf, device=device, verbose=False)[0]
            draw_detections(frame, result)

            now = time.monotonic()
            if now - last_log_time >= args.log_interval:
                log_detections(result)
                last_log_time = now

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
