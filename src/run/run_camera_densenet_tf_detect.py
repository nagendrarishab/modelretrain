"""
    python src/run/run_camera_densenet_tf_detect.py --model-path models/densenet_tf_densenet121_detect_best.keras --source webcam
    python src/run/run_camera_densenet_tf_detect.py --model-path models/densenet_tf_densenet121_detect_best.keras --source droidcam \
        --droidcam-ip 192.168.0.107 --droidcam-port 4747

Same TF/KerasHub RetinaNet pipeline as run_camera_resnet_tf_detect.py -
load_model()/detect() are backbone-agnostic (KerasHub serializes the whole
RetinaNetObjectDetector, backbone included, into the .keras file), so this
is byte-identical code, just pointed at a DenseNet-backbone checkpoint.
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
import keras
import numpy as np
import yaml

logger = logging.getLogger("camera_densenet_tf_detect")


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


LABEL_COLORS = {"open": (0, 200, 0), "closed": (0, 0, 220)}  # BGR


def load_model(model_path, data_yaml="data_detect/data.yaml"):
    model = keras.saving.load_model(model_path)
    names_dict = yaml.safe_load(Path(data_yaml).read_text())["names"]
    class_names = [names_dict[i] for i in sorted(names_dict)]
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


def detect(model, class_names, frame, conf_threshold, height, width):
    # Predicted box coordinates come back in the model's fixed input
    # resolution (height, width), not the original frame's - the
    # preprocessor's image_converter resizes internally at predict() time
    # but nothing rescales predictions back to the caller's input size. So
    # every box is rescaled back to the original frame here, keeping this
    # function's contract (original-frame pixel space) the same as every
    # other detect() in this repo, and the same as evaluate_models.py expects.
    orig_height, orig_width = frame.shape[:2]
    resized = cv2.resize(frame, (width, height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(rgb, axis=0)
    prediction = model.predict(batch, verbose=0)

    boxes = np.asarray(prediction["boxes"])[0]
    classes = np.asarray(prediction["labels"])[0]
    scores = np.asarray(prediction["confidence"])[0]
    num_detections = int(np.asarray(prediction["num_detections"])[0])
    x_scale, y_scale = orig_width / width, orig_height / height

    detections = []
    for box, cls, score in zip(boxes[:num_detections], classes[:num_detections], scores[:num_detections]):
        if score < conf_threshold or not np.all(np.isfinite(box)):
            continue
        y1, x1, y2, x2 = box.tolist()  # RetinaNet yxyx, in (height, width) input space
        x1 = int(np.clip(x1 * x_scale, 0, orig_width))
        x2 = int(np.clip(x2 * x_scale, 0, orig_width))
        y1 = int(np.clip(y1 * y_scale, 0, orig_height))
        y2 = int(np.clip(y2 * y_scale, 0, orig_height))
        class_name = class_names[int(cls)]  # 0-indexed, no background class
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/densenet_tf_densenet121_detect_best.keras")
    parser.add_argument("--data", default="data_detect/data.yaml", help="source of class names")
    parser.add_argument("--height", type=int, default=480, help="must match the model's trained input size")
    parser.add_argument("--width", type=int, default=640, help="must match the model's trained input size")
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

    model, class_names = load_model(args.model_path, args.data)
    logger.info(f"Loaded model, classes: {class_names}, confidence threshold: {args.conf:.0%}")

    cap = open_capture(args)
    window = "Box Detector - DenseNet RetinaNet TF (q to quit)"
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

            detections = detect(model, class_names, frame, args.conf, args.height, args.width)
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
