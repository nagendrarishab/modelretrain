"""
    python src/run_camera_yolo_detect.py --model-path models/yolo26n_best.pt --source webcam
    python src/run_camera_yolo_detect.py --model-path models/yolo26n_best.pt --source droidcam \
        --droidcam-ip 192.168.0.107 --droidcam-port 4747
    python src/run_camera_yolo_detect.py --model-path models/yolo26n_best.pt --source file \
        --video-path Camera_2026-08-21_17-28-33.mp4 --no-preview

    # FP16 inference for a faster per-frame loop (biggest win on CUDA, smaller on MPS/CPU)
    python src/run_camera_yolo_detect.py --model-path models/yolo26n_best.pt --quantize 16

    # tighten/loosen detection behavior: dedup overlapping boxes, keep only class 0,
    # and save a crop of every detection for later re-annotation
    python src/run_camera_yolo_detect.py --model-path models/yolo26n_best.pt \
        --iou 0.5 --classes 0 --agnostic-nms --save-crop
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        source_desc = f"webcam index {args.camera_index}"
    elif args.source == "file":
        cap = cv2.VideoCapture(args.video_path)
        source_desc = f"file {args.video_path}"
    else:
        url = args.droidcam_url or f"http://{args.droidcam_ip}:{args.droidcam_port}/video"
        cap = cv2.VideoCapture(url)
        source_desc = f"DroidCam at {url}"

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source_desc}")
    actual_w, actual_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Streaming from {source_desc} at {actual_w}x{actual_h}")
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
    parser.add_argument("--export-format", default=None,
                         help="export --model-path to this format (e.g. ncnn) and exit, instead of running")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5,
                         help="minimum confidence to consider the box 'visible' and draw anything")
    parser.add_argument("--quantize", default=16,
                         help="inference precision, e.g. 16 for FP16 or 32/None for FP32 - FP16 is the "
                              "default since it speeds up per-frame inference (most noticeable on CUDA; "
                              "smaller gains on MPS/CPU) for a negligible accuracy cost; pass --quantize 32 "
                              "to disable if you need exact FP32 parity")
    parser.add_argument("--max-det", type=int, default=10,
                         help="cap on detections per frame; this is a 1-2 object scene (open/closed box) "
                              "so the default 300 just wastes NMS time on candidates that will never matter")
    parser.add_argument("--iou", type=float, default=0.7,
                         help="IoU threshold for NMS; lower it if the same box is being reported twice")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                         help="restrict detections to these class IDs (e.g. --classes 0 1); "
                              "default keeps every class the model knows")
    parser.add_argument("--agnostic-nms", action="store_true",
                         help="suppress overlapping boxes across different classes, not just within "
                              "the same class - use if 'open' and 'closed' both fire on the same box")
    parser.add_argument("--save-crop", action="store_true",
                         help="save a cropped image of each detection under runs/detect/predict*/crops/ "
                              "- useful for feeding auto_annotate_bboxes.py with fresh training candidates")
    parser.add_argument("--source", choices=["webcam", "droidcam", "file"], default="webcam")
    parser.add_argument("--camera-index", type=int, default=0, help="webcam device index")
    parser.add_argument("--camera-width", type=int, default=1920, help="requested capture width (webcam source only)")
    parser.add_argument("--camera-height", type=int, default=1080, help="requested capture height (webcam source only)")
    parser.add_argument("--video-path", default=None, help="path to an input video file (file source only)")
    parser.add_argument("--output-path", default=None,
                         help="if set, write the annotated frames out to this video file (file source only, "
                              "defaults to <video-path stem>_annotated.mp4)")
    parser.add_argument("--no-preview", action="store_true",
                         help="don't open a live cv2 preview window (file source only)")
    parser.add_argument("--save-dir", default=None,
                         help="if set, periodically save raw (undetected) frames here for later dataset curation")
    parser.add_argument("--save-interval", type=float, default=2.0,
                         help="minimum seconds between saved frames when --save-dir is set")
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

    if args.source == "file":
        if not args.video_path or not Path(args.video_path).exists():
            raise FileNotFoundError(f"No video file at '{args.video_path}'.")

    if not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"No checkpoint at '{args.model_path}'."
        )

    device = str(get_device())
    logger.info(f"Using device: {device}")
    model = YOLO(args.model_path)

    if args.export_format:
        exported_path = model.export(format=args.export_format, half=True)
        logger.info(f"Exported to {exported_path}")
        return

    logger.info(f"Loaded model, classes: {model.names}, confidence threshold: {args.conf:.0%}")

    cap = open_capture(args)
    window = "Box Detector - YOLO (q to quit)"
    last_log_time = 0.0
    last_save_time = 0.0
    show_preview = not (args.source == "file" and args.no_preview)

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving raw frames to {save_dir} every {args.save_interval}s")

    writer = None
    output_path = None
    if args.source == "file":
        output_path = Path(args.output_path) if args.output_path else Path(args.video_path).with_name(
            f"{Path(args.video_path).stem}_annotated.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        logger.info(f"Writing annotated video to {output_path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.source == "file":
                    logger.info("Reached end of video.")
                    break
                logger.warning("Lost the video stream, retrying...")
                cap.release()
                time.sleep(args.reconnect_delay)
                cap = open_capture(args)
                continue

            now = time.monotonic()
            if save_dir is not None and now - last_save_time >= args.save_interval:
                frame_path = save_dir / f"frame_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                cv2.imwrite(str(frame_path), frame)
                last_save_time = now

            result = model.predict(
                frame, imgsz=args.img_size, conf=args.conf, device=device, verbose=False,
                quantize=args.quantize, max_det=args.max_det, iou=args.iou,
                classes=args.classes, agnostic_nms=args.agnostic_nms, save_crop=args.save_crop,
            )[0]
            draw_detections(frame, result)

            if now - last_log_time >= args.log_interval:
                log_detections(result)
                last_log_time = now

            if writer is not None:
                writer.write(frame)

            if show_preview:
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            logger.info(f"Saved annotated video to {output_path}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
