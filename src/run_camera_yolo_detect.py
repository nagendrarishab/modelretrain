"""
    python src/run_camera_yolo_detect.py --model-path models/yolo26n_best.pt --source webcam
    python src/run_camera_yolo_detect.py --model-path models/yolo26n_best.pt --source droidcam \
        --droidcam-ip 192.168.0.107 --droidcam-port 4747
"""
import argparse
import os
import time
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import cv2
import torch
from ultralytics import YOLO


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
    print(f"Streaming from {source_desc}")
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
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"No checkpoint at '{args.model_path}'."ß
        )

    device = str(get_device())
    print(f"Using device: {device}")
    model = YOLO(args.model_path)
    print(f"Loaded model, classes: {model.names}, confidence threshold: {args.conf:.0%}")

    cap = open_capture(args)
    window = "Box Detector - YOLO (q to quit)"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost the video stream, retrying...")
                cap.release()
                time.sleep(args.reconnect_delay)
                cap = open_capture(args)
                continue

            result = model.predict(frame, imgsz=args.img_size, conf=args.conf, device=device, verbose=False)[0]
            draw_detections(frame, result)

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
