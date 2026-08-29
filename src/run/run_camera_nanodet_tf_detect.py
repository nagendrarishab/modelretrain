"""
    python src/run/run_camera_nanodet_tf_detect.py --model-path models/nanodet_tf_best.keras --source webcam
    python src/run/run_camera_nanodet_tf_detect.py --model-path models/nanodet_tf_best.keras \
        --camera-index 1 --save-dir raw_capture --save-interval 2

TensorFlow/Keras counterpart to run_camera_nanodet_detect.py - loads the
plain keras.Model saved by train_nanodet_tf_detect.py (6 raw per-level
outputs: 3 cls_scores + 3 bbox_preds, NHWC) and decodes it the same
anchor-free FCOS-style way. NMS/box-decoding runs entirely in plain numpy
here, outside any traced graph - same "not part of forward()" design as the
PyTorch version, and the reason this family's TFLite export needs no Flex
delegate (see src/convert_tflite.py).

STRIDES/EVAL constants are duplicated verbatim from train_nanodet_tf_detect.py
(same convention the original PyTorch camera script uses) - a change to the
trainer's architecture constants needs the identical change copied here, or
a loaded checkpoint's outputs won't decode correctly.
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
import tensorflow as tf
import yaml

logger = logging.getLogger("camera_nanodet_tf_detect")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Must match train_nanodet_tf_detect.py's copies of these exactly
STRIDES = (8, 16, 32)
NMS_IOU_THRESH = 0.6
MAX_DETECTIONS = 100

LABEL_COLORS = {"open": (0, 200, 0), "closed": (0, 0, 220)}  # BGR


# Custom layer classes below must stay in sync with train_nanodet_tf_detect.py's
# copies exactly (same reason the PyTorch camera script duplicates the whole
# NanoDet/NanoDetNeck/NanoDetHead class hierarchy) - keras.saving.load_model()
# needs these classes registered in this process to deserialize the checkpoint,
# even though (unlike the PyTorch state_dict case) the architecture itself is
# fully described inside the .keras file and doesn't need to be rebuilt by hand.

@keras.saving.register_keras_serializable(package="nanodet_tf")
class ChannelShuffle(keras.layers.Layer):
    def __init__(self, groups=2, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups

    def call(self, x):
        shape = keras.ops.shape(x)
        n, h, w, c = shape[0], shape[1], shape[2], shape[3]
        cpg = c // self.groups
        x = keras.ops.reshape(x, (n, h, w, self.groups, cpg))
        x = keras.ops.transpose(x, (0, 1, 2, 4, 3))
        return keras.ops.reshape(x, (n, h, w, c))

    def get_config(self):
        config = super().get_config()
        config["groups"] = self.groups
        return config


@keras.saving.register_keras_serializable(package="nanodet_tf")
class Scale(keras.layers.Layer):
    def build(self, input_shape):
        self.scale = self.add_weight(name="scale", shape=(), initializer="ones", trainable=True)

    def call(self, x):
        return x * self.scale


@keras.saving.register_keras_serializable(package="nanodet_tf")
class DepthwiseSeparableConv(keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.pad = keras.layers.ZeroPadding2D(1)
        self.dw = keras.layers.DepthwiseConv2D(3, padding="valid", use_bias=False)
        self.pw = keras.layers.Conv2D(channels, 1, use_bias=False)
        self.bn = keras.layers.BatchNormalization(epsilon=1e-5)
        self.act = keras.layers.Activation("relu")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        x = self.pad(x)
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

    def get_config(self):
        config = super().get_config()
        config["channels"] = self.channels
        return config


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


def generate_points(feat_size, stride):
    h, w = feat_size
    ys = (np.arange(h, dtype=np.float32) + 0.5) * stride
    xs = (np.arange(w, dtype=np.float32) + 0.5) * stride
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)


def decode_single(cls_scores, bbox_preds, strides, img_size, score_thresh,
                   nms_iou=NMS_IOU_THRESH, max_dets=MAX_DETECTIONS):
    all_boxes, all_scores, all_labels = [], [], []
    for cls_score, bbox_pred, stride in zip(cls_scores, bbox_preds, strides):
        h, w = cls_score.shape[0], cls_score.shape[1]
        points = generate_points((h, w), stride)
        scores = 1.0 / (1.0 + np.exp(-cls_score.reshape(-1, cls_score.shape[-1])))
        dist = np.exp(bbox_pred.reshape(-1, 4)) * stride

        labels = scores.argmax(axis=1)
        max_scores = scores[np.arange(scores.shape[0]), labels]
        keep = max_scores > score_thresh
        if not keep.any():
            continue

        pts, dist, max_scores, labels = points[keep], dist[keep], max_scores[keep], labels[keep]
        boxes = np.stack([
            pts[:, 0] - dist[:, 0], pts[:, 1] - dist[:, 1],
            pts[:, 0] + dist[:, 2], pts[:, 1] + dist[:, 3],
        ], axis=-1).clip(0, img_size)

        all_boxes.append(boxes)
        all_scores.append(max_scores)
        all_labels.append(labels)

    if not all_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)

    boxes, scores, labels = np.concatenate(all_boxes), np.concatenate(all_scores), np.concatenate(all_labels)
    keep = tf.image.non_max_suppression(boxes, scores, max_dets, iou_threshold=nms_iou).numpy()
    return boxes[keep], scores[keep], labels[keep]


def load_model(model_path, data_yaml="data_detect/data.yaml", img_size=320):
    model = keras.saving.load_model(model_path)
    names_dict = yaml.safe_load(Path(data_yaml).read_text())["names"]
    class_names = [names_dict[i] for i in sorted(names_dict)]
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


def detect(model, class_names, img_size, frame, conf_threshold):
    height, width = frame.shape[:2]
    resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    batch = np.expand_dims(normalized, axis=0)

    outputs = model.predict(batch, verbose=0)
    cls_scores, bbox_preds = outputs[:3], outputs[3:]
    boxes, scores, labels = decode_single(
        [c[0] for c in cls_scores], [b[0] for b in bbox_preds], STRIDES, img_size, conf_threshold,
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", default="models/nanodet_tf_best.keras")
    parser.add_argument("--data", default="data_detect/data.yaml", help="source of class names")
    parser.add_argument("--img-size", type=int, default=320, help="must match the model's trained input size")
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

    model, class_names, img_size = load_model(args.model_path, args.data, args.img_size)
    logger.info(f"Loaded model, classes: {class_names}, img_size: {img_size}, confidence threshold: {args.conf:.0%}")

    cap = open_capture(args)
    window = "Box Detector - NanoDet TF (q to quit)"
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

            detections = detect(model, class_names, img_size, frame, args.conf)
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
