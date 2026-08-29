"""
    python src/evaluate_models.py --model-path models/yolo26n_best.pt
    python src/evaluate_models.py --model-path models/yolo26n_best.pt models/resnet_tf_resnet50_detect_best.keras \
        models/nanodet_best.pt --conf 0.5 --iou 0.5

Deliberately NOT self-contained like this repo's trainers/camera scripts: it
imports load_model()/detect()/get_device() from the run_camera_*_detect.py
files instead of re-deriving each architecture's FPN/backbone wiring by hand
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

sys.path.insert(0, str(Path(__file__).resolve().parent / "run"))

import cv2
import numpy as np
import torch
import yaml
from torchvision.ops import box_iou

from run_camera_resnet_tf_detect import load_model as load_resnet_tf, detect as detect_resnet_tf
import run_camera_mobilenet_tf_detect  # noqa: F401 - side effect only: registers PyramidMobileNetBackbone
                                        # so keras.saving.load_model() can deserialize a MobileNet-TF checkpoint
from run_camera_mobilenetv4_detect import get_device as get_device_mobilenetv4, load_model as load_mobilenetv4, \
    detect as detect_mobilenetv4
from run_camera_nanodet_tf_detect import load_model as load_nanodet_tf, detect as detect_nanodet_tf
from run_camera_resnet_fasterrcnn_detect import get_device as get_device_resnet_fasterrcnn, \
    load_model as load_resnet_fasterrcnn, detect as detect_resnet_fasterrcnn
from run_camera_mobilenetv3_detect import get_device as get_device_mobilenetv3, load_model as load_mobilenetv3, \
    detect as detect_mobilenetv3

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def make_emitter(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"evaluate_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = open(log_path, "w")

    def emit(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    return emit, log_path


def get_device_yolo():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def backbone_family(backbone_name):
    if backbone_name.startswith("mobilenetv4"):
        return "mobilenetv4"
    raise ValueError(f"Unrecognized backbone '{backbone_name}' in checkpoint - expected a mobilenetv4_* name")


def load_yolo(path_str, conf):
    from ultralytics import YOLO
    try:
        yolo_model = YOLO(path_str)
    except Exception as e:
        raise ValueError(f"'{path_str}' looks like a YOLO checkpoint (filename "
                          f"starts with 'yolo') but failed to load: {e}") from e
    yolo_model.to(get_device_yolo())
    class_names = list(yolo_model.names.values())
    task = getattr(yolo_model, "task", "detect")

    def predictor(frame):
        result = yolo_model.predict(frame, conf=conf, verbose=False)[0]
        detections = []
        boxes = result.boxes
        for box, cls, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
            x1, y1, x2, y2 = map(int, box)
            detections.append((x1, y1, x2, y2, yolo_model.names[int(cls)], float(score)))
        return detections

    return f"yolo-{task}", class_names, predictor


def identify_and_load(model_path, conf, data_yaml):
    """Returns (family_label, model_class_names, predictor) where predictor(frame)
    -> list of (x1, y1, x2, y2, class_name, score) in original-frame pixel space,
    already filtered to score >= conf."""
    path_str = str(model_path)
    if Path(path_str).name.lower().startswith("yolo"):
        return load_yolo(path_str, conf)

    if path_str.endswith(".keras"):
        # load_model()/detect() are backbone-agnostic (KerasHub serializes the
        # whole RetinaNetObjectDetector, backbone included) - identical code
        # works for any TF/KerasHub RetinaNet checkpoint. NanoDet-TF is a
        # plain keras.Model instead (no .backbone/.preprocessor attrs, raw
        # per-level cls_scores/bbox_preds output) - detected by their absence.
        model, class_names = load_resnet_tf(path_str, data_yaml)
        if hasattr(model, "backbone") and hasattr(model, "preprocessor"):
            encoder_name = type(model.backbone.image_encoder).__name__
            if "ResNet" in encoder_name:
                family = "resnet-tf"
            elif "EfficientNet" in encoder_name:
                family = "efficientnet-tf"
            elif "DenseNet" in encoder_name:
                family = "densenet-tf"
            elif "MobileNet" in encoder_name:
                family = "mobilenet-tf"
            else:
                family = f"kerashub-{encoder_name.lower()}"
            predictor = lambda frame: detect_resnet_tf(model, class_names, frame, conf, 480, 640)
            return family, class_names, predictor
        else:
            img_size = model.input_shape[1]
            predictor = lambda frame: detect_nanodet_tf(model, class_names, img_size, frame, conf)
            return "nanodet-tf", class_names, predictor

    ckpt = None
    try:
        ckpt = torch.load(path_str, map_location="cpu", weights_only=False)
    except Exception:
        ckpt = None

    if isinstance(ckpt, dict) and "family" in ckpt:
        # Ready-made torchvision Faster R-CNN constructors (ResNet, MobileNetV3) -
        # the checkpoint names its own family explicitly rather than needing
        # inference from a backbone-name string.
        if ckpt["family"] == "resnet-fasterrcnn":
            device = get_device_resnet_fasterrcnn()
            model, class_names = load_resnet_fasterrcnn(path_str, device)
            predictor = lambda frame: detect_resnet_fasterrcnn(model, class_names, frame, device, conf)
            return "resnet-fasterrcnn", class_names, predictor
        if ckpt["family"] == "mobilenetv3-fasterrcnn":
            device = get_device_mobilenetv3()
            model, class_names = load_mobilenetv3(path_str, device)
            predictor = lambda frame: detect_mobilenetv3(model, class_names, frame, device, conf)
            return "mobilenetv3-fasterrcnn", class_names, predictor

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt and "backbone" in ckpt:
        family = backbone_family(ckpt["backbone"])
        device = get_device_mobilenetv4()
        model, class_names = load_mobilenetv4(path_str, device)
        predictor = lambda frame: detect_mobilenetv4(model, class_names, frame, device, conf)
        return family, class_names, predictor

    # Not YOLO-named, and not one of the remaining custom checkpoint dict shapes either.
    got = (f"a dict with keys {sorted(ckpt.keys())}" if isinstance(ckpt, dict)
           else "nothing (torch.load failed)" if ckpt is None
           else type(ckpt).__name__)
    raise ValueError(
        f"'{path_str}' doesn't match any recognized checkpoint shape - expected a "
        f"filename starting with 'yolo' (Ultralytics), a '.keras' file (ResNet/"
        f"EfficientNet RetinaNet or NanoDet, all TF), or a MobileNetV4/ResNet/"
        f"MobileNetV3 Faster R-CNN dict (a 'model_state_dict' key). Got: {got}."
    )


def load_ground_truth(label_path, img_w, img_h, class_names):
    gts = []
    if not label_path.exists():
        return gts
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls_id, cx, cy, w, h = line.split()
        cls_id = int(cls_id)
        cx, cy = float(cx) * img_w, float(cy) * img_h
        bw, bh = float(w) * img_w, float(h) * img_h
        x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        gts.append((x1, y1, x2, y2, class_names[cls_id]))
    return gts


def match_image(preds, gts, name_to_idx, iou_thres, matrix):
    """Same greedy-IoU-match convention as Ultralytics' ConfusionMatrix.process_batch:
    matched pairs -> matrix[pred_class, gt_class] += 1 (diagonal = TP, off-diagonal
    is simultaneously an FP for the predicted class and an FN for the true class);
    unmatched predictions -> matrix[pred_class, background] += 1 (FP); unmatched
    ground truth -> matrix[background, gt_class] += 1 (FN)."""
    nc = len(name_to_idx)

    if not gts:
        for pred in preds:
            matrix[name_to_idx[pred[4]], nc] += 1
        return
    if not preds:
        for gt in gts:
            matrix[nc, name_to_idx[gt[4]]] += 1
        return

    gt_boxes = torch.tensor([g[:4] for g in gts], dtype=torch.float32)
    pred_boxes = torch.tensor([p[:4] for p in preds], dtype=torch.float32)
    iou = box_iou(gt_boxes, pred_boxes).numpy()  # [num_gt, num_pred]

    gi_idx, pi_idx = np.where(iou > iou_thres)
    matched_gt, matched_pred = {}, set()
    if len(gi_idx):
        pairs = np.stack([gi_idx, pi_idx, iou[gi_idx, pi_idx]], axis=1)
        pairs = pairs[pairs[:, 2].argsort()[::-1]]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]  # dedupe by pred
        pairs = pairs[pairs[:, 2].argsort()[::-1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]  # dedupe by gt
        for gi, pi, _ in pairs:
            matched_gt[int(gi)] = int(pi)
            matched_pred.add(int(pi))

    for gi, gt in enumerate(gts):
        gt_cls = name_to_idx[gt[4]]
        if gi in matched_gt:
            pred_cls = name_to_idx[preds[matched_gt[gi]][4]]
            matrix[pred_cls, gt_cls] += 1
        else:
            matrix[nc, gt_cls] += 1

    for pi, pred in enumerate(preds):
        if pi not in matched_pred:
            matrix[name_to_idx[pred[4]], nc] += 1

# Calculations
def summarize(matrix, class_names):
    nc = len(class_names)
    diag = matrix.diagonal()[:nc]
    tp = diag
    fp = matrix[:nc, :].sum(axis=1) - diag
    fn = matrix[:, :nc].sum(axis=0) - diag
    rows = []
    for i, name in enumerate(class_names):
        p = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0 #precision
        r = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0 #recall 
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0 #F1 score
        rows.append((name, int(tp[i]), int(fp[i]), int(fn[i]), p, r, f1))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", nargs="+", required=True,
                         help="one or more checkpoints - YOLO, YOLO-pose, ResNet/EfficientNet "
                              "RetinaNet (TF), NanoDet (TF), or MobileNetV4/ResNet/MobileNetV3 "
                              "Faster R-CNN (PyTorch); auto-detected per file")
    parser.add_argument("--data", default="testcase/data.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--conf", type=float, default=0.5,
                         help="confidence threshold - also the confusion-matrix matching "
                              "threshold, matching every camera script's own deploy default")
    parser.add_argument("--iou", type=float, default=0.5,
                         help="IoU threshold for matching a prediction to a ground-truth box")
    parser.add_argument("--log-dir", default="logs",
                         help="directory to save a timestamped copy of this run's full report to "
                              "(same logs/<prefix>_<timestamp>.log convention as the camera scripts)")
    args = parser.parse_args()

    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"No dataset at '{args.data}'.")

    emit, log_path = make_emitter(args.log_dir)
    emit(f"Logging to {log_path}")

    names_dict = yaml.safe_load(data_yaml_path.read_text())["names"]
    class_names = [names_dict[i] for i in sorted(names_dict)]
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    nc = len(class_names)

    split_dir = data_yaml_path.parent / args.split
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    emit(f"Evaluating on {len(image_paths)} images from {images_dir} "
         f"(conf={args.conf}, iou={args.iou})")

    for model_path in args.model_path:
        if not Path(model_path).exists():
            emit(f"\n{'=' * 70}\nModel: {model_path}\n  SKIPPED - file not found")
            continue

        emit(f"\n{'=' * 70}\nModel: {model_path}")
        family, model_class_names, predictor = identify_and_load(model_path, args.conf, args.data)
        emit(f"Detected family: {family}  classes: {model_class_names}")

        matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            gts = load_ground_truth(labels_dir / f"{image_path.stem}.txt", w, h, class_names)
            preds = predictor(frame)
            match_image(preds, gts, name_to_idx, args.iou, matrix)

        rows = summarize(matrix, class_names)
        emit(f"\n{'Class':<10}{'TP':>6}{'FP':>6}{'FN':>6}{'Precision':>12}{'Recall':>10}{'F1':>8}")
        for name, tp, fp, fn, p, r, f1 in rows:
            emit(f"{name:<10}{tp:>6}{fp:>6}{fn:>6}{p:>12.3f}{r:>10.3f}{f1:>8.3f}")
        mean_p = sum(r[4] for r in rows) / len(rows)
        mean_r = sum(r[5] for r in rows) / len(rows)
        mean_f1 = sum(r[6] for r in rows) / len(rows)
        emit(f"{'mean':<10}{'':>6}{'':>6}{'':>6}{mean_p:>12.3f}{mean_r:>10.3f}{mean_f1:>8.3f}")

        emit("\nConfusion matrix (rows=predicted, cols=actual, last=background/unmatched):")
        col_labels = class_names + ["(none)"]
        emit("        " + "".join(f"{label[:9]:>10}" for label in col_labels))
        for i, label in enumerate(col_labels):
            row = "".join(f"{matrix[i, j]:>10}" for j in range(nc + 1))
            emit(f"{label[:8]:<8}{row}")

    emit(f"\nFull report saved to {log_path}")


if __name__ == "__main__":
    main()
