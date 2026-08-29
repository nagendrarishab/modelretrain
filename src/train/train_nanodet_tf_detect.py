"""
    python src/train/train_nanodet_tf_detect.py --epochs 100

"""
import argparse
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
import torch
import yaml
from keras import layers
from torchmetrics.detection.mean_ap import MeanAveragePrecision

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

STRIDES = (8, 16, 32)
FEAT_CHANNELS = 96
STACKED_CONVS = 2
BACKBONE_CHANNELS = (116, 232, 464)
REGRESS_RANGES = ((0, 64), (64, 128), (128, float("inf")))
REG_LOSS_WEIGHT = 2.0

EVAL_SCORE_THRESH = 0.05
NMS_IOU_THRESH = 0.6
MAX_DETECTIONS = 100

BN_EPS = 1e-5  # matches torch.nn.BatchNorm2d's default, needed for weight-port fidelity


@keras.saving.register_keras_serializable(package="nanodet_tf")
class ChannelShuffle(keras.layers.Layer):
    """Same channel shuffle as torchvision's shufflenetv2.channel_shuffle,
    adapted from NCHW to Keras's NHWC: split channels into `groups` groups,
    transpose group/within-group axes, flatten back - lets information mix
    across the two branches concatenated just before this op.
    """

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


def conv_bn(x, filters, kernel_size, strides, name, relu=True, groups=1):
    if kernel_size > 1:
        x = layers.ZeroPadding2D(kernel_size // 2, name=f"{name}_pad")(x)
    if groups > 1:
        x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding="valid",
                                    use_bias=False, name=f"{name}_dwconv")(x)
    else:
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding="valid",
                           use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(epsilon=BN_EPS, name=f"{name}_bn")(x)
    if relu:
        x = layers.Activation("relu", name=f"{name}_relu")(x)
    return x


def inverted_residual(x, inp, oup, stride, name):
    branch_features = oup // 2
    if stride == 1:
        x1 = x[:, :, :, :branch_features]
        x2 = x[:, :, :, branch_features:]
        b2 = conv_bn(x2, branch_features, 1, 1, f"{name}_b2_pw1")
        b2 = conv_bn(b2, branch_features, 3, 1, f"{name}_b2_dw", relu=False, groups=branch_features)
        b2 = conv_bn(b2, branch_features, 1, 1, f"{name}_b2_pw2")
        out = layers.Concatenate(name=f"{name}_concat")([x1, b2])
    else:
        b1 = conv_bn(x, inp, 3, stride, f"{name}_b1_dw", relu=False, groups=inp)
        b1 = conv_bn(b1, branch_features, 1, 1, f"{name}_b1_pw")
        b2 = conv_bn(x, branch_features, 1, 1, f"{name}_b2_pw1")
        b2 = conv_bn(b2, branch_features, 3, stride, f"{name}_b2_dw", relu=False, groups=branch_features)
        b2 = conv_bn(b2, branch_features, 1, 1, f"{name}_b2_pw2")
        out = layers.Concatenate(name=f"{name}_concat")([b1, b2])
    return ChannelShuffle(2, name=f"{name}_shuffle")(out)


def build_shufflenet_v2_backbone(image_input):
    """torchvision's shufflenet_v2_x1_0 up through stage4 (no conv5/fc -
    those are classification-head-only and unused for detection)."""
    stages_repeats = [4, 8, 4]
    stages_out_channels = [24, 116, 232, 464]

    x = conv_bn(image_input, stages_out_channels[0], 3, 2, "conv1")
    x = layers.ZeroPadding2D(1, name="maxpool_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="valid", name="maxpool")(x)

    input_channels = stages_out_channels[0]
    pyramid = []
    for stage_idx, (repeats, out_channels) in enumerate(zip(stages_repeats, stages_out_channels[1:]), start=2):
        x = inverted_residual(x, input_channels, out_channels, 2, f"stage{stage_idx}_0")
        for i in range(repeats - 1):
            x = inverted_residual(x, out_channels, out_channels, 1, f"stage{stage_idx}_{i + 1}")
        input_channels = out_channels
        pyramid.append(x)
    return pyramid  # [c3 (stride 8), c4 (stride 16), c5 (stride 32)]


def convert_shufflenet_weights(keras_model):
    """Loads torchvision's ImageNet-pretrained shufflenet_v2_x1_0 weights
    into the from-scratch Keras port above, layer by layer. Verified
    bit-for-bit (max abs diff ~1e-6 on a random input) against the PyTorch
    model's own stage2/3/4 outputs before this was used for anything else -
    see the module docstring.
    """
    from torchvision.models import ShuffleNet_V2_X1_0_Weights, shufflenet_v2_x1_0
    torch_model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    sd = torch_model.state_dict()

    def conv_w(k):
        return np.transpose(sd[k].numpy(), (2, 3, 1, 0))  # (out,in,kh,kw) -> (kh,kw,in,out)

    def dwconv_w(k):
        return np.transpose(sd[k].numpy(), (2, 3, 0, 1))  # (out=c,1,kh,kw) -> (kh,kw,c,1)

    def bn_w(prefix):
        return [sd[f"{prefix}.weight"].numpy(), sd[f"{prefix}.bias"].numpy(),
                sd[f"{prefix}.running_mean"].numpy(), sd[f"{prefix}.running_var"].numpy()]

    keras_model.get_layer("conv1_conv").set_weights([conv_w("conv1.0.weight")])
    keras_model.get_layer("conv1_bn").set_weights(bn_w("conv1.1"))

    for stage_idx, repeats in zip([2, 3, 4], [4, 8, 4]):
        for i in range(repeats):
            name = f"stage{stage_idx}_{i}"
            tkey = f"stage{stage_idx}.{i}"
            if i == 0:
                keras_model.get_layer(f"{name}_b1_dw_dwconv").set_weights([dwconv_w(f"{tkey}.branch1.0.weight")])
                keras_model.get_layer(f"{name}_b1_dw_bn").set_weights(bn_w(f"{tkey}.branch1.1"))
                keras_model.get_layer(f"{name}_b1_pw_conv").set_weights([conv_w(f"{tkey}.branch1.2.weight")])
                keras_model.get_layer(f"{name}_b1_pw_bn").set_weights(bn_w(f"{tkey}.branch1.3"))
            keras_model.get_layer(f"{name}_b2_pw1_conv").set_weights([conv_w(f"{tkey}.branch2.0.weight")])
            keras_model.get_layer(f"{name}_b2_pw1_bn").set_weights(bn_w(f"{tkey}.branch2.1"))
            keras_model.get_layer(f"{name}_b2_dw_dwconv").set_weights([dwconv_w(f"{tkey}.branch2.3.weight")])
            keras_model.get_layer(f"{name}_b2_dw_bn").set_weights(bn_w(f"{tkey}.branch2.4"))
            keras_model.get_layer(f"{name}_b2_pw2_conv").set_weights([conv_w(f"{tkey}.branch2.5.weight")])
            keras_model.get_layer(f"{name}_b2_pw2_bn").set_weights(bn_w(f"{tkey}.branch2.6"))


def build_neck(feats, out_channels):
    """1x1 lateral convs to a common channel count + top-down
    nearest-upsample-and-add - same simplified (no bottom-up second pass)
    neck as the PyTorch NanoDetNeck."""
    laterals = [layers.Conv2D(out_channels, 1, name=f"neck_lateral{i}")(f) for i, f in enumerate(feats)]
    for i in range(len(laterals) - 1, 0, -1):
        target_hw = keras.ops.shape(laterals[i - 1])[1:3]
        up = layers.Resizing(laterals[i - 1].shape[1], laterals[i - 1].shape[2],
                              interpolation="nearest", name=f"neck_upsample{i}")(laterals[i])
        laterals[i - 1] = layers.Add(name=f"neck_add{i}")([laterals[i - 1], up])
    return laterals


@keras.saving.register_keras_serializable(package="nanodet_tf")
class DepthwiseSeparableConv(keras.layers.Layer):
    """Depthwise 3x3 + pointwise 1x1 + BN + ReLU - matches the PyTorch
    head's per-conv block exactly, as its own Layer (not a function) so the
    same weights are reused across all 3 pyramid levels when this one
    instance is called 3 times, matching NanoDetHead's weight sharing."""

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.pad = layers.ZeroPadding2D(1)
        self.dw = layers.DepthwiseConv2D(3, padding="valid", use_bias=False)
        self.pw = layers.Conv2D(channels, 1, use_bias=False)
        self.bn = layers.BatchNormalization(epsilon=BN_EPS)
        self.act = layers.Activation("relu")

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


@keras.saving.register_keras_serializable(package="nanodet_tf")
class Scale(keras.layers.Layer):
    """Learnable per-level scalar multiplier on the regression output -
    same as the PyTorch head's Scale module."""

    def build(self, input_shape):
        self.scale = self.add_weight(name="scale", shape=(), initializer="ones", trainable=True)

    def call(self, x):
        return x * self.scale


def build_head(feats, num_classes, feat_channels, stacked_convs, strides):
    cls_convs = [DepthwiseSeparableConv(feat_channels, name=f"head_cls_conv{i}") for i in range(stacked_convs)]
    reg_convs = [DepthwiseSeparableConv(feat_channels, name=f"head_reg_conv{i}") for i in range(stacked_convs)]
    cls_pred = layers.Conv2D(num_classes, 3, padding="same", name="head_cls_pred")
    reg_pred = layers.Conv2D(4, 3, padding="same", name="head_reg_pred")

    cls_scores, bbox_preds = [], []
    for level, (feat, stride) in enumerate(zip(feats, strides)):
        cls_feat = feat
        for conv in cls_convs:
            cls_feat = conv(cls_feat)
        reg_feat = feat
        for conv in reg_convs:
            reg_feat = conv(reg_feat)
        cls_scores.append(cls_pred(cls_feat))
        bbox_preds.append(Scale(name=f"head_scale{level}")(reg_pred(reg_feat)))
    return cls_scores, bbox_preds


def build_model(num_classes, image_size, feat_channels=FEAT_CHANNELS, stacked_convs=STACKED_CONVS, strides=STRIDES):
    image_input = layers.Input(shape=(image_size, image_size, 3), name="image")
    feats = build_shufflenet_v2_backbone(image_input)
    neck_feats = build_neck(feats, feat_channels)
    cls_scores, bbox_preds = build_head(neck_feats, num_classes, feat_channels, stacked_convs, strides)
    model = keras.Model(inputs=image_input, outputs=cls_scores + bbox_preds, name="nanodet_tf")
    convert_shufflenet_weights(model)
    return model


def load_split(split_dir, img_size):
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    images, all_boxes, all_labels = [], [], []
    for image_path in image_paths:
        img = tf.io.decode_image(tf.io.read_file(str(image_path)), channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size), method="bilinear").numpy()
        img = (img / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        boxes, labels = [], []
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                cls_id, cx, cy, bw, bh = map(float, line.split())
                x1 = (cx - bw / 2) * img_size
                y1 = (cy - bh / 2) * img_size
                x2 = (cx + bw / 2) * img_size
                y2 = (cy + bh / 2) * img_size
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls_id))

        images.append(img.astype(np.float32))
        all_boxes.append(np.array(boxes, dtype=np.float32).reshape(-1, 4))
        all_labels.append(np.array(labels, dtype=np.int64))

    return images, all_boxes, all_labels


def generate_points(feat_size, stride):
    h, w = feat_size
    ys = (np.arange(h, dtype=np.float32) + 0.5) * stride
    xs = (np.arange(w, dtype=np.float32) + 0.5) * stride
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)  # [H*W, 2]


def assign_targets(points, ranges_per_point, gt_boxes, gt_labels, num_classes):
    """Pure-numpy port of the PyTorch trainer's FCOS-style assignment - no
    gradient needed here (matches its @torch.no_grad()), since this only
    depends on fixed anchor points and ground-truth labels, never on model
    output."""
    num_points = points.shape[0]
    cls_targets = np.zeros((num_points, num_classes), dtype=np.float32)
    reg_targets = np.zeros((num_points, 4), dtype=np.float32)
    pos_mask = np.zeros(num_points, dtype=bool)

    if gt_boxes.shape[0] == 0:
        return cls_targets, reg_targets, pos_mask

    xs, ys = points[:, 0:1], points[:, 1:2]  # [P, 1]
    x1, y1, x2, y2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]  # [N]

    ltrb = np.stack([
        xs - x1[None, :], ys - y1[None, :], x2[None, :] - xs, y2[None, :] - ys,
    ], axis=-1)  # [P, N, 4]

    inside_box = ltrb.min(axis=-1) > 0  # [P, N]
    max_ltrb = ltrb.max(axis=-1)  # [P, N]
    low, high = ranges_per_point[:, 0:1], ranges_per_point[:, 1:2]
    inside_range = (max_ltrb >= low) & (max_ltrb <= high)

    areas = (x2 - x1) * (y2 - y1)  # [N]
    candidate_areas = np.tile(areas[None, :], (num_points, 1)).copy()
    candidate_areas[~(inside_box & inside_range)] = np.inf

    min_idx = candidate_areas.argmin(axis=1)
    min_area = candidate_areas[np.arange(num_points), min_idx]
    pos_mask = min_area < np.inf

    pos_idx = np.nonzero(pos_mask)[0]
    assigned_gt = min_idx[pos_idx]
    cls_targets[pos_idx, gt_labels[assigned_gt]] = 1.0
    reg_targets[pos_idx] = ltrb[pos_idx, assigned_gt]

    return cls_targets, reg_targets, pos_mask


def generalized_box_iou_loss(pred_boxes, target_boxes):
    """1 - GIoU per box, summed by the caller - same convention as
    torchvision.ops.generalized_box_iou_loss(reduction='sum')."""
    px1, py1, px2, py2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    tx1, ty1, tx2, ty2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

    ix1, iy1 = tf.maximum(px1, tx1), tf.maximum(py1, ty1)
    ix2, iy2 = tf.minimum(px2, tx2), tf.minimum(py2, ty2)
    inter = tf.maximum(ix2 - ix1, 0.0) * tf.maximum(iy2 - iy1, 0.0)
    area_p = tf.maximum(px2 - px1, 0.0) * tf.maximum(py2 - py1, 0.0)
    area_t = tf.maximum(tx2 - tx1, 0.0) * tf.maximum(ty2 - ty1, 0.0)
    union = area_p + area_t - inter
    iou = inter / tf.maximum(union, 1e-7)

    ex1, ey1 = tf.minimum(px1, tx1), tf.minimum(py1, ty1)
    ex2, ey2 = tf.maximum(px2, tx2), tf.maximum(py2, ty2)
    area_c = tf.maximum(ex2 - ex1, 0.0) * tf.maximum(ey2 - ey1, 0.0)
    giou = iou - (area_c - union) / tf.maximum(area_c, 1e-7)
    return tf.reduce_sum(1.0 - giou)


FOCAL_LOSS = keras.losses.BinaryFocalCrossentropy(
    apply_class_balancing=True, alpha=0.25, gamma=2.0, from_logits=True, reduction="sum",
)


def compute_loss(cls_scores, bbox_preds, batch_boxes, batch_labels, strides, regress_ranges, num_classes):
    batch_size = cls_scores[0].shape[0]

    points_per_level, ranges_per_level = [], []
    for cls_score, stride, rng in zip(cls_scores, strides, regress_ranges):
        h, w = cls_score.shape[1], cls_score.shape[2]
        points = generate_points((h, w), stride)
        points_per_level.append(points)
        ranges_per_level.append(np.tile(np.array(rng, dtype=np.float32), (points.shape[0], 1)))
    all_points = np.concatenate(points_per_level, axis=0)
    all_ranges = np.concatenate(ranges_per_level, axis=0)
    all_strides = np.concatenate([
        np.full((p.shape[0],), s, dtype=np.float32) for p, s in zip(points_per_level, strides)
    ])

    flat_cls = keras.ops.concatenate(
        [keras.ops.reshape(cs, (batch_size, -1, num_classes)) for cs in cls_scores], axis=1,
    )  # [B, P, C]
    flat_reg = keras.ops.concatenate(
        [keras.ops.reshape(bp, (batch_size, -1, 4)) for bp in bbox_preds], axis=1,
    )  # [B, P, 4] (pre-exp, pre-stride)

    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_pos = 0
    for b in range(batch_size):
        cls_target, reg_target, pos_mask = assign_targets(
            all_points, all_ranges, batch_boxes[b], batch_labels[b], num_classes,
        )
        total_cls_loss = total_cls_loss + FOCAL_LOSS(cls_target, flat_cls[b])

        num_pos = int(pos_mask.sum())
        total_pos += num_pos
        if num_pos == 0:
            continue

        pos_idx = np.nonzero(pos_mask)[0]
        pts = all_points[pos_idx]
        strides_pos = all_strides[pos_idx]
        pred_dist = keras.ops.exp(keras.ops.take(flat_reg[b], pos_idx, axis=0)) * strides_pos[:, None]
        pred_boxes = keras.ops.stack([
            pts[:, 0] - pred_dist[:, 0], pts[:, 1] - pred_dist[:, 1],
            pts[:, 0] + pred_dist[:, 2], pts[:, 1] + pred_dist[:, 3],
        ], axis=-1)
        target_dist = reg_target[pos_idx]
        target_boxes = np.stack([
            pts[:, 0] - target_dist[:, 0], pts[:, 1] - target_dist[:, 1],
            pts[:, 0] + target_dist[:, 2], pts[:, 1] + target_dist[:, 3],
        ], axis=-1)
        total_reg_loss = total_reg_loss + generalized_box_iou_loss(pred_boxes, target_boxes)

    num_pos = max(total_pos, 1)
    return total_cls_loss / num_pos, total_reg_loss / num_pos


def decode_single(cls_scores, bbox_preds, strides, img_size,
                   score_thresh=EVAL_SCORE_THRESH, nms_iou=NMS_IOU_THRESH, max_dets=MAX_DETECTIONS):
    """Single-image decode: raw per-level (h,w,C)/(h,w,4) arrays -> boxes/scores/labels.
    Runs entirely outside any traced graph - same post-processing-not-part-
    of-forward() design as the PyTorch trainer's decode_single()."""
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


def evaluate(model, images, batch_boxes, batch_labels, img_size, batch_size=16):
    metric = MeanAveragePrecision(box_format="xyxy")
    for start in range(0, len(images), batch_size):
        batch = np.stack(images[start:start + batch_size])
        outputs = model(batch, training=False)
        cls_scores, bbox_preds = outputs[:3], outputs[3:]
        cls_scores = [np.array(c) for c in cls_scores]
        bbox_preds = [np.array(b) for b in bbox_preds]

        preds, targets = [], []
        for i in range(batch.shape[0]):
            boxes, scores, labels = decode_single(
                [c[i] for c in cls_scores], [b[i] for b in bbox_preds], STRIDES, img_size,
            )
            preds.append({
                "boxes": torch.from_numpy(boxes), "scores": torch.from_numpy(scores),
                "labels": torch.from_numpy(labels),
            })
            gt_boxes = batch_boxes[start + i]
            gt_labels = batch_labels[start + i]
            targets.append({
                "boxes": torch.from_numpy(gt_boxes), "labels": torch.from_numpy(gt_labels),
            })
        metric.update(preds, targets)
    result = metric.compute()
    return float(result["map_50"]), float(result["map"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data_detect/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output", default="models/nanodet_tf_best.keras")
    args = parser.parse_args()

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

    print("Loading dataset ...")
    train_images, train_boxes, train_labels = load_split(data_root / "train", args.img_size)
    val_images, val_boxes, val_labels = load_split(data_root / "val", args.img_size)
    print(f"Train: {len(train_images)} images  Val: {len(val_images)} images")

    model = build_model(num_classes, args.img_size)
    optimizer = keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, weight_decay=0.0001)
    step_size = max(args.epochs // 3, 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_map50 = -1.0
    n = len(train_images)

    for epoch in range(1, args.epochs + 1):
        optimizer.learning_rate = args.lr * (0.1 ** (((epoch - 1) // step_size)))

        perm = np.random.permutation(n)
        epoch_loss = 0.0
        num_batches = max(n // args.batch_size, 1)
        for bi in range(num_batches):
            idx = perm[bi * args.batch_size:(bi + 1) * args.batch_size]
            if len(idx) == 0:
                continue
            batch_images = np.stack([train_images[i] for i in idx])
            batch_boxes = [train_boxes[i] for i in idx]
            batch_labels = [train_labels[i] for i in idx]

            with tf.GradientTape() as tape:
                outputs = model(batch_images, training=True)
                cls_scores, bbox_preds = outputs[:3], outputs[3:]
                cls_loss, reg_loss = compute_loss(
                    cls_scores, bbox_preds, batch_boxes, batch_labels, STRIDES, REGRESS_RANGES, num_classes,
                )
                loss = cls_loss + REG_LOSS_WEIGHT * reg_loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += float(loss)

        map50, map5095 = evaluate(model, val_images, val_boxes, val_labels, args.img_size)
        print(f"Epoch {epoch}/{args.epochs}  loss={epoch_loss / num_batches:.4f}  "
              f"val mAP50={map50:.4f}  val mAP50-95={map5095:.4f}")

        if map50 > best_map50:
            best_map50 = map50
            model.save(output_path)

    print(f"\nSaved best model to {output_path} (val mAP50={best_map50:.4f})")
    print(f"Class names ({num_classes}): {class_names}")
    print(
        "\nRun 'python src/evaluate_models.py --model-path "
        f"{output_path}' for precision/recall/F1 on the test split."
    )


if __name__ == "__main__":
    main()
