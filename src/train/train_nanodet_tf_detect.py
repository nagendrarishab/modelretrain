"""
    python src/train/train_nanodet_tf_detect.py --epochs 100

"""
import argparse
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
import torch
import yaml
from keras import layers
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_emitter(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"train_nanodet_tf_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = open(log_path, "w")

    def emit(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    return emit, log_path

STRIDES = (8, 16, 32)
FEAT_CHANNELS = 96
STACKED_CONVS = 2
REG_MAX = 7  # DFL: distance regressed as a distribution over 0..REG_MAX bins, matches NanoDet-Plus's reg_max
REG_LOSS_WEIGHT = 2.0
DFL_LOSS_WEIGHT = 0.25
EMA_DECAY = 0.9998
SIMOTA_CANDIDATE_TOPK = 10  # per-GT dynamic-k is derived from the sum of this many top IoUs

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


def pan_downsample(x, out_channels, name):
    """Stride-2 depthwise-separable downsample used by the neck's bottom-up
    (PAN) pass - same conv_bn building block the backbone/head already use,
    just spatially halving instead of preserving resolution."""
    x = conv_bn(x, out_channels, 3, 2, f"{name}_dw", groups=out_channels)
    return conv_bn(x, out_channels, 1, 1, f"{name}_pw")


def build_neck(feats, out_channels):
    """1x1 lateral convs to a common channel count, then a full PAN pass:
    top-down nearest-upsample-and-add (as before) followed by a bottom-up
    strided-downsample-and-add - the bidirectional fusion real NanoDet-Plus
    gets from its GhostPAN neck (its Ghost module is a cheap-conv efficiency
    trick specifically, so it's skipped here since the goal is accuracy)."""
    laterals = [layers.Conv2D(out_channels, 1, name=f"neck_lateral{i}")(f) for i, f in enumerate(feats)]
    for i in range(len(laterals) - 1, 0, -1):
        up = layers.Resizing(laterals[i - 1].shape[1], laterals[i - 1].shape[2],
                              interpolation="nearest", name=f"neck_upsample{i}")(laterals[i])
        laterals[i - 1] = layers.Add(name=f"neck_add{i}")([laterals[i - 1], up])
    for i in range(len(laterals) - 1):
        down = pan_downsample(laterals[i], out_channels, f"neck_down{i}")
        laterals[i + 1] = layers.Add(name=f"neck_pan_add{i}")([laterals[i + 1], down])
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


def build_head(feats, num_classes, feat_channels, stacked_convs, strides, reg_max):
    cls_convs = [DepthwiseSeparableConv(feat_channels, name=f"head_cls_conv{i}") for i in range(stacked_convs)]
    reg_convs = [DepthwiseSeparableConv(feat_channels, name=f"head_reg_conv{i}") for i in range(stacked_convs)]
    cls_pred = layers.Conv2D(num_classes, 3, padding="same", name="head_cls_pred")
    # 4*(reg_max+1): a discrete probability distribution over 0..reg_max per
    # side (left/top/right/bottom), decoded via integral_distribution_*() -
    # DFL, not a direct scalar regression.
    reg_pred = layers.Conv2D(4 * (reg_max + 1), 3, padding="same", name="head_reg_pred")

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


def build_model(num_classes, image_size, feat_channels=FEAT_CHANNELS, stacked_convs=STACKED_CONVS,
                 strides=STRIDES, reg_max=REG_MAX):
    image_input = layers.Input(shape=(image_size, image_size, 3), name="image")
    feats = build_shufflenet_v2_backbone(image_input)
    neck_feats = build_neck(feats, feat_channels)
    cls_scores, bbox_preds = build_head(neck_feats, num_classes, feat_channels, stacked_convs, strides, reg_max)
    model = keras.Model(inputs=image_input, outputs=cls_scores + bbox_preds, name="nanodet_tf")
    convert_shufflenet_weights(model)
    return model


def load_image(image_path, img_size):
    img = tf.io.decode_image(tf.io.read_file(str(image_path)), channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size), method="bilinear").numpy()
    img = (img / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return img.astype(np.float32)


def load_split(split_dir, img_size):
    # Returns image *paths*, not decoded arrays - decoding every image in a
    # split upfront (as this used to) holds the whole split as float32 in
    # memory at once (18287 train images at img_size=320 is ~22.5GB, more
    # than this repo's 16GB dev machine has - confirmed via an actual OOM
    # kill, exit code 137). Callers load_image() lazily per-batch instead
    # (see main()'s training loop and evaluate() below).
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    image_path_strings, all_boxes, all_labels = [], [], []
    for image_path in image_paths:
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

        image_path_strings.append(str(image_path))
        all_boxes.append(np.array(boxes, dtype=np.float32).reshape(-1, 4))
        all_labels.append(np.array(labels, dtype=np.int64))

    return image_path_strings, all_boxes, all_labels


def generate_points(feat_size, stride):
    h, w = feat_size
    ys = (np.arange(h, dtype=np.float32) + 0.5) * stride
    xs = (np.arange(w, dtype=np.float32) + 0.5) * stride
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)  # [H*W, 2]


def box_iou_np(boxes1, boxes2):
    """Vectorized IoU matrix between two sets of xyxy boxes -> [len(boxes1), len(boxes2)]."""
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area1 = np.maximum(boxes1[:, 2] - boxes1[:, 0], 0) * np.maximum(boxes1[:, 3] - boxes1[:, 1], 0)
    area2 = np.maximum(boxes2[:, 2] - boxes2[:, 0], 0) * np.maximum(boxes2[:, 3] - boxes2[:, 1], 0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-7)


def simota_assign_targets(points, gt_boxes, gt_labels, pred_scores, pred_boxes, num_classes,
                           candidate_topk=SIMOTA_CANDIDATE_TOPK, iou_cost_weight=3.0):
    """Dynamic soft label assignment (SimOTA-style, as used by NanoDet-Plus/
    YOLOX) - replaces the old FCOS nearest-area rule. Builds a per-image cost
    matrix (classification cost + IoU cost) between every ground-truth box
    and every candidate point (center inside that box), gives each GT its
    own dynamic top-k lowest-cost points (k from the sum of its top-`candidate_topk`
    IoUs, so bigger/better-matched GTs claim more points), then resolves any
    point claimed by more than one GT by lowest cost.

    pred_scores/pred_boxes are the model's own current predictions, already
    detached to plain numpy by the caller - no gradient needed here, same
    @torch.no_grad() convention the FCOS assigner this replaces used. The
    positive classification target is the matched box's IoU (a continuous
    0-1 quality score for the quality focal loss), not a hard 1.0.
    """
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

    if not inside_box.any():
        return cls_targets, reg_targets, pos_mask

    iou_matrix = box_iou_np(pred_boxes, gt_boxes)  # [P, N]
    cls_cost = -np.log(pred_scores[:, gt_labels] + 1e-8)  # [P, N]
    iou_cost = -np.log(iou_matrix + 1e-8)
    cost = cls_cost + iou_cost_weight * iou_cost
    cost[~inside_box] = 1e5

    k = min(candidate_topk, num_points)
    topk_ious = np.sort(iou_matrix, axis=0)[::-1][:k]  # [k, N]
    dynamic_ks = np.clip(topk_ious.sum(axis=0).astype(int), 1, None)  # [N]

    num_gt = gt_boxes.shape[0]
    matching_matrix = np.zeros((num_points, num_gt), dtype=bool)
    for n in range(num_gt):
        valid = np.nonzero(inside_box[:, n])[0]
        if valid.size == 0:
            continue
        n_select = min(int(dynamic_ks[n]), valid.size)
        chosen = valid[np.argsort(cost[valid, n])[:n_select]]
        matching_matrix[chosen, n] = True

    multi_match = matching_matrix.sum(axis=1) > 1
    for p in np.nonzero(multi_match)[0]:
        matched_gts = np.nonzero(matching_matrix[p])[0]
        best = matched_gts[np.argmin(cost[p, matched_gts])]
        matching_matrix[p, :] = False
        matching_matrix[p, best] = True

    pos_idx = np.nonzero(matching_matrix.any(axis=1))[0]
    if pos_idx.size == 0:
        return cls_targets, reg_targets, pos_mask
    assigned_gt = matching_matrix[pos_idx].argmax(axis=1)

    pos_mask[pos_idx] = True
    cls_targets[pos_idx, gt_labels[assigned_gt]] = iou_matrix[pos_idx, assigned_gt]
    reg_targets[pos_idx] = ltrb[pos_idx, assigned_gt]

    return cls_targets, reg_targets, pos_mask


def integral_distribution_np(dist_logits, reg_max):
    """Numpy counterpart of integral_distribution_tf, for inference/eval
    decode outside the training graph - see that function's docstring."""
    dist_logits = dist_logits.reshape(-1, 4, reg_max + 1)
    exps = np.exp(dist_logits - dist_logits.max(axis=-1, keepdims=True))
    probs = exps / exps.sum(axis=-1, keepdims=True)
    bins = np.arange(reg_max + 1, dtype=np.float32)
    return (probs * bins).sum(axis=-1)  # [-1, 4], distance in stride units


def integral_distribution_tf(dist_logits, reg_max):
    """Softmax-weighted integral over the reg_max+1 discrete distance bins
    (Distribution Focal Loss decode): converts each side's (left/top/right/
    bottom) probability distribution over integer distances into one
    continuous distance, in stride units - differentiable, used inside
    compute_loss's GIoU path in place of the old direct exp() regression."""
    dist_logits = keras.ops.reshape(dist_logits, (-1, 4, reg_max + 1))
    probs = keras.ops.softmax(dist_logits, axis=-1)
    bins = keras.ops.arange(reg_max + 1, dtype="float32")
    return keras.ops.sum(probs * bins, axis=-1)  # [-1, 4]


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


def quality_focal_loss(cls_target, cls_logits, beta=2.0):
    """Quality Focal Loss (Generalized Focal Loss): sigmoid binary
    cross-entropy scaled by |target - sigmoid(logit)|**beta, so confident-
    and-wrong predictions are penalized hardest. `cls_target` is the
    continuous IoU quality score from simota_assign_targets (0 at negatives),
    not a hard 0/1 label - this is what ties classification confidence to
    localization quality. Sum reduction, same convention as the
    BinaryFocalCrossentropy it replaces."""
    pred_sigmoid = keras.ops.sigmoid(cls_logits)
    scale_factor = keras.ops.abs(cls_target - pred_sigmoid) ** beta
    bce = keras.ops.binary_crossentropy(cls_target, cls_logits, from_logits=True)
    return keras.ops.sum(bce * scale_factor)


def distribution_focal_loss(pred_logits, target_dist_units, reg_max):
    """Distribution Focal Loss: cross-entropy against the two integer bins
    straddling the (fractional) target distance, weighted by proximity -
    trains the per-side distribution itself to be sharp and accurate, on top
    of the GIoU loss on the decoded box. `target_dist_units` is in stride
    units (pre-multiply-by-stride), matching integral_distribution_*()'s
    output space. Sum reduction, matching the file's other losses."""
    pred_logits = keras.ops.reshape(pred_logits, (-1, 4, reg_max + 1))
    target_dist_units = np.clip(target_dist_units, 0.0, float(reg_max) - 1e-3)
    left = np.floor(target_dist_units).astype(np.int64)
    right = left + 1
    weight_left = right.astype(np.float32) - target_dist_units
    weight_right = target_dist_units - left.astype(np.float32)

    log_probs = pred_logits - keras.ops.logsumexp(pred_logits, axis=-1, keepdims=True)
    ce_left = -keras.ops.take_along_axis(log_probs, left[..., None], axis=-1)[..., 0]
    ce_right = -keras.ops.take_along_axis(log_probs, right[..., None], axis=-1)[..., 0]
    loss = ce_left * weight_left + ce_right * weight_right  # [num_pos, 4]
    return keras.ops.sum(loss)


def compute_loss(cls_scores, bbox_preds, batch_boxes, batch_labels, strides, reg_max, num_classes):
    batch_size = cls_scores[0].shape[0]

    points_per_level = []
    for cls_score, stride in zip(cls_scores, strides):
        h, w = cls_score.shape[1], cls_score.shape[2]
        points_per_level.append(generate_points((h, w), stride))
    all_points = np.concatenate(points_per_level, axis=0)
    all_strides = np.concatenate([
        np.full((p.shape[0],), s, dtype=np.float32) for p, s in zip(points_per_level, strides)
    ])

    flat_cls = keras.ops.concatenate(
        [keras.ops.reshape(cs, (batch_size, -1, num_classes)) for cs in cls_scores], axis=1,
    )  # [B, P, C]
    flat_reg = keras.ops.concatenate(
        [keras.ops.reshape(bp, (batch_size, -1, 4 * (reg_max + 1))) for bp in bbox_preds], axis=1,
    )  # [B, P, 4*(reg_max+1)] raw per-side distribution logits

    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_dfl_loss = 0.0
    total_pos = 0
    for b in range(batch_size):
        # Detached (numpy) copies of this image's own current predictions,
        # used only to compute the SimOTA assignment cost - no gradient
        # flows through the assignment itself, same as the FCOS assigner
        # this replaces.
        pred_scores_np = np.array(keras.ops.sigmoid(flat_cls[b]))
        pred_dist_units_np = integral_distribution_np(np.array(flat_reg[b]), reg_max)
        pred_boxes_np = np.stack([
            all_points[:, 0] - pred_dist_units_np[:, 0] * all_strides,
            all_points[:, 1] - pred_dist_units_np[:, 1] * all_strides,
            all_points[:, 0] + pred_dist_units_np[:, 2] * all_strides,
            all_points[:, 1] + pred_dist_units_np[:, 3] * all_strides,
        ], axis=-1)

        cls_target, reg_target, pos_mask = simota_assign_targets(
            all_points, batch_boxes[b], batch_labels[b], pred_scores_np, pred_boxes_np, num_classes,
        )
        total_cls_loss = total_cls_loss + quality_focal_loss(cls_target, flat_cls[b])

        num_pos = int(pos_mask.sum())
        total_pos += num_pos
        if num_pos == 0:
            continue

        pos_idx = np.nonzero(pos_mask)[0]
        pts = all_points[pos_idx]
        strides_pos = all_strides[pos_idx]
        pred_logits_pos = keras.ops.take(flat_reg[b], pos_idx, axis=0)  # [num_pos, 4*(reg_max+1)]
        pred_dist_units = integral_distribution_tf(pred_logits_pos, reg_max)  # [num_pos, 4], differentiable
        pred_dist = pred_dist_units * strides_pos[:, None]
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

        target_dist_units = target_dist / strides_pos[:, None]
        total_dfl_loss = total_dfl_loss + distribution_focal_loss(pred_logits_pos, target_dist_units, reg_max)

    num_pos = max(total_pos, 1)
    return total_cls_loss / num_pos, total_reg_loss / num_pos, total_dfl_loss / num_pos


def decode_single(cls_scores, bbox_preds, strides, img_size, reg_max,
                   score_thresh=EVAL_SCORE_THRESH, nms_iou=NMS_IOU_THRESH, max_dets=MAX_DETECTIONS):
    """Single-image decode: raw per-level (h,w,C)/(h,w,4*(reg_max+1)) arrays
    -> boxes/scores/labels. Runs entirely outside any traced graph - same
    post-processing-not-part-of-forward() design as the PyTorch trainer's
    decode_single()."""
    all_boxes, all_scores, all_labels = [], [], []
    for cls_score, bbox_pred, stride in zip(cls_scores, bbox_preds, strides):
        h, w = cls_score.shape[0], cls_score.shape[1]
        points = generate_points((h, w), stride)
        scores = 1.0 / (1.0 + np.exp(-cls_score.reshape(-1, cls_score.shape[-1])))
        dist_units = integral_distribution_np(bbox_pred.reshape(-1, 4 * (reg_max + 1)), reg_max)
        dist = dist_units * stride

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


def evaluate(model, image_paths, batch_boxes, batch_labels, img_size, reg_max, batch_size=16):
    metric = MeanAveragePrecision(box_format="xyxy")
    for start in range(0, len(image_paths), batch_size):
        batch = np.stack([load_image(p, img_size) for p in image_paths[start:start + batch_size]])
        outputs = model(batch, training=False)
        cls_scores, bbox_preds = outputs[:3], outputs[3:]
        cls_scores = [np.array(c) for c in cls_scores]
        bbox_preds = [np.array(b) for b in bbox_preds]

        preds, targets = [], []
        for i in range(batch.shape[0]):
            boxes, scores, labels = decode_single(
                [c[i] for c in cls_scores], [b[i] for b in bbox_preds], STRIDES, img_size, reg_max,
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


class EMA:
    """Exponential moving average of trainable variables (NanoDet-Plus's
    ExpMovingAverager). update() runs after every optimizer step;
    swap_in()/swap_out() temporarily install the shadow weights for
    eval+checkpointing, then restore the live (non-EMA) weights so training
    continues on those, not the average."""

    def __init__(self, variables, decay):
        self.decay = decay
        self.shadow = [tf.Variable(v, trainable=False) for v in variables]
        self.backup = None

    def update(self, variables):
        for s, v in zip(self.shadow, variables):
            s.assign(self.decay * s + (1.0 - self.decay) * v)

    def swap_in(self, variables):
        self.backup = [tf.Variable(v) for v in variables]
        for v, s in zip(variables, self.shadow):
            v.assign(s)

    def swap_out(self, variables):
        for v, b in zip(variables, self.backup):
            v.assign(b)
        self.backup = None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data_detect/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg-max", type=int, default=REG_MAX,
                         help="DFL: number of distance bins per side is reg_max+1")
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY,
                         help="exponential moving average decay for eval/checkpoint weights")
    parser.add_argument("--output", default="models/nanodet_tf_best.keras")
    parser.add_argument("--log-dir", default="logs", help="Directory for this run's log file")
    args = parser.parse_args()

    emit, log_path = make_emitter(args.log_dir)
    emit(f"Logging to {log_path}")

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

    emit("Loading dataset ...")
    train_images, train_boxes, train_labels = load_split(data_root / "train", args.img_size)
    val_images, val_boxes, val_labels = load_split(data_root / "val", args.img_size)
    emit(f"Train: {len(train_images)} images  Val: {len(val_images)} images")

    model = build_model(num_classes, args.img_size, reg_max=args.reg_max)
    optimizer = keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, weight_decay=0.0001)
    ema = EMA(model.trainable_variables, args.ema_decay)
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
        progress = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for bi in progress:
            idx = perm[bi * args.batch_size:(bi + 1) * args.batch_size]
            if len(idx) == 0:
                continue
            batch_images = np.stack([load_image(train_images[i], args.img_size) for i in idx])
            batch_boxes = [train_boxes[i] for i in idx]
            batch_labels = [train_labels[i] for i in idx]

            with tf.GradientTape() as tape:
                outputs = model(batch_images, training=True)
                cls_scores, bbox_preds = outputs[:3], outputs[3:]
                cls_loss, reg_loss, dfl_loss = compute_loss(
                    cls_scores, bbox_preds, batch_boxes, batch_labels, STRIDES, args.reg_max, num_classes,
                )
                loss = cls_loss + REG_LOSS_WEIGHT * reg_loss + DFL_LOSS_WEIGHT * dfl_loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            ema.update(model.trainable_variables)
            epoch_loss += float(loss)
            progress.set_postfix(loss=f"{float(loss):.4f}")

        ema.swap_in(model.trainable_variables)
        map50, map5095 = evaluate(model, val_images, val_boxes, val_labels, args.img_size, args.reg_max)
        emit(f"Epoch {epoch}/{args.epochs}  loss={epoch_loss / num_batches:.4f}  "
             f"val mAP50={map50:.4f}  val mAP50-95={map5095:.4f}")

        if map50 > best_map50:
            best_map50 = map50
            model.save(output_path)
        ema.swap_out(model.trainable_variables)

    emit(f"\nSaved best model to {output_path} (val mAP50={best_map50:.4f})")
    emit(f"Class names ({num_classes}): {class_names}")
    emit(
        "\nRun 'python src/evaluate_models.py --model-path "
        f"{output_path}' for precision/recall/F1 on the test split."
    )


if __name__ == "__main__":
    main()
