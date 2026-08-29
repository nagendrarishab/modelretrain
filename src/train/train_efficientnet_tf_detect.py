"""
    python src/train/train_efficientnet_tf_detect.py --epochs 30

TensorFlow/KerasHub replacement for the old torchvision Faster R-CNN
EfficientNet trainer: a RetinaNet (single-stage, static-shape) detector on
an EfficientNet-B0 backbone, trained from an ImageNet-only pretrained
preset (no COCO detection head) - the same transfer-learning setup the old
trainer used (efficientnet_b0(weights="IMAGENET1K_V1") + a from-scratch
RPN/ROI head), just with a single-stage head instead of Faster R-CNN's
two-stage one. Single-stage means no dynamic-shape RPN/ROI/NMS control
flow, so the saved model converts to TFLite in full (see
src/convert_tflite.py).

Identical pipeline to train_resnet_tf_detect.py (see that file for the
per-design-choice rationale - warmup callback, background-image padding,
Adam@5e-5, fixed-size image loading) - the only difference is the backbone
preset (efficientnet_b0_ra_imagenet instead of resnet_50_imagenet).

Unlike the old torchvision checkpoint, class indices here are 0-indexed
with no reserved background label - KerasHub's RetinaNet doesn't need one.
"""
import argparse
from pathlib import Path

import keras
import keras_hub
import numpy as np
import tensorflow as tf
import yaml

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def load_split(split_dir, height, width):
    # Images are resized to a fixed (height, width) here (boxes rescaled to
    # match) rather than left to the preprocessor's own resizing: RetinaNet's
    # preprocessor expects a ragged_rank=1 batch (a list of full images), and
    # ds.ragged_batch() on variable (H, W) images instead produces a doubly
    # ragged tensor it can't Pad. Fixing the size up front sidesteps that and
    # matches the fixed input shape convert_tflite.py compiles for anyway.
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    images, boxes, classes = [], [], []
    for image_path in image_paths:
        img = tf.io.decode_image(tf.io.read_file(str(image_path)), channels=3, expand_animations=False)
        img = tf.image.resize(img, (height, width), method="bilinear")
        img = tf.cast(img, tf.uint8)

        img_boxes, img_classes = [], []
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                cls_id, cx, cy, bw, bh = map(float, line.split())
                x1 = (cx - bw / 2) * width
                y1 = (cy - bh / 2) * height
                x2 = (cx + bw / 2) * width
                y2 = (cy + bh / 2) * height
                img_boxes.append([y1, x1, y2, x2])  # KerasHub RetinaNet expects yxyx
                img_classes.append(int(cls_id))  # 0-indexed, no background offset

        if not img_boxes:
            # RetinaNetLabelEncoder's target_gather crashes on a genuinely
            # empty (0, 4) gt_boxes array for a sample (background-only
            # images, see raw/background/ in the annotation step) - pad with
            # one dummy zero-area box tagged background_class (-1, the same
            # sentinel the label encoder already assigns unmatched anchors)
            # so every sample has >=1 row; IoU with a zero-area box is 0
            # everywhere, so it can never become a false positive match.
            img_boxes.append([0.0, 0.0, 0.0, 0.0])
            img_classes.append(-1)

        images.append(img.numpy())
        boxes.append(np.array(img_boxes, dtype=np.float32).reshape(-1, 4))
        classes.append(np.array(img_classes, dtype=np.int32))

    return images, boxes, classes


def make_dataset(images, boxes, classes, batch_size, shuffle):
    def gen():
        for img, b, c in zip(images, boxes, classes):
            yield img, {"boxes": b, "labels": c}

    height, width = images[0].shape[0], images[0].shape[1]
    output_signature = (
        tf.TensorSpec(shape=(height, width, 3), dtype=tf.uint8),
        {
            "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            "labels": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        },
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images))
    return ds.ragged_batch(batch_size)


class WarmupCallback(keras.callbacks.Callback):
    """Ramps optimizer.learning_rate from target_lr/1000 to target_lr over
    warmup_steps (global, not per-epoch), then leaves it alone.

    A freshly-initialized detection head (RetinaNet's classification head
    starts strongly biased toward background, see use_prior_probability in
    PredictionHead) produces very large loss/gradients on the first few
    batches - empirically this explodes cls_logits_loss to 1e21+ within one
    epoch without warmup. Same instability, same fix as the old torchvision
    trainer's warmup_lr_scheduler (see git history).

    Implemented as a callback rather than a keras.optimizers.schedules.
    LearningRateSchedule so it composes with ReduceLROnPlateau below -
    that callback reads/writes optimizer.learning_rate as a plain float
    (backend.convert_to_numpy(optimizer.learning_rate)), which fails on a
    LearningRateSchedule object.
    """

    def __init__(self, target_lr, warmup_steps):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_steps = max(warmup_steps, 1)
        self.step = 0

    def on_train_batch_begin(self, batch, logs=None):
        if self.step >= self.warmup_steps:
            return
        alpha = self.step / self.warmup_steps
        lr = self.target_lr * ((1.0 / 1000.0) * (1.0 - alpha) + alpha)
        self.model.optimizer.learning_rate = lr
        self.step += 1


def build_model(num_classes, height, width):
    image_encoder = keras_hub.models.Backbone.from_preset("efficientnet_b0_ra_imagenet")
    backbone = keras_hub.models.RetinaNetBackbone(
        image_encoder=image_encoder, min_level=3, max_level=5, use_p5=True,
    )
    # image_size must be set for the converter's bounding-box-aware Resizing
    # layer to run at all (it no-ops otherwise) - that's also what converts
    # ragged per-image box/label lists into the dense, padded tensors
    # RetinaNetLabelEncoder requires (it doesn't accept tf.RaggedTensor).
    preprocessor = keras_hub.models.RetinaNetObjectDetectorPreprocessor(
        image_converter=keras_hub.layers.RetinaNetImageConverter(
            scale=1 / 255, image_size=(height, width), bounding_box_format="yxyx",
        ),
    )
    return keras_hub.models.RetinaNetObjectDetector(
        backbone=backbone, num_classes=num_classes, preprocessor=preprocessor,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data_detect/data.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5,
                         help="Adam learning rate - matches ObjectDetector.compile()'s own "
                              "'auto' default; the box/cls losses use reduction='sum' over "
                              "thousands of anchors, so an SGD-scale lr (~1e-3) reliably blows "
                              "up cls_logits_loss within a few steps")
    parser.add_argument("--height", type=int, default=480, help="Fixed input height to train at")
    parser.add_argument("--width", type=int, default=640, help="Fixed input width to train at")
    parser.add_argument("--output", default="models/efficientnet_tf_efficientnet_b0_detect_best.keras")
    args = parser.parse_args()

    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"No dataset at '{args.data}'. Run src/dataset/auto_annotate_bboxes.py then "
            f"src/dataset/prepare_detect_dataset.py first."
        )
    data_root = data_yaml_path.parent
    names_dict = yaml.safe_load(data_yaml_path.read_text())["names"]
    class_names = [names_dict[i] for i in sorted(names_dict)]
    num_classes = len(class_names)

    print("Loading dataset ...")
    train_images, train_boxes, train_classes = load_split(data_root / "train", args.height, args.width)
    val_images, val_boxes, val_classes = load_split(data_root / "val", args.height, args.width)
    print(f"Train: {len(train_images)} images  Val: {len(val_images)} images")

    train_ds = make_dataset(train_images, train_boxes, train_classes, args.batch_size, shuffle=True)
    val_ds = make_dataset(val_images, val_boxes, val_classes, args.batch_size, shuffle=False)

    model = build_model(num_classes, args.height, args.width)
    steps_per_epoch = max(len(train_images) // args.batch_size, 1)
    warmup_steps = min(1000, steps_per_epoch - 1) if steps_per_epoch > 1 else 1
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
    )  # box_loss/classification_loss left on "auto" (Huber / binary focal - see ObjectDetector.compile)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        WarmupCallback(target_lr=args.lr, warmup_steps=warmup_steps),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(output_path), monitor="val_loss", save_best_only=True,
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    print(f"\nSaved best model to {output_path}")
    print(f"Class names ({num_classes}): {class_names}")
    print(
        "\nRun 'python src/evaluate_models.py --model-path "
        f"{output_path}' for precision/recall/F1 on the test split."
    )


if __name__ == "__main__":
    main()
