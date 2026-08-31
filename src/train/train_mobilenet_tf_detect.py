"""
    python src/train/train_mobilenet_tf_detect.py --backbone mobilenet_v3_large_100_imagenet --epochs 30
    ./.venv/bin/python src/train/train_mobilenet_tf_detect.py --epochs 2 --batch-size 16
"""
import argparse
from datetime import datetime
from pathlib import Path

import keras
import keras_hub
import numpy as np
import tensorflow as tf
import yaml

tf.config.set_visible_devices([], "GPU")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

BACKBONE_CHOICES = ["mobilenet_v3_large_100_imagenet", "mobilenet_v3_large_100_imagenet_21k"]


def make_emitter(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"train_mobilenet_tf_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = open(log_path, "w")

    def emit(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    return emit, log_path

PYRAMID_STRIDES = (8, 16, 32)


def attach_pyramid_outputs(encoder, target_strides=PYRAMID_STRIDES):
    stride = 2
    layer_name_at_stride = {}
    for stack_idx, sub_strides in enumerate(encoder.stackwise_num_strides):
        for block_idx, s in enumerate(sub_strides):
            stride *= s
            layer_name_at_stride[stride] = f"block_{stack_idx + 1}_{block_idx}"

    missing = [s for s in target_strides if s not in layer_name_at_stride]
    if missing:
        raise ValueError(
            f"'{encoder.name}' never reaches stride(s) {missing} (only reaches "
            f"{sorted(layer_name_at_stride)}) - pick a preset with enough "
            f"downsampling stages, e.g. one of the *_large_* presets."
        )

    encoder.pyramid_outputs = {
        f"P{i + 3}": encoder.get_layer(layer_name_at_stride[s]).output
        for i, s in enumerate(target_strides)
    }
    return encoder


@keras.saving.register_keras_serializable(package="mobilenet_tf")
class PyramidMobileNetBackbone(keras_hub.models.MobileNetBackbone):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        attach_pyramid_outputs(self)


def load_pyramid_mobilenet(preset):
    pretrained = keras_hub.models.MobileNetBackbone.from_preset(preset)
    encoder = PyramidMobileNetBackbone(**pretrained.get_config())
    encoder.set_weights(pretrained.get_weights())
    return encoder


def load_split(split_dir, height, width):
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    image_path_strings, boxes, classes = [], [], []
    for image_path in image_paths:
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
            img_boxes.append([0.0, 0.0, 0.0, 0.0])
            img_classes.append(-1)

        image_path_strings.append(str(image_path))
        boxes.append(np.array(img_boxes, dtype=np.float32).reshape(-1, 4))
        classes.append(np.array(img_classes, dtype=np.int32))

    return image_path_strings, boxes, classes


def make_dataset(image_paths, boxes, classes, batch_size, shuffle, height, width):
    def gen():
        for p, b, c in zip(image_paths, boxes, classes):
            yield p, {"boxes": b, "labels": c}

    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        {
            "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            "labels": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        },
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths))

    def read_image(path, targets):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (height, width), method="bilinear")
        img = tf.cast(img, tf.uint8)
        return img, targets

    ds = ds.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.ragged_batch(batch_size).prefetch(tf.data.AUTOTUNE)


class WarmupCallback(keras.callbacks.Callback):

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


class EmitCallback(keras.callbacks.Callback):
    """Logs each epoch's metrics via emit()"""

    def __init__(self, emit):
        super().__init__()
        self.emit = emit

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics = "  ".join(f"{k}={v:.4f}" for k, v in logs.items())
        self.emit(f"Epoch {epoch + 1}/{self.params['epochs']}  {metrics}")


def build_model(num_classes, height, width, backbone_preset):
    image_encoder = load_pyramid_mobilenet(backbone_preset)
    backbone = keras_hub.models.RetinaNetBackbone(
        image_encoder=image_encoder, min_level=3, max_level=5, use_p5=True,
    )

    preprocessor = keras_hub.models.RetinaNetObjectDetectorPreprocessor(
        image_converter=keras_hub.layers.RetinaNetImageConverter(
            scale=1 / 255, image_size=(height, width), bounding_box_format="yxyx",
        ),
    )
    return keras_hub.models.RetinaNetObjectDetector(
        backbone=backbone, num_classes=num_classes, preprocessor=preprocessor,
    )


def main():
    keras.mixed_precision.set_global_policy("mixed_float16")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data_detect/data.yaml")
    parser.add_argument("--backbone", default="mobilenet_v3_large_100_imagenet", choices=BACKBONE_CHOICES,
                         help="MobileNetV3-Large preset - the plain ImageNet-1k one (default) or the "
                              "ImageNet-21k-pretrained variant")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5,
                         help="Adam learning rate - matches ObjectDetector.compile()'s own "
                              "'auto' default; the box/cls losses use reduction='sum' over "
                              "thousands of anchors, so an SGD-scale lr (~1e-3) reliably blows "
                              "up cls_logits_loss within a few steps")
    parser.add_argument("--height", type=int, default=480,
                         help="Fixed input height to train at - keep this and --width divisible by 32 "
                              "(the P3/P4/P5 pyramid's coarsest stride), or the nearest-upsample-and-add "
                              "in RetinaNet's FeaturePyramid can mismatch shapes between levels and raise "
                              "'Inputs have incompatible shapes' (confirmed directly: 240x320 fails this "
                              "way, the 480x640 default doesn't)")
    parser.add_argument("--width", type=int, default=640, help="Fixed input width to train at - see --height")
    parser.add_argument("--patience", type=int, default=5,
                         help="Epochs with no val_loss improvement before ReduceLROnPlateau halves the lr")
    parser.add_argument("--output", default="models/mobilenet_tf_mobilenetv3_large_detect_best.keras")
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
    names_dict = yaml.safe_load(data_yaml_path.read_text())["names"]
    class_names = [names_dict[i] for i in sorted(names_dict)]
    num_classes = len(class_names)

    emit("Loading dataset ...")
    train_images, train_boxes, train_classes = load_split(data_root / "train", args.height, args.width)
    val_images, val_boxes, val_classes = load_split(data_root / "val", args.height, args.width)
    emit(f"Train: {len(train_images)} images  Val: {len(val_images)} images")

    train_ds = make_dataset(train_images, train_boxes, train_classes, args.batch_size, shuffle=True, height=args.height, width=args.width)
    val_ds = make_dataset(val_images, val_boxes, val_classes, args.batch_size, shuffle=False, height=args.height, width=args.width)

    model = build_model(num_classes, args.height, args.width, args.backbone)
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
            monitor="val_loss", factor=0.5, patience=args.patience, min_lr=1e-7, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(output_path), monitor="val_loss", save_best_only=True,
        ),
        EmitCallback(emit),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    emit(f"\nSaved best model to {output_path}")
    emit(f"Class names ({num_classes}): {class_names}")
    emit(
        "\nRun 'python src/evaluate_models.py --model-path "
        f"{output_path}' for precision/recall/F1 on the test split."
    )


if __name__ == "__main__":
    main()
