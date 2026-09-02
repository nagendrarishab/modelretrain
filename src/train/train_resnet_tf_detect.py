"""
    python src/train/train_resnet_tf_detect.py --epochs 30

    # GPU run (default) - only one process gets the shared Metal GPU at a time, so use the
    # largest --batch-size that fits instead of splitting it across concurrent GPU runs:
    ./.venv/bin/python src/train/train_resnet_tf_detect.py --epochs 2 --batch-size 16

    # CPU run - pair with a GPU run of another backbone for real parallelism (see --device):
    ./.venv/bin/python src/train/train_resnet_tf_detect.py --epochs 2 --batch-size 4 --device cpu
"""
import argparse
from datetime import datetime
from pathlib import Path

import keras
import keras_hub
import numpy as np
import tensorflow as tf
import yaml

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def log_device(emit):
    """Reports which device TF will place ops on. Unlike torch's get_device()/.to(device)
    pattern used in the other train_*.py scripts, TF auto-places ops on whatever GPU is
    visible - a CUDA GPU on Linux, or Apple Silicon's GPU via the tensorflow-metal plugin -
    both surfacing as device_type "GPU", so there's nothing to select or move tensors to."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        names = ", ".join(gpu.name for gpu in gpus)
        emit(f"Using GPU: {names}")
    else:
        emit("No GPU found - training on CPU")


def make_emitter(log_dir):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"train_resnet_tf_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = open(log_path, "w")

    def emit(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    return emit, log_path


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


def make_dataset(image_paths, boxes, classes, batch_size, shuffle, height, width, cache_path=None):
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

    def read_image(path, targets):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (height, width), method="bilinear")
        img = tf.cast(img, tf.uint8)
        return img, targets

    ds = ds.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        ds = ds.cache(cache_path)  # cache decoded/resized images so only epoch 1 pays disk+decode cost
    if shuffle:
        # shuffle after cache so each epoch gets a fresh permutation instead of replaying epoch 1's order.
        # Buffer holds decoded images (not cheap path strings) here, so cap it well below the full
        # dataset size - a full-size buffer would try to hold ~17GB of images in RAM at once, and a
        # partial/warmup read of the dataset (e.g. Keras building the model on one batch) would force
        # a full buffer fill without ever reaching a clean end-of-sequence, causing the cache write to
        # be discarded every time (the "did not fully read the dataset being cached" warning).
        ds = ds.shuffle(buffer_size=min(len(image_paths), 2048))
    return ds.ragged_batch(batch_size).prefetch(tf.data.AUTOTUNE)


def resolve_cache_path(cache_path, emit):
    """Claims cache_path for this process to build, or returns None to skip caching this
    run if another process is already building it (train_efficientnet_tf_detect.py shares
    the same cache dir/filenames since it decodes the same images at the same size)."""
    if Path(f"{cache_path}.index").exists():
        return cache_path  # already fully built - safe for any number of concurrent readers
    lock_path = Path(f"{cache_path}.lock")
    try:
        lock_path.touch(exist_ok=False)
    except FileExistsError:
        emit(f"Cache at {cache_path} is being built by another run - skipping caching for this run "
             f"(delete {lock_path} if that run crashed without finishing)")
        return None
    return cache_path


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
    def __init__(self, emit):
        super().__init__()
        self.emit = emit

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics = "  ".join(f"{k}={v:.4f}" for k, v in logs.items())
        self.emit(f"Epoch {epoch + 1}/{self.params['epochs']}  {metrics}")


def build_model(num_classes, height, width):
    image_encoder = keras_hub.models.Backbone.from_preset("resnet_18_imagenet")
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5,
                         help="Adam learning rate - matches ObjectDetector.compile()'s own "
                              "'auto' default; the box/cls losses use reduction='sum' over "
                              "thousands of anchors, so an SGD-scale lr (~1e-3) reliably blows "
                              "up cls_logits_loss within a few steps")
    parser.add_argument("--height", type=int, default=480, help="Fixed input height to train at")
    parser.add_argument("--width", type=int, default=640, help="Fixed input width to train at")
    parser.add_argument("--patience", type=int, default=5,
                         help="Epochs with no val_loss improvement before ReduceLROnPlateau halves the lr")
    parser.add_argument("--output", default="models/resnet_tf_resnet18_detect_best.keras")
    parser.add_argument("--log-dir", default="logs", help="Directory for this run's log file")
    parser.add_argument("--cache-dir", default="cache/detect_ds",
                         help="Directory to disk-cache decoded/resized images so only epoch 1 pays "
                              "disk+decode cost. Delete this dir if the dataset or --height/--width change. "
                              "Pass '' to disable caching.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu"],
                         help="'cpu' forces CPU-only training (e.g. to run alongside another script's "
                              "GPU run, or work around a GPU-specific issue) by hiding any GPU from TF. "
                              "'auto' (default) trains on GPU if one is visible (CUDA, or Apple Silicon "
                              "via tensorflow-metal), else CPU")
    args = parser.parse_args()

    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")  # must run before TF touches any GPU op

    emit, log_path = make_emitter(args.log_dir)
    emit(f"Logging to {log_path}")
    log_device(emit)

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

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_cache = resolve_cache_path(str(cache_dir / f"train_{args.height}x{args.width}"), emit)
        val_cache = resolve_cache_path(str(cache_dir / f"val_{args.height}x{args.width}"), emit)
    else:
        train_cache = val_cache = None
    train_ds = make_dataset(train_images, train_boxes, train_classes, args.batch_size, shuffle=True, height=args.height, width=args.width, cache_path=train_cache)
    val_ds = make_dataset(val_images, val_boxes, val_classes, args.batch_size, shuffle=False, height=args.height, width=args.width, cache_path=val_cache)

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
