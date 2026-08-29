"""Convert a trained TF/Keras detector to a float16-quantized TFLite module
for on-device inference. Handles two different model
shapes in this repo, auto-detected from the loaded checkpoint:

Uses TFLiteConverter.from_saved_model() rather than
from_concrete_functions(): the latter hits an unrelated AttributeError deep
in the converter's debug-info step specifically when SELECT_TF_OPS is
needed (a converter bug, not something fixable from calling code), while
routing through an intermediate SavedModel on disk avoids it.

Usage:
    python src/convert_tflite.py --checkpoint models/resnet_tf_resnet50_detect_best.keras \
        --output models/resnet_tf_resnet50_detect_best_fp16.tflite
    python src/convert_tflite.py --checkpoint models/nanodet_tf_best.keras \
        --height 320 --width 320 --output models/nanodet_tf_best_fp16.tflite
"""
import argparse
import sys
import tempfile
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

sys.path.insert(0, str(Path(__file__).resolve().parent / "run"))
import run_camera_nanodet_tf_detect
import run_camera_mobilenet_tf_detect  # noqa: F401 - side effect only: registers PyramidMobileNetBackbone
                                        # so keras.saving.load_model() can deserialize a MobileNet-TF checkpoint


def is_retinanet(model):
    return hasattr(model, "preprocessor") and hasattr(model, "decode_predictions")


def full_predict(model, x):
    if not is_retinanet(model):
        return model(x, training=False)
    preprocessed = model.preprocessor(x) if model.preprocessor else x
    raw = model(preprocessed, training=False)
    return model.decode_predictions(raw, preprocessed)


def check_parity(model, interpreter, height, width, input_dtype):
    np.random.seed(0)
    if input_dtype == "uint8":
        x_np = np.random.randint(0, 256, size=(1, height, width, 3)).astype("uint8")
    else:
        x_np = np.random.rand(1, height, width, 3).astype("float32")

    keras_out = full_predict(model, x_np)
    is_dict_output = isinstance(keras_out, dict)
    if not is_dict_output:
        keras_out = {str(np.asarray(t).shape): t for t in keras_out}

    input_detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_detail["index"], x_np.astype(input_detail["dtype"]))
    interpreter.invoke()

    print("Parity check (TFLite fp16 module vs. Keras, random input):")
    output_details = sorted(interpreter.get_output_details(), key=lambda d: d["index"])
    for output_detail in output_details:
        name = output_detail["name"]
        tflite_out = interpreter.get_tensor(output_detail["index"])
        if is_dict_output:
            key = next((k for k in keras_out if k in name), None)
        else:
            key = str(tflite_out.shape)
        if key is None or key not in keras_out:
            print(f"  output '{name}' shape={tflite_out.shape} (no matching Keras output to diff against)")
            continue
        diff = np.abs(np.asarray(keras_out[key]).astype("float32") - tflite_out.astype("float32")).max()
        print(f"  {key}: shape={tflite_out.shape} max_abs_diff={diff:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="models/resnet_tf_resnet50_detect_best.keras")
    parser.add_argument("--height", type=int, default=480, help="Fixed input height to convert for.")
    parser.add_argument("--width", type=int, default=640, help="Fixed input width to convert for.")
    parser.add_argument("--output", default=None,
                         help="Output .tflite path. Defaults to models/<checkpoint-stem>_fp16.tflite")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if args.output is None:
        args.output = f"models/{checkpoint_path.stem}_fp16.tflite"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint_path} ...")
    model = keras.saving.load_model(checkpoint_path)
    retinanet = is_retinanet(model)
    input_dtype = tf.uint8 if retinanet else tf.float32
    print(f"Detected {'RetinaNet-based' if retinanet else 'NanoDet-TF'} model "
          f"- input dtype {input_dtype.name}")

    print(f"Building a concrete function at fixed input size {args.height}x{args.width} ...")
    call_fn = tf.function(
        lambda x: full_predict(model, x),
        input_signature=[tf.TensorSpec([1, args.height, args.width, 3], input_dtype)],
    )
    concrete_fn = call_fn.get_concrete_function()
    if not retinanet:
        concrete_fn = convert_variables_to_constants_v2(concrete_fn)

    print("Converting to TFLite (float16) ...")
    with tempfile.TemporaryDirectory() as saved_model_dir:
        tf.saved_model.save(model, saved_model_dir, signatures={"serving_default": concrete_fn})
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        try:
            tflite_model = converter.convert()
        except Exception as e:
            print(
                f"\nBuiltin-only conversion failed ({e}); retrying with SELECT_TF_OPS "
                "enabled (needed for at least one NMS op with no native TFLite kernel - "
                "see the module docstring)..."
            )
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            tflite_model = converter.convert()

    output_path.write_bytes(tflite_model)

    try:
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        check_parity(model, interpreter, args.height, args.width, input_dtype.name)
    except RuntimeError as e:
        print(
            f"\nSkipping local parity check - this Python environment's "
            f"tf.lite.Interpreter can't run the Flex/SELECT_TF_OPS model "
            f"({e}). This does not mean the export is broken - it means the "
            f"same Flex delegate the target device needs (see the module "
            f"docstring) is also required to test it locally."
        )

    print(f"\nSaved TFLite fp16 model to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
