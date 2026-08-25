"""Shared helpers for exporting a torchvision ResNet+FPN backbone to ONNX and
compiling it with Apache TVM (Relax). Used by convert-tvm.py (standalone
conversion) and run_camera_resnet_detect_tvm.py (live hybrid inference).

See convert-tvm.py's module docstring for why only the backbone - and not
the RPN/ROI heads/NMS - is compiled.
"""
from pathlib import Path

import onnx
import torch
from onnxsim import simplify
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx


def load_backbone(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = resnet_fpn_backbone(backbone_name=ckpt["backbone"], weights=None, trainable_layers=0)
    backbone_state = {
        k[len("backbone."):]: v
        for k, v in ckpt["model_state_dict"].items()
        if k.startswith("backbone.")
    }
    backbone.load_state_dict(backbone_state)
    backbone.eval()
    return backbone, ckpt


def export_onnx(backbone, height, width, onnx_path):
    dummy = torch.rand(1, 3, height, width)
    with torch.no_grad():
        out = backbone(dummy)
    feature_names = list(out.keys())

    torch.onnx.export(
        backbone, dummy, str(onnx_path),
        input_names=["image"], output_names=[f"feat_{k}" for k in feature_names],
        opset_version=17, dynamo=False,
    )
    return feature_names


def compile_to_tvm(onnx_path):
    model = onnx.load(str(onnx_path))
    # Constant-folds the shape/resize arithmetic baked in at the fixed
    # input size - without this, TVM's Relax ONNX frontend chokes on the
    # dynamic-shape Resize op left over from FPN's upsample-and-add.
    model, check = simplify(model)
    if not check:
        raise RuntimeError("onnx-simplifier could not validate the simplified model")

    mod = from_onnx(model)  # keep_params_in_input=False: weights baked in as constants
    mod = relax.get_pipeline("zero")(mod)
    return tvm.compile(mod, tvm.target.Target("llvm"))


def compile_backbone_cached(backbone, height, width, cache_path):
    """Compiles `backbone` for a fixed (height, width) input, reusing a
    previously compiled module at `cache_path` if one already exists for
    that exact shape (the .so filename embeds height/width, so a shape
    change naturally misses the cache instead of silently loading a
    mismatched module)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return tvm.runtime.load_module(str(cache_path))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = cache_path.with_suffix(".onnx")
    feature_names = export_onnx(backbone, height, width, onnx_path)
    ex = compile_to_tvm(onnx_path)
    ex.export_library(str(cache_path))
    return ex


FEATURE_NAMES = ["0", "1", "2", "3", "pool"]
