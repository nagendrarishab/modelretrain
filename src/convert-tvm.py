"""Compile the ResNet+FPN backbone of a trained Faster R-CNN detector
(models/resnet_<backbone>_detect_best.pt) to a deployable Apache TVM module.

Only the backbone (ResNet + FPN: plain conv/batchnorm/relu/upsample) is
compiled - it's the compute-heavy part and the only part that survives
TVM's Relax ONNX frontend cleanly. The RPN, ROI heads and NMS have
data-dependent control flow (top-k proposal selection, NMS, variable box
counts) that this frontend can't convert: it fails on dynamic-shape Resize
sizes and an int32/int64 dtype mismatch inside the RPN's proposal-merging
Add op. That's a frontend limitation, not something fixable from calling
code, so those stages must keep running in PyTorch.

Usage:
    python src/convert-tvm.py --checkpoint models/resnet_resnet18_detect_best.pt \
        --height 480 --width 640 --output models/resnet18_backbone_tvm.so
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import tvm
from tvm import relax
from tvm_backbone import compile_to_tvm, export_onnx, load_backbone


def check_parity(backbone, ex, height, width, feature_names):
    np.random.seed(0)
    x_np = np.random.rand(1, 3, height, width).astype("float32")
    with torch.no_grad():
        torch_out = backbone(torch.from_numpy(x_np))

    device = tvm.cpu()
    vm = relax.VirtualMachine(ex, device)
    tvm_out = vm["main"](tvm.runtime.tensor(x_np, device=device))

    print("Parity check (compiled TVM module vs. PyTorch, random input):")
    for i, name in enumerate(feature_names):
        diff = np.abs(torch_out[name].numpy() - tvm_out[i].numpy()).max()
        print(f"  feat[{name}] shape={tuple(torch_out[name].shape)} max_abs_diff={diff:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="models/resnet_resnet18_detect_best.pt")
    parser.add_argument("--height", type=int, default=480,
                         help="Fixed input height to compile for (TVM needs a static shape).")
    parser.add_argument("--width", type=int, default=640,
                         help="Fixed input width to compile for (TVM needs a static shape).")
    parser.add_argument("--output", default=None,
                         help="Output .so path. Defaults to models/<checkpoint-stem>_backbone_tvm.so")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if args.output is None:
        args.output = f"models/{checkpoint_path.stem}_backbone_tvm.so"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    onnx_path = output_path.with_suffix(".onnx")

    print(f"Loading backbone from {checkpoint_path} ...")
    backbone, _ = load_backbone(checkpoint_path)

    print(f"Exporting to ONNX at fixed input size {args.height}x{args.width} ...")
    feature_names = export_onnx(backbone, args.height, args.width, onnx_path)

    print("Compiling to Apache TVM (Relax, llvm target) ...")
    ex = compile_to_tvm(onnx_path)

    check_parity(backbone, ex, args.height, args.width, feature_names)

    ex.export_library(str(output_path))
    print(f"\nSaved compiled TVM backbone to {output_path}")
    print(f"Intermediate ONNX graph kept at {onnx_path}")
    print(
        "\nNOTE: only the ResNet+FPN backbone is compiled. The RPN, ROI heads, "
        "and NMS still need to run in PyTorch on the feature maps this module "
        "produces - see the module docstring for why."
    )


if __name__ == "__main__":
    main()
