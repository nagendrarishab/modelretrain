"""
Single-model YOLO training benchmark, run as its own process so a crash/OOM
on one model size doesn't take down the rest of the sweep.

    python src/benchmark_yolo_mac.py --model yolo26n.pt --epochs 2 --batch 16 --out bench_out/yolo26n.json
"""
import argparse
import csv
import json
import os
import resource
import time
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import torch
from ultralytics import YOLO


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_results_csv(save_dir):
    csv_path = Path(save_dir) / "results.csv"
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return [{k.strip(): v.strip() for k, v in row.items()} for row in reader]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data_detect/data.yaml")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = get_device()
    print(f"[{args.model}] device={device} batch={args.batch} epochs={args.epochs}")

    model = YOLO(args.model)
    n_params = sum(p.numel() for p in model.model.parameters())

    t0 = time.time()
    error = None
    save_dir = None
    try:
        train_results = model.train(
            data=str(Path(args.data).resolve()),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.img_size,
            device=device,
            project="bench_runs",
            name=args.model.replace(".pt", ""),
            exist_ok=True,
            plots=False,
            verbose=False,
        )
        save_dir = str(train_results.save_dir)
    except Exception as e:  # noqa: BLE001 - want to record failure and move on
        error = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0

    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS reports bytes

    rows = read_results_csv(save_dir) if save_dir else []
    last_row = rows[-1] if rows else {}

    def find_col(row, needle):
        for k in row:
            if needle in k:
                return row[k]
        return None

    result = {
        "model": args.model,
        "device": device,
        "batch": args.batch,
        "epochs_requested": args.epochs,
        "epochs_completed": len(rows),
        "img_size": args.img_size,
        "params": n_params,
        "elapsed_sec": elapsed,
        "sec_per_epoch": elapsed / len(rows) if rows else None,
        "peak_rss_gb": peak_rss_gb,
        "map50": float(find_col(last_row, "mAP50(B)")) if find_col(last_row, "mAP50(B)") else None,
        "map50_95": float(find_col(last_row, "mAP50-95(B)")) if find_col(last_row, "mAP50-95(B)") else None,
        "box_loss": float(find_col(last_row, "box_loss")) if find_col(last_row, "box_loss") else None,
        "error": error,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
