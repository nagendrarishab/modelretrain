#!/bin/bash
# Runs the Mac YOLO training benchmark across all yolo26 model sizes.
# Each model runs as its own process (src/benchmark_yolo_mac.py) so an OOM
# on a bigger model doesn't kill the whole sweep.
set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-2}"
OUT_DIR="bench_out"
mkdir -p "$OUT_DIR"

declare -a MODELS=(yolo26n.pt yolo26s.pt yolo26m.pt yolo26l.pt yolo26x.pt)
declare -A BATCH=( [yolo26n.pt]=16 [yolo26s.pt]=16 [yolo26m.pt]=8 [yolo26l.pt]=4 [yolo26x.pt]=2 )

for model in "${MODELS[@]}"; do
    batch="${BATCH[$model]}"
    out="$OUT_DIR/${model%.pt}.json"
    echo "=== Benchmarking $model (batch=$batch, epochs=$EPOCHS) ==="
    python3 src/benchmark_yolo_mac.py --model "$model" --epochs "$EPOCHS" --batch "$batch" --out "$out"
    echo "=== Done $model ==="
done

echo "All benchmarks complete. Results in $OUT_DIR/"
