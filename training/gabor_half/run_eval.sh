#!/usr/bin/env bash
set -euo pipefail
ROOT=${PARAKEET_ROOT:-/home/nathanroll/parakeet-ft}
EXP="$ROOT/gabor_half_20260906"
MODEL=$1
LABEL=$2
cd "$ROOT"
NEMO_NAME="orukeet-gabor-eval-$LABEL" ./nemo.sh env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
 PYTHONPATH="$EXP/export-deps" \
 python "$EXP/evaluate_fast.py" --model "$MODEL" --output "$EXP/eval-$LABEL" --batch-size 32 \
 "$EXP"/development/*.jsonl
