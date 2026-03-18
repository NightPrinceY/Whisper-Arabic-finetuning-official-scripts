#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Quick smoke test — 100 steps, tiny data, single GPU
#  Run this first to verify the pipeline works end-to-end before full training
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

source ~/CollegeX/bin/activate
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$PYTHONPATH"
export HF_TOKEN="${HF_TOKEN}"

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/smoke_${TIMESTAMP}.log"

echo "Running smoke test (100 steps, single GPU) — log: ${LOG_FILE}"

# Single GPU for smoke test
CUDA_VISIBLE_DEVICES=5 accelerate launch \
    --num_processes=1 \
    --mixed_precision=fp16 \
    src/training/train.py \
    --config configs/config.yaml \
    --smoke_test \
    --output_dir ./outputs/smoke_test \
    2>&1 | tee "${LOG_FILE}"

echo "Smoke test done. Check: ${LOG_FILE}"
