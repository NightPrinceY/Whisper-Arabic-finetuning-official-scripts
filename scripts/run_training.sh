#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Launch full fine-tuning: 8× RTX 2080 Ti, fp16, local everyayah dataset
# ─────────────────────────────────────────────────────────────────────────────

set -eo pipefail

# ── Ensure PYTHONPATH exists (avoids unbound variable errors) ─────────────────
PYTHONPATH="${PYTHONPATH:-}"

# ── Activate environment ──────────────────────────────────────────────────────
source ~/CollegeX/bin/activate

# ── Project root ──────────────────────────────────────────────────────────────
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$PYTHONPATH"

# ── HuggingFace token ─────────────────────────────────────────────────────────
export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# ── GPU config ────────────────────────────────────────────────────────────────
# Use 8 GPUs after WSL restart (wsl --shutdown from Windows PowerShell)
# Use 4 GPUs until then: CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "GPUs seen by PyTorch = $NUM_GPUS"

# ── WSL2 NCCL settings (no P2P in WSL2, use shared memory instead) ────────────
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=WARN

# ── Logging directory ─────────────────────────────────────────────────────────
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_${TIMESTAMP}.log"

echo "Starting training — log: ${LOG_FILE}"
echo "TensorBoard: tensorboard --logdir outputs/whisper-small-quran/tensorboard"

# ── Resume checkpoint if provided ─────────────────────────────────────────────
RESUME_ARG=""
if [ -n "${RESUME:-}" ]; then
    RESUME_ARG="--resume_from_checkpoint $RESUME"
fi

# ── Launch training ───────────────────────────────────────────────────────────
accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=fp16 \
    --dynamo_backend=no \
    --gradient_accumulation_steps=4 \
    src/training/train.py \
    --config configs/config.yaml \
    $RESUME_ARG \
    2>&1 | tee "$LOG_FILE"

echo "Training finished. Log: $LOG_FILE"
