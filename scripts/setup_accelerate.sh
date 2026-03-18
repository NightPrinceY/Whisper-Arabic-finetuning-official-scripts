#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  One-time accelerate config setup for 4-GPU fp16 training
#  Run once before first training: bash scripts/setup_accelerate.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

source ~/CollegeX/bin/activate

mkdir -p ~/.cache/huggingface/accelerate

cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "Accelerate config written to ~/.cache/huggingface/accelerate/default_config.yaml"
accelerate env
