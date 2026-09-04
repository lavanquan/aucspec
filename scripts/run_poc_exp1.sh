#!/bin/bash
set -x
cd /software/projects/pawsey1257/quanla/aucspec
module load pytorch/2.7.1-rocm6.3.3

PROJECT=/software/projects/pawsey1257/quanla/aucspec
export PYTHONUSERBASE="$PROJECT/.pythonuserbase-rocm-vllm"
export TRITON_CACHE_DIR="$PROJECT/.triton-cache"
export XDG_CACHE_HOME="$PROJECT/.cache"
export HF_HOME="$PROJECT/.hf-home"
export HF_HUB_CACHE="$PROJECT/models"
export TRANSFORMERS_CACHE="$PROJECT/models"
export HF_DATASETS_CACHE="$PROJECT/.hf-datasets-cache"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT/.torchinductor-cache"
export MPLCONFIGDIR="$PROJECT/.config/matplotlib"
export XDG_CONFIG_HOME="$PROJECT/.config"
export PIP_CACHE_DIR="$PROJECT/.pip-cache"
export WANDB_DIR="$PROJECT/.wandb"
export WANDB_CACHE_DIR="$PROJECT/.cache/wandb"
export VLLM_NO_USAGE_STATS=1
export VLLM_USE_TRITON_FLASH_ATTN=0
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCHINDUCTOR_CACHE_DIR" \
  "$MPLCONFIGDIR" "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"

# GPU-utilization proof, sampled every 2s in the background for the whole
# run, so we can verify inference is actually executing on the GPU (not
# just weights sitting in VRAM while compute silently falls back to CPU).
( while true; do
    date +%s.%N
    rocm-smi --showuse 2>&1
    sleep 2
  done > "$PROJECT/poc_exp1_gpu_usage.log" ) &
GPU_MONITOR_PID=$!

python3 legacy/scripts/run_simulation.py --config "$PROJECT/poc_exp1.yaml" --detailed-log
SIM_EXIT=$?

kill "$GPU_MONITOR_PID" 2>/dev/null
echo "=== POC_EXP1_DONE exit=$SIM_EXIT ==="
