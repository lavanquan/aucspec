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

# Need pandas/pyyaml on top of the module's python for the driver script
# itself (not just the simulator subprocess, which uses the same env).
python3 -m pip install --break-system-packages --quiet pandas pyyaml 2>&1 | tail -5

python3 scripts/run_exp5_alpha_hetero.py "$@"
echo "=== EXP5_ALPHA_HETERO_DONE exit=$? ==="
