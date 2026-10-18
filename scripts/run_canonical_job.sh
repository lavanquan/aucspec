#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:8
#SBATCH --time=2:00:00
# usage: sbatch --job-name=can_N<N>_s<S> \
#   --output=/scratch/pawsey1257/quanla/canonical_raw/<tag>/logs/n<N>_s<S>_%j.log \
#   <worktree>/scripts/run_canonical_job.sh <WORKTREE> <RUN_TAG> <N> <SEED>
# Source comes from <WORKTREE> (a clean checkout); caches/models from the project dir.
set -x
WT="$1"; RUN_TAG="$2"; N="$3"; SEED="$4"
cd "$WT"
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
export VLLM_NO_USAGE_STATS=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export CANONICAL_SCRATCH=/scratch/pawsey1257/quanla/canonical_raw
python3 -m pip install --break-system-packages --quiet pandas pyyaml 2>&1 | tail -3
python3 scripts/canonical_launch.py run --run-tag "$RUN_TAG" --n "$N" --seed "$SEED"
echo "=== CANONICAL_DONE n=$N seed=$SEED exit=$? ==="
