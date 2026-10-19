#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:8
#SBATCH --time=2:00:00
# Canonical launcher for runtime "V1-batchwall-tritonfa33" (Triton FlashAttention, Triton 3.3.1, V0 engine).
# usage: sbatch --job-name=can_N<N>_s<S> \
#   --output=/scratch/pawsey1257/quanla/canonical_raw/<tag>/logs/n<N>_s<S>_%j.log \
#   <worktree>/scripts/run_canonical_job.sh <WORKTREE> <RUN_TAG> <N> <SEED>
# The run directory must already exist (canonical_launch.py init on the login node). No pip installs happen here.
set -uo pipefail
WT="$1"; RUN_TAG="$2"; N="$3"; SEED="$4"
cd "$WT" || exit 66
module load pytorch/2.7.1-rocm6.3.3
PROJECT=/software/projects/pawsey1257/quanla/aucspec
SHIM=/software/projects/pawsey1257/quanla/aucspec_runtime/tritonfa33/shim
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
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
unset VLLM_ATTENTION_BACKEND VLLM_USE_V1 ALLOW_DEFECTIVE_LEGACY_RUNTIME
export VLLM_USE_TRITON_FLASH_ATTN=1
export PYTHONPATH="$SHIM${PYTHONPATH:+:$PYTHONPATH}"
export CANONICAL_SCRATCH=/scratch/pawsey1257/quanla/canonical_raw
echo "runtime=V1-batchwall-tritonfa33 TRITON_FA=$VLLM_USE_TRITON_FLASH_ATTN PYTHONPATH=$PYTHONPATH"
python3 scripts/preflight_runtime.py || { echo "=== PREFLIGHT_FAILED ===" >&2; exit 65; }
python3 scripts/canonical_launch.py run --run-tag "$RUN_TAG" --n "$N" --seed "$SEED"
rc=$?
echo "=== CANONICAL_DONE n=$N seed=$SEED exit=$rc ==="
exit $rc
