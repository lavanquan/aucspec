#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:8
#SBATCH --time=2:00:00
# >>> DEFECTIVE LEGACY RUNTIME GUARD (naive-SDPA ROCm fallback: batched verification is wrong for >= 3 sequences) >>>
# Kept only for historical reproduction. New canonical work must use scripts/run_canonical_job.sh (V1-batchwall-tritonfa33).
if [ "${ALLOW_DEFECTIVE_LEGACY_RUNTIME:-0}" != "1" ]; then
  echo "REFUSING: this launcher selects the defective legacy runtime (VLLM_USE_TRITON_FLASH_ATTN=0). Set ALLOW_DEFECTIVE_LEGACY_RUNTIME=1 only to reproduce historical results." >&2
  exit 64
fi
# <<< END GUARD
# usage: sbatch --job-name=c3_N<N>_s<S> --output=<run_dir>/logs/n<N>_s<S>_%j.log scripts/run_x3_cert_job.sh <N> <SEED>
set -x
N="$1"
SEED="$2"
RUN_TAG="${RUN_TAG:-budget100_x3_certification}"
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
export VLLM_NO_USAGE_STATS=1
export VLLM_USE_TRITON_FLASH_ATTN=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
python3 -m pip install --break-system-packages --quiet pandas pyyaml matplotlib 2>&1 | tail -3
python3 scripts/certify_x3_run.py run --run-tag "$RUN_TAG" --n "$N" --seed "$SEED"
echo "=== X3CERT_DONE n=$N seed=$SEED exit=$? ==="
