#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --time=0:20:00
# >>> DEFECTIVE LEGACY RUNTIME GUARD (naive-SDPA ROCm fallback: batched verification is wrong for >= 3 sequences) >>>
# Kept only for historical reproduction. New canonical work must use scripts/run_canonical_job.sh (V1-batchwall-tritonfa33).
if [ "${ALLOW_DEFECTIVE_LEGACY_RUNTIME:-0}" != "1" ]; then
  echo "REFUSING: this launcher selects the defective legacy runtime (VLLM_USE_TRITON_FLASH_ATTN=0). Set ALLOW_DEFECTIVE_LEGACY_RUNTIME=1 only to reproduce historical results." >&2
  exit 64
fi
# <<< END GUARD
# usage: sbatch --job-name=greedy_diag --output=/scratch/pawsey1257/quanla/greedy_diag/<tag>/slurm_%j.log \
#        <worktree>/scripts/run_greedy_diag_job.sh <WORKTREE> <CONFIG> <CALIB_RAW> <TAG>
set -x
WT="$1"; CFG="$2"; CALIB="$3"; TAG="$4"
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
OUT=/scratch/pawsey1257/quanla/greedy_diag/"$TAG"
mkdir -p "$OUT"
date +%s > "$OUT/job_start_unix.txt"
python3 -m pip install --break-system-packages --quiet pandas pyyaml 2>&1 | tail -3
python3 scripts/diagnose_greedy_verifier.py --config "$CFG" --out-dir "$OUT" --calib-raw "$CALIB"
rc=$?
date +%s > "$OUT/job_end_unix.txt"
echo "=== GREEDY_DIAG_DONE exit=$rc ==="
