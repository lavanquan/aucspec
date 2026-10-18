#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --time=0:45:00
# usage: sbatch --job-name=verif_calib --output=/scratch/pawsey1257/quanla/calibration_raw/<tag>/slurm_%j.log \
#        <worktree>/scripts/run_calibration_job.sh <WORKTREE> <CONFIG> <TAG>
# One job, one node, 45 min. Raw JSONL is checkpointed (fsync per call) on /scratch.
set -x
WT="$1"; CFG="$2"; TAG="$3"
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
OUT=/scratch/pawsey1257/quanla/calibration_raw/"$TAG"
mkdir -p "$OUT"
date +%s > "$OUT/job_start_unix.txt"
python3 -m pip install --break-system-packages --quiet pandas pyyaml 2>&1 | tail -3
python3 scripts/calibrate_verifier.py --config "$CFG" --out-dir "$OUT" --deadline-min 38
rc=$?
date +%s > "$OUT/job_end_unix.txt"
echo "=== CALIBRATION_DONE exit=$rc ==="
