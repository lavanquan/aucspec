#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --job-name=profile_target
#SBATCH --output=profile_target_%j.log
# Verifier profile with profile_target.py defaults (B 1..32, gamma 0..16,
# context 128..4096), then the Section 3 roofline fit on the result.
#   sbatch scripts/run_profile_target.sh
set -x
cd /software/projects/pawsey1257/quanla/aucspec
module load pytorch/2.7.1-rocm6.3.3

PROJECT=/software/projects/pawsey1257/quanla/aucspec
export PYTHONUSERBASE="$PROJECT/.pythonuserbase-rocm-vllm"
export PYTHONPYCACHEPREFIX="/scratch/pawsey1257/quanla/.pycache"
export TRITON_CACHE_DIR="$PROJECT/.triton-cache"
export XDG_CACHE_HOME="$PROJECT/.cache"
export HF_HOME="$PROJECT/.hf-home"
export HF_HUB_CACHE="$PROJECT/models"
export TRANSFORMERS_CACHE="$PROJECT/models"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT/.torchinductor-cache"
export PYTHONUNBUFFERED=1
export VLLM_NO_USAGE_STATS=1
source scripts/setonix_vllm_env.sh
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

OUT=results/section3
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/profile_commit.txt"
git status --porcelain >> "$OUT/profile_commit.txt"

python3 legacy/scripts/profile_target.py \
  --config configs/profile_target.yaml \
  --output "$OUT/target_profile.csv" \
  --summary-output "$OUT/target_profile_summary.json"
rc=$?
echo "=== PROFILE_TARGET_DONE exit=$rc ==="
[ $rc -eq 0 ] || exit $rc

python3 scripts/section3_model.py fit \
  --profile-csv "$OUT/target_profile.csv" \
  --output "$OUT/roofline_fit.json"
echo "=== ROOFLINE_FIT_DONE exit=$? ==="
