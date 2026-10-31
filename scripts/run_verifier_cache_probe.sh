#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=probe_cache
#SBATCH --output=probe_cache_%j.log
# Check that target prefix caching keeps verification exact and actually
# reuses the context KV (legacy/scripts/probe_verifier_cache.py).
#   sbatch scripts/run_verifier_cache_probe.sh
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
export VLLM_USE_TRITON_FLASH_ATTN=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

OUT=results/section3
mkdir -p "$OUT"
for mode in off on; do
  python3 legacy/scripts/probe_verifier_cache.py --config configs/profile_target.yaml \
    --prefix-caching $mode --output "$OUT/probe_cache_$mode.json" "$@"
done
python3 legacy/scripts/probe_verifier_cache.py --compare "$OUT/probe_cache_off.json" "$OUT/probe_cache_on.json"
echo "=== PROBE_CACHE_DONE ==="
