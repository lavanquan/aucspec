#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:8
#SBATCH --time=2:00:00
#SBATCH --job-name=v2_smoke
#SBATCH --output=v2_smoke_%j.log
# Short end-to-end check of the V2 verifier in the simulator before the
# frontier campaign (scripts/v2_smoke.py). Same environment as
# scripts/run_capacity_frontier.sh.
#   sbatch scripts/run_v2_smoke.sh [v2_smoke.py args]
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

python3 scripts/v2_smoke.py "$@"
echo "=== V2_SMOKE_DONE exit=$? ==="
