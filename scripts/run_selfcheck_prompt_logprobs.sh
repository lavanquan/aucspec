#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=selfcheck_plp
#SBATCH --output=selfcheck_plp_%j.log
# Does prompt_logprobs agree with the engine's own greedy generation?
#   sbatch scripts/run_selfcheck_prompt_logprobs.sh  (legacy/scripts/selfcheck_prompt_logprobs.py)
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

python3 legacy/scripts/selfcheck_prompt_logprobs.py --config configs/profile_target.yaml \
  --context-lengths 300,1024 "$@"
echo "=== SELFCHECK_DONE ==="
