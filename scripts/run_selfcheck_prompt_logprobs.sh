#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
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

# Batched prefill mixes sequences with the default ROCm attention path; try
# naive SDPA (flash_attn blocked), fp16 instead of bf16, and the V1 engine.
run() {  # run <tag> <VLLM_USE_V1> [selfcheck args...]
  local tag=$1 v1=$2; shift 2
  VLLM_USE_V1=$v1 python3 legacy/scripts/selfcheck_prompt_logprobs.py \
    --config configs/profile_target.yaml --context-lengths 300,1024 "$@" \
    --output "results/section3/selfcheck_${tag}.json" || echo "!! variant $tag failed"
}
run control 0
run naive_sdpa 0 --block-flash-attn
run fp16 0 --dtype float16
run v1 1
echo "=== SELFCHECK_DONE ==="
