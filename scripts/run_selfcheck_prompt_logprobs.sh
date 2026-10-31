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

# The only working V0 path is naive SDPA (no flash_attn installed), and it mixes
# sequences in batched prefill. Triton FA and V1 fail because the userbase
# triton does not match torch. Test them with the container's own triton put
# ahead of the userbase via a one-package overlay (nothing installed/removed).
SYS_TRITON=$(PYTHONNOUSERSITE=1 python3 -c "import triton, os; print(os.path.dirname(triton.__file__))")
OVERLAY=/scratch/pawsey1257/quanla/triton_overlay
mkdir -p "$OVERLAY" && ln -sfn "$SYS_TRITON" "$OVERLAY/triton"
echo "userbase triton: $(python3 -c 'import triton; print(triton.__version__, triton.__file__)')"
echo "container triton: $(PYTHONPATH=$OVERLAY python3 -c 'import triton; print(triton.__version__, triton.__file__)')"

run() {  # run <tag> <VLLM_USE_V1> <VLLM_USE_TRITON_FLASH_ATTN> [selfcheck args...]
  local tag=$1 v1=$2 fa=$3; shift 3
  PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}" VLLM_USE_V1=$v1 VLLM_USE_TRITON_FLASH_ATTN=$fa \
    python3 legacy/scripts/selfcheck_prompt_logprobs.py \
    --config configs/profile_target.yaml --context-lengths 300,1024 "$@" \
    --output "results/section3/selfcheck_${tag}.json" || echo "!! variant $tag failed"
}
run systriton_v0_tritonfa 0 1
run systriton_v1 1 1
run systriton_v1_eager 1 1 --enforce-eager
echo "=== SELFCHECK_DONE ==="
