#!/bin/bash
set -x
cd /software/projects/pawsey1257/quanla/aucspec
module load pytorch/2.7.1-rocm6.3.3
# Nothing should be written under $HOME -- its per-user quota is tight
# regardless of total filesystem free space (confirmed: 149MB used already
# trips it, on a filesystem with 3.3T free). Redirect every cache dir we
# know of to project space up front, instead of fixing them one crash at a
# time (already hit this for pip installs during the vLLM build, and for
# Triton's JIT cache and Hugging Face's cache during earlier runs of this
# script -- HF_HOME below is broader than HF_HUB_CACHE alone, since HF also
# writes a token file/assets cache under $HOME/.cache/huggingface that
# HF_HUB_CACHE does not cover).
PROJECT=/software/projects/pawsey1257/quanla/aucspec
export PYTHONUSERBASE="$PROJECT/.pythonuserbase-rocm-vllm"
export TRITON_CACHE_DIR="$PROJECT/.triton-cache"
export XDG_CACHE_HOME="$PROJECT/.cache"
export HF_HOME="$PROJECT/.hf-home"
export HF_HUB_CACHE="$PROJECT/models"
export TRANSFORMERS_CACHE="$PROJECT/models"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT/.torchinductor-cache"
export MPLCONFIGDIR="$PROJECT/.config/matplotlib"
export XDG_CONFIG_HOME="$PROJECT/.config"
export PIP_CACHE_DIR="$PROJECT/.pip-cache"
export WANDB_DIR="$PROJECT/.wandb"
export WANDB_CACHE_DIR="$PROJECT/.cache/wandb"
# vLLM writes $XDG_CONFIG_HOME/vllm/usage_stats.json and spawns a
# wandb-style debug logger under $XDG_CACHE_HOME/wandb regardless of the
# above unless usage stats reporting is disabled outright.
export VLLM_NO_USAGE_STATS=1
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" \
  "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR"

# vLLM's own warning log ("Model architecture 'Qwen2ForCausalLM' is
# partially supported by ROCm...") already pointed at this: pip pulled the
# latest Triton (3.8.0) while vllm==0.9.1's Triton flash-attention kernel
# (vllm/attention/ops/triton_flash_attention.py) was written against an
# older Triton API and crashes with
# AttributeError("'_aggregate_type' object has no attribute 'element_ty'").
# Switch to vLLM's CK (Composable Kernel) flash-attention backend instead,
# which avoids that Triton code path entirely.
export VLLM_USE_TRITON_FLASH_ATTN=0

python3 scripts/collect_trace.py \
  --target-model Qwen/Qwen2.5-7B-Instruct \
  --draft-model Qwen/Qwen2.5-1.5B-Instruct \
  --gamma 4 \
  --rounds-per-prompt 15 \
  --draft-device cuda:1 \
  --out results/trace_real.jsonl
echo "=== COLLECT_TRACE_DONE exit=$? ==="
