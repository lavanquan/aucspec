# Source after `module load pytorch/...` and PYTHONUSERBASE in Setonix job scripts.
#
# The userbase (.pythonuserbase-rocm-vllm) carries triton 3.8.0, which does not
# match the module's torch 2.7.1 (triton 3.3.1): vLLM's Triton flash attention
# fails to compile, V1 fails in inductor, and V0 silently falls back to its
# naive SDPA path, which mixes sequences in batched prefill (21/64 wrong
# verifier positions at B=8, selfcheck_prompt_logprobs.py, 2026-10-31). Put the
# module's own triton ahead of the userbase one with a one-package overlay
# and use Triton flash attention.
SYS_TRITON=$(PYTHONNOUSERSITE=1 python3 -c "import triton, os; print(os.path.dirname(triton.__file__))")
TRITON_OVERLAY=${TRITON_OVERLAY:-/scratch/pawsey1257/$USER/triton_overlay}
mkdir -p "$TRITON_OVERLAY" && ln -sfn "$SYS_TRITON" "$TRITON_OVERLAY/triton"
export PYTHONPATH="$TRITON_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_TRITON_FLASH_ATTN=1
python3 - <<'PY' || { echo "!! triton overlay check failed" >&2; exit 1; }
import torch, triton
print(f"triton {triton.__version__} ({triton.__file__}), torch {torch.__version__}")
assert not triton.__version__.startswith("3.8"), "userbase triton still first on sys.path"
PY

# Models are already in $HF_HUB_CACHE. vLLM 0.9.1 still lists repo files on
# the Hub at every engine start (get_pooling_config); six concurrent jobs
# from one Pawsey IP got HTTP 429 and four screening runs died (2026-10-31).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
