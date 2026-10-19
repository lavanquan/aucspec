#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --time=0:35:00
# usage: sbatch --job-name=tfa33_qual --output=/scratch/pawsey1257/quanla/rocm_qualification/<tag>/slurm_%j.log \
#   <worktree>/scripts/run_rocm_qualification_job.sh <WORKTREE> <CONFIG> <TAG> <SMOKE_RUN_TAG> <FROZEN_CONTEXTS> <FROZEN_SHA256>
# Runtime V1-batchwall-tritonfa33. FAIL-CLOSED: exit 0 only if qualification (dev + held-out) AND the integration smoke pass.
# No pip installs; Triton shim first on PYTHONPATH; the real exit code is propagated to Slurm.
set -uo pipefail
WT="$1"; CFG="$2"; TAG="$3"; SMOKE_TAG="$4"; FROZEN="$5"; FROZEN_SHA="$6"
cd "$WT" || exit 66
module load pytorch/2.7.1-rocm6.3.3
PROJECT=/software/projects/pawsey1257/quanla/aucspec
SHIM=/software/projects/pawsey1257/quanla/aucspec_runtime/tritonfa33/shim
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
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
unset VLLM_ATTENTION_BACKEND VLLM_USE_V1 ALLOW_DEFECTIVE_LEGACY_RUNTIME
export VLLM_USE_TRITON_FLASH_ATTN=1
export PYTHONPATH="$SHIM${PYTHONPATH:+:$PYTHONPATH}"
export CANONICAL_SCRATCH=/scratch/pawsey1257/quanla/canonical_raw
OUT=/scratch/pawsey1257/quanla/rocm_qualification/"$TAG"
mkdir -p "$OUT"
date +%s > "$OUT/job_start_unix.txt"
HEAD_SHA="$(git rev-parse HEAD)"
python3 scripts/rocm_qual_orchestrate.py --config "$CFG" --out-dir "$OUT" --smoke-run-tag "$SMOKE_TAG" --smoke-n 8 \
  --reuse-frozen "$FROZEN" --expect-frozen-sha256 "$FROZEN_SHA" --expected-sha "$HEAD_SHA"
rc=$?
date +%s > "$OUT/job_end_unix.txt"
echo "=== ROCM_QUALIFICATION_DONE exit=$rc ==="
exit $rc
