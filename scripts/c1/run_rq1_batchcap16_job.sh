#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:8
#SBATCH --time=01:15:00
# RQ1 BATCH_CAP_16 job (C1.3, exploratory): ONE run-seed, N = 8 (burn-in, diagnostic only) 16 24 32 sequentially, fixed gamma=2, max_batch_size=16.
# Walltime 75 min requested (cumulative cap 3.2 node-h incl. job 51179457 is checked by budget_guard before submission).
# NOT SUBMITTED BY DEFAULT.  Preconditions: user approval; `canonical_launch.py init --run-tag <TAG> --config
# configs/rq1_batchcap16_fixed_gamma2_tritonfa33.yaml --planned-runs 4` done on the login node at a clean, pushed commit.
# usage: sbatch --job-name=rq1_s<S> --output=<scratch>/<TAG>/logs/seed<S>_%j.log scripts/c1/run_rq1_batchcap16_job.sh <WORKTREE> <TAG> <SEED> [first-job-gate]
set -uo pipefail
WT="$1"; RUN_TAG="$2"; SEED="$3"
cd "$WT" || exit 66
module load pytorch/2.7.1-rocm6.3.3
PROJECT=/software/projects/pawsey1257/quanla/aucspec
SHIM=/software/projects/pawsey1257/quanla/aucspec_runtime/tritonfa33/shim
export PYTHONUSERBASE="$PROJECT/.pythonuserbase-rocm-vllm"
export TRITON_CACHE_DIR="$PROJECT/.triton-cache" XDG_CACHE_HOME="$PROJECT/.cache" HF_HOME="$PROJECT/.hf-home"
export HF_HUB_CACHE="$PROJECT/models" TRANSFORMERS_CACHE="$PROJECT/models" HF_DATASETS_CACHE="$PROJECT/.hf-datasets-cache"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT/.torchinductor-cache" MPLCONFIGDIR="$PROJECT/.config/matplotlib" XDG_CONFIG_HOME="$PROJECT/.config"
export VLLM_NO_USAGE_STATS=1 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
unset VLLM_ATTENTION_BACKEND VLLM_USE_V1 ALLOW_DEFECTIVE_LEGACY_RUNTIME
export VLLM_USE_TRITON_FLASH_ATTN=1
export PYTHONPATH="$SHIM${PYTHONPATH:+:$PYTHONPATH}"
export CANONICAL_SCRATCH=/scratch/pawsey1257/quanla/canonical_raw
python3 scripts/preflight_runtime.py || { echo "=== PREFLIGHT_FAILED ===" >&2; exit 65; }
export EXPECT_MAX_BATCH=16
need_min() { case "$1" in 8) echo 9;; 16) echo 11;; 24) echo 12;; *) echo 13;; esac; }
for N in ${N_LIST:-8 16 24 32}; do
  # never start a run that cannot finish inside the walltime (per-N conservative need): abort the job instead of being killed mid-run
  LEFT=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null | awk -F: '{ if (NF==3) print $1*60+$2; else if (NF==2) print $1; else print 0 }')
  if [ -n "$LEFT" ] && [ "$LEFT" -lt "$(need_min $N)" ]; then echo "=== STOP: only ${LEFT} min left; N=$N not started ===" >&2; exit 4; fi
  python3 scripts/canonical_launch.py run --run-tag "$RUN_TAG" --n "$N" --seed "$SEED"
  rc=$?
  echo "=== RUN_DONE n=$N seed=$SEED exit=$rc ==="
  [ $rc -ne 0 ] && { echo "=== STOP: run failed or INVALID ===" >&2; exit $rc; }
  CSV=$(python3 scripts/c1/pilot_gate.py --path "$CANONICAL_SCRATCH" "$RUN_TAG" "$N" "$SEED")
  python3 scripts/c1/pilot_gate.py "$CSV" "$N" || { echo "=== GATE_FAILED n=$N seed=$SEED ===" >&2; exit 3; }
done
echo "=== PILOT_JOB_DONE seed=$SEED ==="
