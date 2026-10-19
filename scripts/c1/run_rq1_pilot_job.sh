#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:8
#SBATCH --time=01:04:00
# RQ1 minimal pilot job: ONE run-seed, six concurrency levels run sequentially (N=8 first = gate), fixed gamma=2.
# Walltime 64 min = 3.2 node-h / 3 jobs (the 3.2 cap is a hard limit; see scripts/c1/budget_guard.py).
# Projected actual use of this job: ~31 min (causal factor 1) to ~48 min (factor 2); factor 3 would need 64 min.
# NOT SUBMITTED BY DEFAULT.  Preconditions: user approval; `canonical_launch.py init --run-tag <TAG> --config
# configs/rq1_pilot_fixed_gamma2_tritonfa33.yaml --planned-runs 18` done on the login node at a clean, pushed commit.
# usage: sbatch --job-name=rq1_s<S> --output=<scratch>/<TAG>/logs/seed<S>_%j.log scripts/c1/run_rq1_pilot_job.sh <WORKTREE> <TAG> <SEED> [first-job-gate]
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
for N in 8 1 2 4 12 16; do
  # do not start a run that cannot finish inside the walltime (conservative per-run need: 11 min)
  LEFT=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null | awk -F: '{ if (NF==3) print $1*60+$2; else if (NF==2) print $1; else print 0 }')
  if [ -n "$LEFT" ] && [ "$LEFT" -lt 11 ] && [ "$N" != "8" ]; then echo "=== STOP: only ${LEFT} min left; N=$N not started ===" >&2; exit 4; fi
  python3 scripts/canonical_launch.py run --run-tag "$RUN_TAG" --n "$N" --seed "$SEED"
  rc=$?
  echo "=== RUN_DONE n=$N seed=$SEED exit=$rc ==="
  [ $rc -ne 0 ] && { echo "=== STOP: run failed or INVALID ===" >&2; exit $rc; }
  CSV=$(python3 scripts/c1/pilot_gate.py --path "$CANONICAL_SCRATCH" "$RUN_TAG" "$N" "$SEED")
  python3 scripts/c1/pilot_gate.py "$CSV" "$N" || { echo "=== GATE_FAILED n=$N seed=$SEED ===" >&2; exit 3; }
done
echo "=== PILOT_JOB_DONE seed=$SEED ==="
