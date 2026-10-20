#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
# RQ1 C1.4 FINAL PROBE job (exploratory protocol): ONE run-seed, N = 28 then 32 (only if 28 passes every validity gate and time remains), batch_cap_16.
# Walltime 60 min requested (cumulative cap 3.2 node-h incl. job 51179457 is checked by budget_guard before submission).
# NOT SUBMITTED BY DEFAULT.  Preconditions: user approval; `canonical_launch.py init --run-tag <TAG> --config
# configs/rq1_cap16_probe_fixed_gamma2_tritonfa33.yaml --planned-runs 2` done on the login node at a clean, pushed commit.
# usage: sbatch --job-name=rq1_s<S> --output=<scratch>/<TAG>/logs/seed<S>_%j.log scripts/c1/run_rq1_cap16_probe_job.sh <WORKTREE> <TAG> <SEED> [first-job-gate]
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
export GATE_PROTOCOL=exploratory_v2   # C1.4: real/virtual ratio is telemetry, all validity checks stay fail-closed
need_min() { case "$1" in 28) echo 15;; *) echo 17;; esac; }   # conservative: r extrapolated to 4.4 (N=28) and 5.0 (N=32), + startup/teardown
for N in ${N_LIST:-28 32}; do
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
