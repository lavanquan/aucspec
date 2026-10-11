#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --job-name=b100_6p0
#SBATCH --output=budget100_x6p0_%j.log
set -e
bash scripts/run_capacity_frontier.sh --resume --policies capacity_dpp --run-tag budget100_x6p0
echo "=== BUDGET100_6p0_DONE exit=0 ==="
