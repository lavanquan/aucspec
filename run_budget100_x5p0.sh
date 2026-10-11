#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --job-name=b100_5p0
#SBATCH --output=budget100_x5p0_%j.log
set -e
bash scripts/run_capacity_frontier.sh --search --policies capacity_dpp --search-axis n --config configs/capacity_frontier_budget100_x5p0.yaml --run-tag budget100_x5p0
echo "=== BUDGET100_5p0_DONE exit=0 ==="
