#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --job-name=b100_4p5
#SBATCH --output=budget100_x4p5_%j.log
set -e
bash scripts/run_capacity_frontier.sh --resume --policies capacity_dpp --run-tag budget100_x4p5
echo "=== BUDGET100_4p5_DONE exit=0 ==="
