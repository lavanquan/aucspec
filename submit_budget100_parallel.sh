#!/bin/bash
set -e
cd /software/projects/pawsey1257/quanla/aucspec

for x in 2.0 3.0 4.0 4.5 5.0 5.5 6.0; do
  tag=$(echo "$x" | tr '.' 'p')
  cfg="configs/capacity_frontier_budget100_x${tag}.yaml"
  runtag="budget100_x${tag}"
  script="run_budget100_x${tag}.sh"
  cat > "$script" << EOF
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --job-name=b100_${tag}
#SBATCH --output=budget100_x${tag}_%j.log
set -e
bash scripts/run_capacity_frontier.sh --search --policies capacity_dpp --search-axis n --config ${cfg} --run-tag ${runtag}
echo "=== BUDGET100_${tag}_DONE exit=0 ==="
EOF
  cat > "run_budget100_x${tag}_resume.sh" << EOF
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=pawsey1257-gpu
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --job-name=b100_${tag}
#SBATCH --output=budget100_x${tag}_%j.log
set -e
bash scripts/run_capacity_frontier.sh --resume --policies capacity_dpp --run-tag ${runtag}
echo "=== BUDGET100_${tag}_DONE exit=0 ==="
EOF
  echo "submitting $script"
  sbatch "$script"
done
