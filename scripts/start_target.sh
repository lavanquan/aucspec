#!/usr/bin/env bash
set -euo pipefail

echo "This project now uses vLLM offline candidate verification inside the simulator."
echo "Do not start a separate vLLM server. Run:"
echo "  python scripts/run_simulation.py --config configs/default.yaml"
