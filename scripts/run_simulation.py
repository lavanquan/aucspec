from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_specsim.simulator import EdgeSpecSimulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=["gsm8k", "math", "cnn_dailymail"])
    parser.add_argument("--num-questions", type=int)
    parser.add_argument("--split")
    parser.add_argument("--math-subject")
    parser.add_argument("--target-model")
    parser.add_argument("--draft-model")
    parser.add_argument("--num-clients", type=int, help="Override simulation.num_clients from the config")
    parser.add_argument("--detailed-log", action="store_true", help="Write full per-round token traces to results/rounds.csv")
    args = parser.parse_args()
    simulator = EdgeSpecSimulator(
        args.config,
        dataset_name=args.dataset,
        num_questions=args.num_questions,
        split=args.split,
        math_subject=args.math_subject,
        target_model=args.target_model,
        draft_model=args.draft_model,
        num_clients=args.num_clients,
        detailed_log=args.detailed_log,
    )
    asyncio.run(simulator.run())


if __name__ == "__main__":
    main()
