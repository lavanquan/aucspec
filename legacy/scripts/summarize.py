from __future__ import annotations

import argparse
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_specsim.metrics import summarize_round_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="results/rounds.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    summary = summarize_round_csv(df)
    per_sample = summary["per_sample"]
    per_client = summary["per_client"]
    print(f"rounds: {len(df)}")
    print(f"useful tokens: {summary['useful_tokens']}")
    print(f"accepted tokens: {summary['accepted_tokens']}")
    print(f"goodput: {summary['goodput_tps']:.3f} token/s")
    print(f"accepted-token goodput: {summary['accepted_token_goodput_tps']:.3f} token/s")
    print(f"completion time: {summary['completion_time_ms']:.1f} ms")
    print(f"mean TTFT: {summary['mean_ttft_ms']:.1f} ms")
    print(f"mean inter-token latency: {summary['mean_inter_token_latency_ms']:.1f} ms")
    print(f"server utilization: {summary['server_utilization']:.3f}")
    print(f"draft GPU utilization mean: {summary['draft_gpu_utilization_mean']:.3f}")
    print(f"draft GPU utilization max: {summary['draft_gpu_utilization_max']:.3f}")
    print(f"acceptance rate: {summary['acceptance_rate']:.3f}")
    print(f"straggler ratio: {summary['straggler_ratio']:.3f}")
    print(f"mean latency: {summary['mean_round_latency_ms']:.1f} ms")
    print(f"mean sample interactivity: {summary['mean_sample_interactivity_s_per_token']:.4f} s/token")
    print(f"mean client interactivity: {summary['mean_client_interactivity_s_per_token']:.4f} s/token")
    print(f"min client interactivity: {summary['min_client_interactivity_s_per_token']:.4f} s/token")
    print(f"max client interactivity: {summary['max_client_interactivity_s_per_token']:.4f} s/token")
    print(f"interactivity Jain fairness: {summary['interactivity_jain_fairness']:.4f}")
    print(
        "interactivity proportional fairness utility: "
        f"{summary['interactivity_proportional_fairness_utility']:.4f}"
    )
    print(f"mean gamma: {df['gamma'].mean():.2f}")
    print(f"mean accepted length: {df['accepted_length'].mean():.2f}")
    print("\nPer-client:")
    per_client_summary = (
        df.groupby("client_id")
        .agg(
            rounds=("round_id", "count"),
            mean_gamma=("gamma", "mean"),
            accepted=("accepted_length", "sum"),
            useful=("useful_tokens", "sum"),
            mean_latency_ms=("round_latency_ms", "mean"),
        )
        .reset_index()
        .merge(
            per_client[
                [
                    "client_id",
                    "samples",
                    "active_e2e_ms",
                    "interactivity_s_per_token",
                    "goodput_tps",
                    "accepted_token_goodput_tps",
                    "mean_ttft_ms",
                    "mean_inter_token_latency_ms",
                ]
            ],
            on="client_id",
            how="left",
        )
    )
    print(per_client_summary.round(4))


if __name__ == "__main__":
    main()
