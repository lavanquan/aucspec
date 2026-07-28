from __future__ import annotations

import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="results/rounds.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    duration_s = df["round_latency_ms"].sum() / 1000.0
    goodput = df["useful_tokens"].sum() / max(duration_s, 1e-9)
    print(f"rounds: {len(df)}")
    print(f"useful tokens: {df['useful_tokens'].sum()}")
    print(f"aggregate sequential-normalized goodput: {goodput:.3f} token/s")
    print(f"mean latency: {df['round_latency_ms'].mean():.1f} ms")
    print(f"mean gamma: {df['gamma'].mean():.2f}")
    print(f"mean accepted length: {df['accepted_length'].mean():.2f}")
    print("\nPer-client:")
    print(df.groupby("client_id").agg(
        rounds=("round_id", "count"),
        mean_gamma=("gamma", "mean"),
        accepted=("accepted_length", "sum"),
        useful=("useful_tokens", "sum"),
        mean_latency_ms=("round_latency_ms", "mean"),
    ).round(2))


if __name__ == "__main__":
    main()
