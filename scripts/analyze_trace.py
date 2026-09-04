"""Analyze results/trace_real.jsonl with sim/workloads/trace_loader.py --
TASKS.md T4.3: measure alpha_i(t) over a sliding window, check Assumption 1
(i.i.d. Bernoulli acceptance) via lag-1 autocorrelation, compare domains.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sim.workloads.trace_loader import (
    domain_alpha_summary,
    group_by_stream,
    lag1_autocorrelation,
    load_trace_jsonl,
    token_accept_flags,
    token_level_alpha_series,
)


def main() -> None:
    path = REPO_ROOT / "results" / "trace_real.jsonl"
    records = load_trace_jsonl(path)
    print(f"Loaded {len(records)} round-records from {path}")

    streams = group_by_stream(records)
    print(f"{len(streams)} streams: {sorted(streams.keys())}\n")

    print("=== Per-stream empirical acceptance rate + lag-1 autocorrelation ===")
    for stream_id, stream_records in sorted(streams.items()):
        flags = token_accept_flags(stream_records)
        n_success = sum(1 for f in flags if f)
        alpha_hat = n_success / len(flags) if flags else float("nan")
        ac = lag1_autocorrelation(flags)
        domain = stream_records[0].domain
        print(
            f"  {stream_id:10s} domain={domain:5s} rounds={len(stream_records):3d} "
            f"tokens={len(flags):4d} alpha_hat={alpha_hat:.3f} lag1_autocorr={ac:+.3f}"
        )

    print("\n=== Domain-level pooled acceptance rate ===")
    domain_alpha = domain_alpha_summary(records)
    for domain, alpha in sorted(domain_alpha.items()):
        print(f"  {domain}: alpha_hat = {alpha:.4f}")

    print("\n=== Pooled i.i.d. check (Assumption 1) ===")
    all_flags = token_accept_flags(records)
    pooled_ac = lag1_autocorrelation(all_flags)
    print(f"  All {len(all_flags)} tokens pooled: lag-1 autocorrelation = {pooled_ac:+.4f}")
    for domain in sorted(domain_alpha):
        domain_records = [r for r in records if r.domain == domain]
        domain_flags = token_accept_flags(domain_records)
        domain_ac = lag1_autocorrelation(domain_flags)
        print(f"  {domain} only ({len(domain_flags)} tokens): lag-1 autocorrelation = {domain_ac:+.4f}")

    print("\n=== Sliding-window alpha(t) drift, per stream (window=5 rounds' tokens) ===")
    for stream_id, stream_records in sorted(streams.items()):
        flags = token_accept_flags(stream_records)
        if len(flags) < 10:
            continue
        window_series = token_level_alpha_series(stream_records, window=10)
        # Print a compact trajectory sample (first, middle, last) instead of
        # the full per-token series.
        n = len(window_series)
        sample_idxs = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        traj = ", ".join(f"t{idx}={window_series[idx]:.2f}" for idx in sample_idxs)
        print(f"  {stream_id:10s}: {traj}")


if __name__ == "__main__":
    main()
