"""Build the fixed empirical acceptance profile for the KnownAlpha ablation
(AUC_SPEC Section 4.1.2) from one or more rounds CSVs of earlier runs.

For every speculative round (gamma > 0) with `m` inspected positions and
`a` accepted tokens, positions 1..a are successes and position a+1 is a
failure if a < m; positions past m are censored (no evidence). The output
alpha_k is the censored conditional acceptance estimate s_k / (s_k + f_k),
per client (nested populations keep client ids stable across N) and pooled.

Usage:
    python scripts/build_known_alpha_profile.py RUN1.csv [RUN2.csv ...] \
        --gamma-max 4 --output configs/known_alpha_profile.json [--pooled]

Then set `controller.policy: capacity_dpp_known_alpha` and
`controller.known_alpha_profile_path: configs/known_alpha_profile.json`.
Use runs disjoint from the ones the ablation is evaluated on.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _profile(successes: dict[int, float], failures: dict[int, float], gamma_max: int) -> list[float] | None:
    values: list[float] = []
    for k in range(1, gamma_max + 1):
        observed = successes.get(k, 0.0) + failures.get(k, 0.0)
        if observed > 0:
            values.append(successes.get(k, 0.0) / observed)
        elif values:
            values.append(values[-1])  # never inspected: reuse the deepest observed position
        else:
            return None
    return values


def build_profiles(frames: list[pd.DataFrame], gamma_max: int, measurement_only: bool) -> dict:
    succ: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    fail: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for df in frames:
        if measurement_only and "in_measurement_window" in df.columns:
            df = df[df["in_measurement_window"].astype(bool)]
        df = df[df["gamma"] > 0]
        inspected_col = "acceptance_observed_tokens" if "acceptance_observed_tokens" in df.columns else "gamma"
        for cid, gamma, accepted, inspected in zip(
            df["client_id"].astype(int), df["gamma"].astype(int),
            df["accepted_length"].astype(int), df[inspected_col].astype(int),
        ):
            m = max(0, min(int(inspected), int(gamma)))
            a = max(0, min(int(accepted), m))
            for k in range(1, a + 1):
                succ[cid][k] += 1.0
            if a < m:
                fail[cid][a + 1] += 1.0

    per_client = {}
    pooled_s: dict[int, float] = defaultdict(float)
    pooled_f: dict[int, float] = defaultdict(float)
    for cid in sorted(set(succ) | set(fail)):
        prof = _profile(succ[cid], fail[cid], gamma_max)
        if prof is not None:
            per_client[int(cid)] = prof
        for k, v in succ[cid].items():
            pooled_s[k] += v
        for k, v in fail[cid].items():
            pooled_f[k] += v
    pooled = _profile(pooled_s, pooled_f, gamma_max)
    if pooled is None:
        raise SystemExit("no speculative rounds found: cannot build a profile")
    return {"per_client": per_client, "pooled": pooled}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", help="rounds CSV(s) written by run_simulation.py --detailed-log")
    p.add_argument("--gamma-max", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pooled", action="store_true",
                   help="write one profile shared by all clients instead of per-client profiles")
    p.add_argument("--include-warmup", action="store_true",
                   help="also use rounds outside the measurement window")
    args = p.parse_args()

    frames = [pd.read_csv(path, low_memory=False) for path in args.csv]
    out = build_profiles(frames, args.gamma_max, measurement_only=not args.include_warmup)
    payload = out["pooled"] if args.pooled else {"per_client": out["per_client"], "pooled": out["pooled"]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"pooled profile: {[round(v, 4) for v in out['pooled']]}")
    print(f"clients with a profile: {len(out['per_client'])}; wrote {args.output}")


if __name__ == "__main__":
    main()
