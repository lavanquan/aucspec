"""Exp 5 -- final confirmatory diagnostic: does AUC have a real mechanism?

Per AUC_FRONTIER_DIAGNOSTIC.md's conclusion (Case A: genuine near-fixed-
capacity redistribution at N=21/gamma_max=4, verified across 3 prior
confirmatory runs -- fixed batching scheduler+timing, forced real server
scarcity via verify_token_budget=6, added latency/bandwidth
heterogeneity), the frontier Y*(x) still came out flat. This is the final
diagnostic requested instead of running more baseline experiments:

  - Keep the scarce config (verify_token_budget=6, weighted_utility
    scheduler, batch_wait_ms=150) that already produces real batching
    contention (fill_ratio ~78-80% in the prior confirmatory run).
  - Add REAL acceptance heterogeneity: three equal-size groups of 7
    clients each draw from a different real dataset (gsm8k / cnn_dailymail
    summarization / competition MATH), so their draft-target acceptance
    rate alpha_i is genuinely different (measured, not injected as a
    synthetic Bernoulli parameter) -- see exp5_alpha_hetero_base.yaml's
    clients.client_classes and simulator.py's per-class dataset override.
  - Sweep the interactivity requirement x_req exactly as in Exp 1.
  - For each x, report THREE quantities instead of just Y(x):
      Y(x)      = sum_i x_i                       (system goodput)
      eta(x)    = useful_tokens / verified_tokens  (verification efficiency)
      share_c(x)= sum_{i in class c} x_i / Y(x)    (per-group service share)

If, as x increases, the low-alpha group's share grows, eta(x) falls, and
Y(x) falls -- that is direct evidence that the AUC objective is trading
off a real mechanism (which clients get served under a tightening
interactivity floor), not just re-labeling a fixed capacity split.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sim.metrics.auc import trapezoidal_auc
from sim.metrics.frontier import OperatingPoint, build_frontier

BASE_CONFIG_PATH = REPO_ROOT / "exp5_alpha_hetero_base.yaml"
OUT_DIR = REPO_ROOT / "results" / "exp5_alpha_hetero"
RAW_DIR = OUT_DIR / "raw"
SEEDS = [1, 2, 3]
FIXED_V = 100.0

# Same span as Exp 1 -- the measured ceiling there was around x~1.6.
X_TARGETS = [0.1, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0, 2.5]

CLASS_NAMES = ["easy_gsm8k", "medium_cnn_summarize", "hard_math"]


def build_run_config(base_cfg: dict, x_target: float, seed: int, output_csv: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["controller"]["policy"] = "adaptive_index"
    cfg["controller"]["V"] = float(FIXED_V)
    cfg["controller"]["min_interactivity_tps"] = float(x_target)
    cfg["simulation"]["min_interactivity_tps"] = float(x_target)
    cfg["simulation"]["controller_v"] = float(FIXED_V)
    cfg["simulation"]["seed"] = int(seed)
    cfg["simulation"]["output_csv"] = output_csv
    for name in CLASS_NAMES:
        cfg["clients"]["client_classes"][name]["dataset"]["seed"] = int(seed)
    return cfg


def run_one(label: str, x_target: float, seed: int) -> Path:
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    output_csv = f"results/exp5_alpha_hetero/raw/{label}_seed{seed}.csv"
    cfg = build_run_config(base_cfg, x_target, seed, output_csv)
    run_cfg_path = RAW_DIR / f"{label}_seed{seed}.yaml"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(run_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    print(f"=== running {label} seed={seed} (x_target={x_target}, V={FIXED_V}) ===", flush=True)
    result = subprocess.run(
        [sys.executable, "legacy/scripts/run_simulation.py", "--config", str(run_cfg_path), "--detailed-log"],
        cwd=str(REPO_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"run_simulation.py failed for {label} seed={seed} (exit {result.returncode})")
    return REPO_ROOT / output_csv


def measure_operating_point(rounds_csv: Path) -> dict:
    df = pd.read_csv(rounds_csv, low_memory=False)
    meas = df[df["in_measurement_window"].astype(bool)]
    if meas.empty:
        raise ValueError(f"{rounds_csv}: no rows in the measurement window")

    class_of_client = meas.groupby("client_id")["draft_device_class"].first().to_dict()

    x_per_client: dict[int, float] = {}
    for client_id, group in meas.groupby("client_id"):
        duration_s = group["client_receive_ms"].max() / 1000.0
        useful = group["useful_tokens"].sum()
        x_per_client[int(client_id)] = float(useful) / max(1e-9, duration_s)
    y = sum(x_per_client.values())
    min_x = min(x_per_client.values())

    # eq. (6)'s denominator proxy: eta(x) = useful tokens / verified tokens,
    # same definition as diagnose_frontier.py's useful_tokens_per_verification_token.
    batches = meas.drop_duplicates(subset=["verification_batch_id"])
    verified_tokens_total = batches["verification_batch_token_cost"].sum()
    useful_total = meas["useful_tokens"].sum()
    eta = float(useful_total) / max(1.0, float(verified_tokens_total))

    x_by_class = {name: 0.0 for name in CLASS_NAMES}
    for client_id, x_i in x_per_client.items():
        cls = class_of_client.get(client_id, "unknown")
        x_by_class[cls] = x_by_class.get(cls, 0.0) + x_i
    share_by_class = {cls: v / max(1e-9, y) for cls, v in x_by_class.items()}

    return {
        "Y": y,
        "min_x": min_x,
        "eta": eta,
        **{f"share_{cls}": share_by_class.get(cls, 0.0) for cls in CLASS_NAMES},
        **{f"x_{cls}": x_by_class.get(cls, 0.0) for cls in CLASS_NAMES},
    }


def all_frontier_jobs() -> list[tuple[str, float, int]]:
    return [(f"x_{x_target}", x_target, seed) for x_target in X_TARGETS for seed in SEEDS]


def run_shard(shard_index: int, num_shards: int) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [job for i, job in enumerate(all_frontier_jobs()) if i % num_shards == shard_index]
    points_path = OUT_DIR / f"points_shard{shard_index}.csv"
    rows = []
    for label, x_target, seed in jobs:
        rounds_csv = run_one(label, x_target, seed)
        metrics = measure_operating_point(rounds_csv)
        row = {
            "label": label,
            "policy": "adaptive_index",
            "x_target": x_target,
            "seed": seed,
            "constraint_met": metrics["min_x"] >= x_target,
            **metrics,
        }
        rows.append(row)
        print(
            f"  -> Y={metrics['Y']:.3f} min_x={metrics['min_x']:.3f} eta={metrics['eta']:.3f} "
            f"shares={ {c: round(metrics[f'share_{c}'], 3) for c in CLASS_NAMES} } "
            f"(target={x_target}, met={metrics['min_x'] >= x_target})",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(points_path, index=False)
    return points_path


def aggregate() -> None:
    shard_paths = sorted(glob.glob(str(OUT_DIR / "points_shard*.csv")))
    if not shard_paths:
        raise FileNotFoundError(f"no points_shard*.csv found under {OUT_DIR}")
    raw_df = pd.concat([pd.read_csv(p) for p in shard_paths], ignore_index=True)
    raw_df.to_csv(OUT_DIR / "points_raw.csv", index=False)

    agg_cols = {"Y": "mean", "min_x": "mean", "eta": "mean", "constraint_met": "mean"}
    for cls in CLASS_NAMES:
        agg_cols[f"share_{cls}"] = "mean"

    summary_rows = []
    for x_target, group in raw_df.groupby("x_target"):
        row = {"x_target": x_target, "n_seeds": len(group)}
        for col, how in agg_cols.items():
            row[f"{col}_mean"] = group[col].mean()
        row["Y_std"] = group["Y"].std(ddof=0)
        row["constraint_met_all_seeds"] = bool(group["constraint_met"].all())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("x_target")
    summary_df.to_csv(OUT_DIR / "summary_x.csv", index=False)

    print("\n=== Exp5 diagnostic: Y(x), eta(x), and per-group service share vs x ===")
    for _, row in summary_df.iterrows():
        flag = "OK" if row["constraint_met_all_seeds"] else "VIOLATED"
        shares = ", ".join(f"{c}={row[f'share_{c}_mean']:.3f}" for c in CLASS_NAMES)
        print(
            f"  x={row['x_target']:.2f}  Y={row['Y_mean']:.3f}  eta={row['eta_mean']:.3f}  "
            f"shares[{shares}]  min_x={row['min_x_mean']:.3f} [{flag}]"
        )

    feasible = summary_df[summary_df["constraint_met_all_seeds"]]
    points = [
        OperatingPoint(min_interactivity=row["min_x_mean"], goodput=row["Y_mean"], label=f"x_{row['x_target']}")
        for _, row in feasible.iterrows()
    ]
    frontier = build_frontier(points)
    if len(frontier) >= 2:
        auc_value = trapezoidal_auc(frontier)
        x_lo, x_hi = frontier[0].min_interactivity, frontier[-1].min_interactivity
    else:
        auc_value = 0.0
        x_lo = x_hi = None

    # Mechanism verdict: does low-alpha (hard_math, presumptively lowest
    # acceptance) share increase with x while eta and Y fall?
    if len(summary_df) >= 2:
        lo, hi = summary_df.iloc[0], summary_df.iloc[-1]
        hard_share_rises = bool(hi["share_hard_math_mean"] > lo["share_hard_math_mean"])
        eta_falls = bool(hi["eta_mean"] < lo["eta_mean"])
        y_falls = bool(hi["Y_mean"] < lo["Y_mean"])
        mechanism_verdict = bool(hard_share_rises and eta_falls and y_falls)
    else:
        hard_share_rises = eta_falls = y_falls = mechanism_verdict = None

    out = {
        "auc_over_measured_range": auc_value,
        "integration_bounds_x": [x_lo, x_hi],
        "mechanism_verdict": mechanism_verdict,
        "hard_math_share_rises_with_x": hard_share_rises,
        "eta_falls_with_x": eta_falls,
        "Y_falls_with_x": y_falls,
        "frontier_points": [{"min_x": p.min_interactivity, "Y": p.goodput, "label": p.label} for p in frontier],
    }
    with open(OUT_DIR / "mechanism_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nAUC over measured range {x_lo}..{x_hi} = {auc_value:.4f}")
    print(f"Mechanism verdict (share up + eta down + Y down as x increases): {mechanism_verdict}")
    print(f"\nWrote points_raw.csv, summary_x.csv, mechanism_summary.json under {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    if args.aggregate:
        aggregate()
        return
    if args.shard_index is None:
        raise SystemExit("pass --shard-index N --num-shards M, or --aggregate after all shards finish")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard-index must be in [0, num_shards)")
    run_shard(args.shard_index, args.num_shards)


if __name__ == "__main__":
    main()
