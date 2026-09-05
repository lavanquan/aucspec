"""Exp 1 driver: trace the AUC-controller frontier by sweeping the
INTERACTIVITY REQUIREMENT x (min_interactivity_tps / eq. 12's x, the
constraint threshold), not V -- V only controls how aggressively the
controller trades off goodput against how fast it satisfies that
constraint (Theorem 3); it is not the frontier's parameter (Definition 2,
Proposition 2iv say the frontier is a one-parameter sweep of the
CONSTRAINT). An earlier version of this script swept V with x fixed at an
infeasible 4.0 tok/s, which pinned every run near the same saturated
operating point regardless of V and produced a nearly-flat "frontier" --
see the git history for that mistake and the conversation that caught it.

Baselines (goodspeed/turbospec/fixed_slo/gelato/max_throughput) do not
depend on x at all, so their previously-collected real-GPU points are
reused unchanged (see BASELINE_POINTS_CSV) rather than re-run.

For each x in X_TARGETS and each seed, this:
  1. writes a per-run YAML overriding controller.min_interactivity_tps
     (and its simulation-level fallback) plus the seeds, on top of
     exp1_base.yaml (V fixed at FIXED_V for every point -- only x varies),
  2. runs legacy/scripts/run_simulation.py as a subprocess,
  3. reads back the resulting detailed rounds CSV, restricted to
     in_measurement_window rows, and computes each client's interactivity
     x_i and the system goodput Y = sum_i x_i,
  4. appends one row per (x, seed) to results/exp1_frontier/points_raw_shardN.csv,
     including whether min_i x_i actually met the requested x (constraint
     feasibility check).

After all shards: --aggregate merges them with the reused baseline points,
builds the frontier, computes AUC(xi) over the union of measured x values
(NOT literally eq.(6)'s [0, x_bar(xi)] unless the swept x values reach
both ends -- the summary reports the actual integration bounds used so
this is never silently overstated), and reports the y=x feasibility check
per point.
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

BASE_CONFIG_PATH = REPO_ROOT / "exp1_base.yaml"
OUT_DIR = REPO_ROOT / "results" / "exp1_frontier"
RAW_DIR = OUT_DIR / "raw"
SEEDS = [1, 2, 3]
FIXED_V = 100.0

# Chosen from the earlier (mis-swept) run's own evidence: every V from 1 to
# 1000 saturated around min_x ~= 1.57-1.61 tok/s under an infeasible
# x=4.0 requirement -- i.e. the system's real ceiling x_bar(xi) is
# apparently near there. Span comfortably below it up to past it, so the
# sweep covers both the feasible region and the point where it breaks.
X_TARGETS = [0.1, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0, 2.5]

# Baselines already collected with real GPU runs in the previous (V-swept)
# job -- they don't depend on x, so reuse rather than re-run.
BASELINE_LABELS = ["max_throughput", "goodspeed", "turbospec", "fixed_slo", "gelato"]


def build_run_config(base_cfg: dict, x_target: float, seed: int, output_csv: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["controller"]["policy"] = "adaptive_index"
    cfg["controller"]["V"] = float(FIXED_V)
    cfg["controller"]["min_interactivity_tps"] = float(x_target)
    cfg["simulation"]["min_interactivity_tps"] = float(x_target)
    cfg["simulation"]["controller_v"] = float(FIXED_V)
    cfg["simulation"]["seed"] = int(seed)
    cfg["dataset"]["seed"] = int(seed)
    cfg["simulation"]["output_csv"] = output_csv
    return cfg


def run_one(label: str, x_target: float, seed: int) -> Path:
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    output_csv = f"results/exp1_frontier/raw/{label}_seed{seed}.csv"
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


def measure_operating_point(rounds_csv: Path) -> tuple[float, float, dict[int, float]]:
    """Returns (Y, min_i x_i, per_client_x) from one run's detailed CSV."""
    df = pd.read_csv(rounds_csv, low_memory=False)
    df = df[df["in_measurement_window"].astype(bool)]
    if df.empty:
        raise ValueError(f"{rounds_csv}: no rows in the measurement window")
    x_per_client: dict[int, float] = {}
    for client_id, group in df.groupby("client_id"):
        duration_s = group["client_receive_ms"].max() / 1000.0
        useful = group["useful_tokens"].sum()
        x_per_client[int(client_id)] = float(useful) / max(1e-9, duration_s)
    y = sum(x_per_client.values())
    min_x = min(x_per_client.values())
    return y, min_x, x_per_client


def all_frontier_jobs() -> list[tuple[str, float, int]]:
    """Flat (label, x_target, seed) list for the AUC-controller x-sweep only."""
    return [(f"x_{x_target}", x_target, seed) for x_target in X_TARGETS for seed in SEEDS]


def run_shard(shard_index: int, num_shards: int) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [job for i, job in enumerate(all_frontier_jobs()) if i % num_shards == shard_index]
    points_path = OUT_DIR / f"points_frontier_shard{shard_index}.csv"
    rows = []
    for label, x_target, seed in jobs:
        rounds_csv = run_one(label, x_target, seed)
        y, min_x, _ = measure_operating_point(rounds_csv)
        rows.append(
            {
                "label": label,
                "policy": "adaptive_index",
                "x_target": x_target,
                "seed": seed,
                "Y": y,
                "min_x": min_x,
                "constraint_met": min_x >= x_target,
            }
        )
        print(f"  -> Y={y:.3f} min_x={min_x:.3f} (target={x_target}, met={min_x >= x_target})", flush=True)
        pd.DataFrame(rows).to_csv(points_path, index=False)
    return points_path


def load_baseline_points() -> pd.DataFrame:
    """Reuse the baseline arms' points from the previous run's aggregated
    points_raw.csv (they don't depend on x, so no need to re-run them)."""
    prior_path = OUT_DIR / "points_raw.csv"
    if not prior_path.exists():
        print(f"WARNING: {prior_path} not found -- no baseline points to reuse.", flush=True)
        return pd.DataFrame(columns=["label", "policy", "seed", "Y", "min_x"])
    df = pd.read_csv(prior_path)
    return df[df["label"].isin(BASELINE_LABELS)].copy()


def aggregate() -> None:
    shard_paths = sorted(glob.glob(str(OUT_DIR / "points_frontier_shard*.csv")))
    if not shard_paths:
        raise FileNotFoundError(f"no points_frontier_shard*.csv found under {OUT_DIR}")
    frontier_df = pd.concat([pd.read_csv(p) for p in shard_paths], ignore_index=True)
    frontier_df.to_csv(OUT_DIR / "points_frontier_raw.csv", index=False)

    baseline_df = load_baseline_points()

    frontier_summary_rows = []
    for x_target, group in frontier_df.groupby("x_target"):
        frontier_summary_rows.append(
            {
                "label": group["label"].iloc[0],
                "x_target": x_target,
                "n_seeds": len(group),
                "Y_mean": group["Y"].mean(),
                "Y_std": group["Y"].std(ddof=0),
                "min_x_mean": group["min_x"].mean(),
                "min_x_std": group["min_x"].std(ddof=0),
                "constraint_met_all_seeds": bool(group["constraint_met"].all()),
                "constraint_met_frac": group["constraint_met"].mean(),
            }
        )
    frontier_summary_df = pd.DataFrame(frontier_summary_rows).sort_values("x_target")
    frontier_summary_df.to_csv(OUT_DIR / "frontier_summary_x.csv", index=False)

    print("\n=== Feasibility check: min_x achieved vs x requested (should be >= for a valid frontier point) ===")
    for _, row in frontier_summary_df.iterrows():
        flag = "OK" if row["constraint_met_all_seeds"] else "VIOLATED"
        print(
            f"  x_target={row['x_target']:.2f}  min_x_mean={row['min_x_mean']:.3f}  "
            f"Y_mean={row['Y_mean']:.3f}  [{flag}, met in {row['constraint_met_frac']*100:.0f}% of seeds]"
        )

    baseline_summary_rows = []
    for label, group in baseline_df.groupby("label"):
        baseline_summary_rows.append(
            {
                "label": label,
                "policy": group["policy"].iloc[0],
                "n_seeds": len(group),
                "Y_mean": group["Y"].mean(),
                "Y_std": group["Y"].std(ddof=0),
                "min_x_mean": group["min_x"].mean(),
                "min_x_std": group["min_x"].std(ddof=0),
            }
        )
    baseline_summary_df = pd.DataFrame(baseline_summary_rows)
    baseline_summary_df.to_csv(OUT_DIR / "baseline_summary.csv", index=False)

    # Only feed feasible (constraint actually met) points into the frontier --
    # a point where the controller failed to reach its own requirement is
    # not a valid sample of Y*(x;xi).
    feasible = frontier_summary_df[frontier_summary_df["constraint_met_all_seeds"]]
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
    auc_summary = {
        "auc_over_measured_range": auc_value,
        "integration_bounds_x": [x_lo, x_hi],
        "note": (
            "This integrates only over the x range actually swept and found "
            "feasible above -- it is eq.(6)'s AUC(xi) = integral_0^x_bar only "
            "if x_lo is ~0 and x_hi is ~x_bar(xi). Check integration_bounds_x "
            "against the feasibility table before quoting this as AUC(xi)."
        ),
        "frontier_points": [{"min_x": p.min_interactivity, "Y": p.goodput, "label": p.label} for p in frontier],
    }
    with open(OUT_DIR / "auc_summary.json", "w", encoding="utf-8") as f:
        json.dump(auc_summary, f, indent=2)

    print(f"\nAUC over measured range {x_lo}..{x_hi} = {auc_value:.4f}")
    print("Baselines (reused from prior run):")
    print(baseline_summary_df.to_string(index=False))
    print(f"\nWrote points_frontier_raw.csv, frontier_summary_x.csv, baseline_summary.csv, auc_summary.json under {OUT_DIR}")


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
