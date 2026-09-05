"""AUC frontier diagnostic (see AUC_FRONTIER_DIAGNOSTIC.md): compute the
Section 6 aggregate metrics for every x_requirement already run in
results/exp1_frontier/raw/x_*.csv (fcfs scheduler), to determine whether
the flat frontier is a saturated-verifier artifact, a missing x->controller
coupling, or a genuine structural property.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("/software/projects/pawsey1257/quanla/aucspec/results/exp1_frontier/raw")
OUT_CSV = Path("/software/projects/pawsey1257/quanla/aucspec/results/AUC_FRONTIER_DIAGNOSTIC_metrics.csv")


def parse_x_target(path: Path) -> float:
    m = re.match(r"x_([0-9.]+)_seed(\d+)\.csv", path.name)
    return float(m.group(1))


def parse_seed(path: Path) -> int:
    m = re.match(r"x_([0-9.]+)_seed(\d+)\.csv", path.name)
    return int(m.group(2))


def analyze_run(path: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)
    meas = df[df["in_measurement_window"].astype(bool)]
    if meas.empty:
        meas = df  # fall back if the measurement window filter is too strict

    # Per-client interactivity / goodput (same definition as run_exp1_frontier.py)
    x_per_client = {}
    for cid, g in meas.groupby("client_id"):
        duration_s = g["client_receive_ms"].max() / 1000.0
        x_per_client[cid] = g["useful_tokens"].sum() / max(1e-9, duration_s)
    y = sum(x_per_client.values())
    min_x = min(x_per_client.values()) if x_per_client else float("nan")

    total_elapsed_s = meas["client_receive_ms"].max() / 1000.0
    # One row per (batch, client) but batch-level fields repeat per client in
    # that batch; dedupe on verification_batch_id for batch-level stats.
    batches = meas.drop_duplicates(subset="verification_batch_id")
    verifier_tokens_per_s = batches["verification_batch_token_cost"].sum() / max(1e-9, total_elapsed_s)
    useful_per_verify_token = meas["useful_tokens"].sum() / max(1, batches["verification_batch_token_cost"].sum())
    fill_ratio = batches["verification_batch_token_cost"] / batches["verification_batch_token_budget"]
    underfilled_frac = (fill_ratio < 1.0).mean()

    return {
        "x_target": parse_x_target(path),
        "seed": parse_seed(path),
        "system_goodput": y,
        "achieved_x_min": min_x,
        "constraint_satisfied": min_x >= parse_x_target(path),
        "verifier_tokens_per_second": verifier_tokens_per_s,
        "useful_tokens_per_verification_token": useful_per_verify_token,
        "mean_batch_size": batches["verification_batch_size"].mean(),
        "mean_verification_tokens_per_batch": batches["verification_batch_token_cost"].mean(),
        "mean_fill_ratio": fill_ratio.mean(),
        "underfilled_batch_fraction": underfilled_frac,
        "mean_selected_gamma": meas["gamma"].mean(),
        "mean_server_queue": meas["server_queue"].mean(),
        "p95_server_queue": meas["server_queue"].quantile(0.95),
        "mean_z_queue": meas["z_queue"].mean(),
        "mean_device_queue": meas["device_queue"].mean(),
        "n_rounds": len(meas),
        "n_batches": len(batches),
    }


def main() -> None:
    paths = sorted(glob.glob(str(RAW_DIR / "x_*.csv")))
    rows = [analyze_run(Path(p)) for p in paths]
    df = pd.DataFrame(rows).sort_values(["x_target", "seed"])
    df.to_csv(OUT_CSV, index=False)

    summary = df.groupby("x_target").agg(
        n_seeds=("seed", "count"),
        Y_mean=("system_goodput", "mean"),
        min_x_mean=("achieved_x_min", "mean"),
        constraint_met_frac=("constraint_satisfied", "mean"),
        verifier_tok_s=("verifier_tokens_per_second", "mean"),
        useful_per_verify_tok=("useful_tokens_per_verification_token", "mean"),
        mean_batch_size=("mean_batch_size", "mean"),
        mean_fill_ratio=("mean_fill_ratio", "mean"),
        underfilled_frac=("underfilled_batch_fraction", "mean"),
        mean_gamma=("mean_selected_gamma", "mean"),
        mean_server_queue=("mean_server_queue", "mean"),
        mean_z_queue=("mean_z_queue", "mean"),
    ).reset_index()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(summary.to_string(index=False))
    summary.to_csv(str(OUT_CSV).replace(".csv", "_summary.csv"), index=False)


if __name__ == "__main__":
    main()
