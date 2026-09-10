"""AUC_ACHIEVABLE_REGION_DIAGNOSTIC.md driver: probe the reachable
(x_min, Y) operating region directly, independent of the AUC/adaptive_index
controller's own x_requirement response.

Section 11 of that doc is the point: the flat Y_controller(x) measured in
Exp5 (results/AUC_FRONTIER_DIAGNOSTIC_REPORT.md Section 11) only shows
that the CURRENT controller does not move when x_requirement tightens. It
does not show that no feasible policy could reach a higher x_min. This
script instead sweeps a diagnostic-only server-side priority multiplier
m_hard on the hard-MATH client class (clients.diagnostic_priority_multiplier,
applied to the verification weight w_i = m_c(i) * (V + Z_i) in
simulator.py's submit() call -- see that file's comment at the call site)
while holding x_requirement fixed at a benign, non-binding 0.1 and leaving
every other deployment parameter exactly as in the Exp5 diagnostic
(N=21, three equal dataset-based client groups, verify_token_budget=6,
weighted_utility scheduler, batch_wait_ms=150). The same
clients.diagnostic_priority_multiplier lever does not touch acceptance,
draft/network latency, dataset sampling, reward accounting, x_requirement,
or the virtual-queue update equations -- only which requests the verifier
batcher prioritizes.

Sweep A (this script's default): m_easy=m_medium=1, m_hard in
{1,2,4,8,16,32,64}, 3 seeds each, same seeds/prompt assignments as the
baseline (random.seed(simulation.seed) is called before class/dataset
assignment in EdgeSpecSimulator.__init__, and diagnostic_priority_multiplier
does not consume any random draws, so class membership and per-client
datasets are identical across every m_hard value for a given seed).

Sweep B (--sweep b): a small (m_medium, m_hard) grid, run only if Sweep A
raises the hard group's service share without clearly maximizing x_min.
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

BASE_CONFIG_PATH = REPO_ROOT / "configs" / "auc_achievable_region_base.yaml"
OUT_DIR = REPO_ROOT / "results" / "auc_achievable_region"
RAW_DIR = OUT_DIR / "raw"
SEEDS = [1, 2, 3]
X_REQ_FIXED = 0.1
FIXED_V = 100.0

CLASS_NAMES = ["easy_gsm8k", "medium_cnn_summarize", "hard_math"]

M_HARD_SWEEP_A = [1, 2, 4, 8, 16, 32, 64]
# Sweep "am": the corrected Sweep A. The empirical bottleneck class (lowest
# measured alpha_hat ~0.77, lowest x_min) is medium_cnn_summarize, NOT
# hard_math -- hard_math actually has the HIGHEST alpha_hat (~0.93) and
# highest x_min, so boosting its priority (Sweep A) can never lift the
# system min_x. Sweep "am" boosts the class that actually sets the floor.
M_MEDIUM_SWEEP_AM = [1, 2, 4, 8, 16, 32, 64]
M_MEDIUM_SWEEP_B = [1, 2, 4, 8]
M_HARD_SWEEP_B = [1, 2, 4, 8, 16, 32]


def all_jobs(sweep: str) -> list[tuple[str, float, float, int]]:
    """Returns (label, m_medium, m_hard, seed) tuples; m_easy is always 1."""
    if sweep == "a":
        return [
            (f"mhard_{m_hard}", 1.0, float(m_hard), seed)
            for m_hard in M_HARD_SWEEP_A
            for seed in SEEDS
        ]
    if sweep == "am":
        return [
            (f"mmed_{m_medium}", float(m_medium), 1.0, seed)
            for m_medium in M_MEDIUM_SWEEP_AM
            for seed in SEEDS
        ]
    if sweep == "b":
        return [
            (f"mmed_{m_medium}_mhard_{m_hard}", float(m_medium), float(m_hard), seed)
            for m_medium in M_MEDIUM_SWEEP_B
            for m_hard in M_HARD_SWEEP_B
            for seed in SEEDS
        ]
    raise ValueError(f"unknown sweep {sweep!r}")


def build_run_config(
    base_cfg: dict,
    m_medium: float,
    m_hard: float,
    seed: int,
    output_csv: str,
    verify_token_budget: int | None = None,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["controller"]["policy"] = "adaptive_index"
    cfg["controller"]["V"] = float(FIXED_V)
    cfg["controller"]["min_interactivity_tps"] = float(X_REQ_FIXED)
    cfg["simulation"]["min_interactivity_tps"] = float(X_REQ_FIXED)
    cfg["simulation"]["controller_v"] = float(FIXED_V)
    cfg["simulation"]["seed"] = int(seed)
    cfg["simulation"]["output_csv"] = output_csv
    if verify_token_budget is not None:
        cfg["verification_batching"]["verify_token_budget"] = int(verify_token_budget)
    cfg["clients"]["diagnostic_priority_multiplier"] = {
        "easy_gsm8k": 1.0,
        "medium_cnn_summarize": float(m_medium),
        "hard_math": float(m_hard),
    }
    for name in CLASS_NAMES:
        cfg["clients"]["client_classes"][name]["dataset"]["seed"] = int(seed)
    return cfg


def run_one(
    label: str,
    m_medium: float,
    m_hard: float,
    seed: int,
    verify_token_budget: int | None = None,
) -> Path:
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    output_csv = f"results/auc_achievable_region/raw/{label}_seed{seed}.csv"
    cfg = build_run_config(base_cfg, m_medium, m_hard, seed, output_csv, verify_token_budget)
    run_cfg_path = RAW_DIR / f"{label}_seed{seed}.yaml"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(run_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    print(f"=== running {label} seed={seed} (m_medium={m_medium}, m_hard={m_hard}) ===", flush=True)
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
    total_elapsed_s = meas["client_receive_ms"].max() / 1000.0

    # verifier_token_cost per (client, round) row = gamma + 1 -- this is the
    # request's OWN cost within whatever batch it landed in (NOT the
    # verification_batch_token_cost column, which is a batch-level total
    # repeated across every client row in that batch).
    meas = meas.copy()
    meas["own_verifier_token_cost"] = meas["gamma"].astype(float) + 1.0
    meas["own_wait_ms"] = meas["batch_start_ms"] - meas["server_arrival_ms"]

    x_per_client: dict[int, float] = {}
    for client_id, group in meas.groupby("client_id"):
        duration_s = group["client_receive_ms"].max() / 1000.0
        useful = group["useful_tokens"].sum()
        x_per_client[int(client_id)] = float(useful) / max(1e-9, duration_s)
    y = sum(x_per_client.values())
    x_values = sorted(x_per_client.values())
    x_min = x_values[0]
    x_p10 = pd.Series(x_values).quantile(0.10)
    x_median = pd.Series(x_values).median()
    x_max = x_values[-1]

    batches = meas.drop_duplicates(subset=["verification_batch_id"])
    verified_tokens_total = batches["verification_batch_token_cost"].sum()
    useful_total = meas["useful_tokens"].sum()
    eta = float(useful_total) / max(1.0, float(verified_tokens_total))
    verifier_tokens_per_s = float(verified_tokens_total) / max(1e-9, total_elapsed_s)
    fill_ratio = (batches["verification_batch_token_cost"] / batches["verification_batch_token_budget"]).mean()
    mean_batch_size = batches["verification_batch_size"].mean()
    mean_selected_gamma = meas["gamma"].mean()

    out: dict = {
        "Y": y,
        "min_x": x_min,
        "x_p10": x_p10,
        "x_median": x_median,
        "x_max": x_max,
        "eta": eta,
        "verifier_tokens_per_s": verifier_tokens_per_s,
        "fill_ratio": fill_ratio,
        "mean_batch_size": mean_batch_size,
        "target_utilization": fill_ratio,  # proxy: fraction of the fixed verify_token_budget actually used
        "mean_selected_gamma": mean_selected_gamma,
    }

    total_verifier_tokens = meas["own_verifier_token_cost"].sum()
    total_useful = meas["useful_tokens"].sum()
    for cls in CLASS_NAMES:
        client_ids = [cid for cid, c in class_of_client.items() if c == cls]
        cls_x = [x_per_client[cid] for cid in client_ids if cid in x_per_client]
        g = meas[meas["draft_device_class"] == cls]
        out[f"share_{cls}"] = (sum(cls_x) / y) if y > 0 else 0.0
        out[f"x_mean_{cls}"] = (sum(cls_x) / len(cls_x)) if cls_x else float("nan")
        out[f"x_min_{cls}"] = min(cls_x) if cls_x else float("nan")
        out[f"alpha_hat_mean_{cls}"] = g["alpha_hat"].mean()
        out[f"alpha_ucb_mean_{cls}"] = g["alpha_ucb"].mean()
        out[f"gamma_mean_{cls}"] = g["gamma"].mean()
        out[f"verification_opportunities_{cls}"] = len(g)
        cls_verifier_tokens = g["own_verifier_token_cost"].sum()
        out[f"verifier_token_positions_{cls}"] = cls_verifier_tokens
        out[f"verifier_token_share_{cls}"] = (
            cls_verifier_tokens / total_verifier_tokens if total_verifier_tokens > 0 else 0.0
        )
        cls_useful = g["useful_tokens"].sum()
        out[f"committed_useful_tokens_{cls}"] = cls_useful
        out[f"committed_useful_token_share_{cls}"] = cls_useful / total_useful if total_useful > 0 else 0.0
        out[f"avg_wait_ms_{cls}"] = g["own_wait_ms"].mean()
        out[f"avg_z_queue_{cls}"] = g["z_queue"].mean()

    return out


def run_all(
    sweep: str,
    shard_index: int | None,
    num_shards: int,
    verify_token_budget: int | None = None,
    tag: str = "",
) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    jobs = all_jobs(sweep)
    suffix = f"{tag}" if tag else ""
    if shard_index is not None:
        points_path = OUT_DIR / f"points_sweep{sweep}{suffix}_shard{shard_index}.csv"
        jobs = [job for i, job in enumerate(jobs) if i % num_shards == shard_index]
    else:
        points_path = OUT_DIR / f"points_sweep{sweep}{suffix}.csv"
    rows = []
    for label, m_medium, m_hard, seed in jobs:
        run_label = f"{label}{suffix}"
        rounds_csv = run_one(run_label, m_medium, m_hard, seed, verify_token_budget)
        metrics = measure_operating_point(rounds_csv)
        row = {
            "label": run_label,
            "sweep": f"{sweep}{suffix}",
            "m_easy": 1.0,
            "m_medium": m_medium,
            "m_hard": m_hard,
            "seed": seed,
            "verify_token_budget": verify_token_budget if verify_token_budget is not None else 6,
            **metrics,
        }
        rows.append(row)
        print(
            f"  -> Y={metrics['Y']:.3f} min_x={metrics['min_x']:.3f} eta={metrics['eta']:.3f} "
            f"share_hard={metrics['share_hard_math']:.3f} verifier_tok_share_hard={metrics['verifier_token_share_hard_math']:.3f}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(points_path, index=False)
    return points_path


def aggregate(sweep: str) -> None:
    shard_paths = sorted(glob.glob(str(OUT_DIR / f"points_sweep{sweep}_shard*.csv")))
    single_path = OUT_DIR / f"points_sweep{sweep}.csv"
    if shard_paths:
        raw_df = pd.concat([pd.read_csv(p) for p in shard_paths], ignore_index=True)
    elif single_path.exists():
        raw_df = pd.read_csv(single_path)
    else:
        raise FileNotFoundError(f"no points_sweep{sweep}*.csv found under {OUT_DIR}")

    raw_path = OUT_DIR / "points_raw.csv"
    if raw_path.exists():
        prior = pd.read_csv(raw_path)
        raw_df = pd.concat([prior[prior["sweep"] != sweep], raw_df], ignore_index=True)
    raw_df.to_csv(raw_path, index=False)

    group_cols = ["sweep", "m_easy", "m_medium", "m_hard"]
    numeric_cols = [c for c in raw_df.columns if c not in group_cols + ["label", "seed"]]
    summary_df = raw_df.groupby(group_cols)[numeric_cols].mean().reset_index()
    summary_df["n_seeds"] = raw_df.groupby(group_cols)["seed"].count().values
    summary_df.to_csv(OUT_DIR / "points_summary.csv", index=False)

    # Empirical upper envelope Y_hat_star(x) = max over policies with
    # achieved x_min >= x, evaluated at every observed x_min value (no
    # spline interpolation, per Section 8 Figure 2).
    sorted_by_x = summary_df.sort_values("min_x")
    envelope_rows = []
    for _, row in sorted_by_x.iterrows():
        x0 = row["min_x"]
        feasible = summary_df[summary_df["min_x"] >= x0]
        if feasible.empty:
            continue
        best = feasible.loc[feasible["Y"].idxmax()]
        envelope_rows.append(
            {
                "x": x0,
                "Y_hat_star": best["Y"],
                "achieving_label": f"m_hard={best['m_hard']:.0f},m_medium={best['m_medium']:.0f}",
            }
        )
    envelope_df = pd.DataFrame(envelope_rows).drop_duplicates(subset=["x"]).sort_values("x")
    envelope_df.to_csv(OUT_DIR / "frontier_points.csv", index=False)

    print(f"\n=== AUC achievable-region diagnostic: sweep {sweep} summary ===")
    cols = ["m_medium", "m_hard", "n_seeds", "Y", "min_x", "eta", "share_easy_gsm8k", "share_medium_cnn_summarize", "share_hard_math", "verifier_token_share_hard_math"]
    print(summary_df[cols].sort_values(["m_medium", "m_hard"]).to_string(index=False))
    print("\n=== Empirical upper envelope Y_hat_star(x) ===")
    print(envelope_df.to_string(index=False))

    # Mechanism check. Sweep "a" boosts hard_math and varies m_hard;
    # sweep "am" boosts medium_cnn_summarize (the empirical bottleneck) and
    # varies m_medium. Pick the target class and swept multiplier column
    # accordingly.
    if sweep.startswith("am"):
        target_cls, mult_col = "medium_cnn_summarize", "m_medium"
    else:
        target_cls, mult_col = "hard_math", "m_hard"
    swept = summary_df[summary_df["sweep"] == sweep].sort_values(mult_col)
    mechanism_summary: dict = {}
    if len(swept) >= 2:
        lo, hi = swept.iloc[0], swept.iloc[-1]
        vshare_col = f"verifier_token_share_{target_cls}"
        sshare_col = f"share_{target_cls}"
        verifier_share_rises = bool(hi[vshare_col] > lo[vshare_col])
        service_share_rises = bool(hi[sshare_col] > lo[sshare_col])
        x_min_rises = bool(hi["min_x"] > lo["min_x"] + 1e-6)
        eta_falls = bool(hi["eta"] < lo["eta"])
        y_falls = bool(hi["Y"] < lo["Y"])
        lever_moves_allocation = verifier_share_rises or service_share_rises

        if not lever_moves_allocation:
            outcome = "D"
            outcome_text = (
                f"Outcome D: the diagnostic_priority_multiplier lever does not change real "
                f"service allocation ({target_cls} verifier-token / service share did not "
                f"rise with {mult_col}). This is an invalid diagnostic run -- do not "
                f"interpret the frontier; fix the scheduler coupling or use explicit group "
                f"quotas."
            )
        elif x_min_rises and (eta_falls or y_falls):
            outcome = "A"
            outcome_text = (
                f"Outcome A: increasing {mult_col} raises {target_cls}'s share of scarce "
                f"verifier capacity AND raises achieved min_x, while eta/Y fall -- a genuine "
                f"goodput-interactivity trade-off exists in the reachable region that the "
                f"AUC/adaptive_index controller is not tracing."
            )
        elif x_min_rises and not (eta_falls or y_falls):
            outcome = "C"
            outcome_text = (
                "Outcome C: min_x improves without a significant Y/eta drop -- the previous "
                "controller was using an inefficient allocation even within the same "
                "capacity region; the AUC objective still has limited curvature but the "
                "controller should be fixed."
            )
        else:
            outcome = "B"
            outcome_text = (
                f"Outcome B: {target_cls}'s service/verifier-token share clearly rises with "
                f"{mult_col}, but achieved min_x does not improve -- the ceiling is not "
                f"caused by insufficient verifier priority; the bottleneck lies elsewhere on "
                f"the per-client critical path (draft throughput, RTT, or sequential round "
                f"structure). Run the isolated {target_cls} capacity test (Section 10) next."
            )
        mechanism_summary = {
            "outcome": outcome,
            "outcome_text": outcome_text,
            "sweep": sweep,
            "target_class": target_cls,
            "swept_multiplier": mult_col,
            "lo_multiplier": float(lo[mult_col]),
            "hi_multiplier": float(hi[mult_col]),
            f"verifier_share_{target_cls}_lo": float(lo[vshare_col]),
            f"verifier_share_{target_cls}_hi": float(hi[vshare_col]),
            f"service_share_{target_cls}_lo": float(lo[sshare_col]),
            f"service_share_{target_cls}_hi": float(hi[sshare_col]),
            "min_x_lo": float(lo["min_x"]),
            "min_x_hi": float(hi["min_x"]),
            "eta_lo": float(lo["eta"]),
            "eta_hi": float(hi["eta"]),
            "Y_lo": float(lo["Y"]),
            "Y_hi": float(hi["Y"]),
            "lever_moves_allocation": lever_moves_allocation,
            "x_min_rises": x_min_rises,
            "eta_falls": eta_falls,
            "Y_falls": y_falls,
        }
        print(f"\n{outcome_text}")

    with open(OUT_DIR / "mechanism_summary.json", "w", encoding="utf-8") as f:
        json.dump(mechanism_summary, f, indent=2)

    print(f"\nWrote points_raw.csv, points_summary.csv, frontier_points.csv, mechanism_summary.json under {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=["a", "am", "b"], default="a")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument(
        "--verify-token-budget",
        type=int,
        default=None,
        help="Override verification_batching.verify_token_budget (base config default: 6). "
        "Used for the widened-budget follow-up check when the base config's batches "
        "turn out to admit ~1 request each (no real within-batch competition for the "
        "priority multiplier to act on).",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Suffix appended to run labels/output files, to keep a follow-up sweep "
        "(e.g. a widened-budget check) from overwriting the main sweep's files.",
    )
    args = parser.parse_args()

    if args.aggregate:
        aggregate(args.sweep)
        return
    run_all(args.sweep, args.shard_index, args.num_shards, args.verify_token_budget, args.tag)


if __name__ == "__main__":
    main()
