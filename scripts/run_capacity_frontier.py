"""Outer N* capacity-search driver (CAPACITY_AUC_NSTAR_IMPLEMENTATION.md
Sections 12-14, 17 Exp1, 20-21).

Modes:
  --pilot        small x grid, short measurement windows (Phase 4 sanity)
  --search       full x grid, long windows
  --resume       continue a search using the on-disk candidate cache
  --analyze-only rebuild frontier / ICA / figures from the cache, no GPU

For each x (increasing, so N*(x_{k-1}) upper-bounds N*(x_k), Section 13.3)
we binary-search the largest integer N with a feasible run, calling
run_simulation.py once per (N, x, seed). Every candidate is cached to
results/capacity_frontier/candidate_cache.jsonl and never re-run unless
--force.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "legacy" / "src"))

from edge_specsim.capacity import (  # noqa: E402
    CandidateCache,
    CandidateMeasurement,
    classify_feasibility,
    ica,
    monotone_scale_search,
    step_area_bounds,
    tail_slope,
)
from edge_specsim.metrics import (  # noqa: E402
    compute_sustained_service_rates,
    summarize_sustained_service_rates,
)

BASE_CONFIG_PATH = REPO_ROOT / "configs" / "capacity_frontier_base.yaml"
OUT_DIR = REPO_ROOT / "results" / "capacity_frontier"
RAW_DIR = OUT_DIR / "raw"
CACHE_PATH = OUT_DIR / "candidate_cache.jsonl"


def _config_hash(cfg: dict) -> str:
    """Hash of the deployment-relevant config (excludes seed / num_clients /
    x / output path, which are the search variables)."""
    slim = copy.deepcopy(cfg)
    for k in ("output_csv", "seed", "num_clients", "min_interactivity_tps"):
        slim.get("simulation", {}).pop(k, None)
    slim.get("controller", {}).pop("min_interactivity_tps", None)
    payload = json.dumps(slim, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_run_config(base_cfg: dict, n: int, x: float, seed: int, meas_s: float, out_csv: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["simulation"]["num_clients"] = int(n)
    cfg["simulation"]["seed"] = int(seed)
    cfg["simulation"]["min_interactivity_tps"] = float(x)
    cfg["simulation"]["output_csv"] = out_csv
    cfg["controller"]["policy"] = "capacity_dpp"
    cfg["controller"]["min_interactivity_tps"] = float(x)
    cfg["controller"]["V"] = float(base_cfg["controller"]["V"])
    cfg["experiment"]["measurement_seconds"] = float(meas_s)
    return cfg


def measure_candidate(rounds_csv: Path, n: int, x: float, seed: int) -> CandidateMeasurement:
    df = pd.read_csv(rounds_csv, low_memory=False)
    if "in_measurement_window" in df.columns:
        win = df[df["in_measurement_window"].astype(bool)]
    else:
        win = df
    if win.empty:
        # nothing landed in the window -> treat as zero service (infeasible)
        return CandidateMeasurement(
            scale=n, n_active=n, x_requirement=x, seed=seed, min_rate_tps=0.0,
            per_client_rates={i: 0.0 for i in range(n)},
        )

    start_ms = float(win["client_receive_ms"].min())
    end_ms = float(win["client_receive_ms"].max())
    t_meas_s = max(1e-6, (end_ms - start_ms) / 1000.0)

    rates_df = compute_sustained_service_rates(
        win, t_measurement_s=t_meas_s, admitted_client_ids=list(range(n))
    )
    summ = summarize_sustained_service_rates(rates_df)
    per_client = dict(zip(rates_df["client_id"].astype(int), rates_df["sustained_service_rate_tps"].astype(float)))

    # queue-tail slopes over the measurement window (Section 12.1)
    def _slope(col: str) -> float:
        if col not in win.columns:
            return 0.0
        s = win.sort_values("client_receive_ms")
        return tail_slope((s["client_receive_ms"] / 1000.0).tolist(), s[col].astype(float).tolist())

    max_z = 0.0
    max_dq = 0.0
    for _, g in win.groupby("client_id"):
        gs = g.sort_values("client_receive_ms")
        ts = (gs["client_receive_ms"] / 1000.0).tolist()
        if "z_queue" in gs.columns:
            max_z = max(max_z, tail_slope(ts, gs["z_queue"].astype(float).tolist()))
        if "device_queue" in gs.columns:
            max_dq = max(max_dq, tail_slope(ts, gs["device_queue"].astype(float).tolist()))

    return CandidateMeasurement(
        scale=n,
        n_active=n,
        x_requirement=x,
        seed=seed,
        min_rate_tps=summ["min_sustained_service_rate_tps"],
        per_client_rates=per_client,
        max_z_slope=max_z,
        max_device_queue_slope=max_dq,
        server_queue_slope=_slope("server_queue"),
        goodput_tps=summ["mean_sustained_service_rate_tps"] * n,
        mean_gamma=float(win["gamma"].mean()) if "gamma" in win.columns else 0.0,
        mean_batch_size=float(win["verification_batch_size"].mean()) if "verification_batch_size" in win.columns else 0.0,
        mean_fill_ratio=(
            float((win["verification_batch_token_cost"] / win["verification_batch_token_budget"]).mean())
            if {"verification_batch_token_cost", "verification_batch_token_budget"}.issubset(win.columns)
            else 0.0
        ),
        extra={"rate_summary": summ},
    )


def run_one(base_cfg: dict, n: int, x: float, seed: int, meas_s: float, cfg_hash: str,
            cache: CandidateCache, force: bool) -> CandidateMeasurement:
    cached = None if force else cache.get(x, n, "capacity_dpp", seed, cfg_hash)
    if cached is not None:
        return cached
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    label = f"n{n}_x{x}_s{seed}"
    out_csv = f"results/capacity_frontier/raw/{label}.csv"
    cfg = build_run_config(base_cfg, n, x, seed, meas_s, out_csv)
    cfg_path = RAW_DIR / f"{label}.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    print(f"=== run n={n} x={x} seed={seed} meas={meas_s}s ===", flush=True)
    r = subprocess.run(
        [sys.executable, "legacy/scripts/run_simulation.py", "--config", str(cfg_path), "--detailed-log"],
        cwd=str(REPO_ROOT), capture_output=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"run_simulation.py failed for {label} (exit {r.returncode})")
    m = measure_candidate(REPO_ROOT / out_csv, n, x, seed)
    cache.put(x, n, "capacity_dpp", seed, cfg_hash, m)
    return m


def feasibility_at(base_cfg: dict, n: int, x: float, meas_s: float, cfg_hash: str,
                   cache: CandidateCache, cs: dict, force: bool) -> tuple[str, list[CandidateMeasurement]]:
    seeds = list(range(1, int(cs["initial_seeds"]) + 1))
    meas = [run_one(base_cfg, n, x, s, meas_s, cfg_hash, cache, force) for s in seeds]
    res = classify_feasibility(
        meas, x,
        rate_tolerance_fraction=float(cs["rate_tolerance_fraction"]),
        queue_slope_tolerance=float(cs["queue_slope_tolerance"]),
    )
    while res.status == "uncertain" and len(seeds) < int(cs["max_seeds"]):
        s = len(seeds) + 1
        seeds.append(s)
        meas.append(run_one(base_cfg, n, x, s, meas_s, cfg_hash, cache, force))
        res = classify_feasibility(
            meas, x,
            rate_tolerance_fraction=float(cs["rate_tolerance_fraction"]),
            queue_slope_tolerance=float(cs["queue_slope_tolerance"]),
        )
    return res.status, meas


def search(mode: str, force: bool = False) -> None:
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    cs = base_cfg["capacity_search"]
    cfg_hash = _config_hash(base_cfg)
    cache = CandidateCache(CACHE_PATH)

    if mode == "pilot":
        x_grid = list(cs["x_grid_pilot"])
        meas_s = float(cs["measurement_seconds_pilot"])
        n_cap = min(int(cs["n_ref"]), 12)
    else:
        x_grid = list(cs["x_grid_final"])
        meas_s = float(cs["measurement_seconds_final"])
        n_cap = int(cs["n_ref"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    decisions = []
    hi_hint = None
    for x in sorted(x_grid):
        def feas(n: int) -> str:
            st, ms = feasibility_at(base_cfg, n, x, meas_s, cfg_hash, cache, cs, force)
            decisions.append({"x": x, "n": n, "status": st,
                              "min_rate_mean": sum(m.min_rate_tps for m in ms) / len(ms)})
            return st

        n_star, trace = monotone_scale_search(
            feas, lo_feasible=int(cs["n_lo_feasible"]), hi_cap=n_cap, hi_hint=hi_hint,
        )
        hi_hint = max(1, n_star)  # monotonicity reuse for the next (larger) x

        # gather the boundary candidate's measurements for reporting
        _, boundary_ms = feasibility_at(base_cfg, max(1, n_star), x, meas_s, cfg_hash, cache, cs, force=False)
        mean = lambda k: sum(getattr(m, k) for m in boundary_ms) / len(boundary_ms)
        rows.append({
            "policy": "capacity_dpp",
            "x_requirement": x,
            "N_hat_star": n_star,
            "boundary_last_feasible_N": n_star,
            "boundary_first_infeasible_N": trace.get("boundary_first_infeasible"),
            "seeds": len(boundary_ms),
            "min_rate_at_last_feasible": mean("min_rate_tps"),
            "queue_stability_at_last_feasible": mean("max_z_slope"),
            "Y_at_last_feasible": mean("goodput_tps"),
            "mean_gamma": mean("mean_gamma"),
            "mean_batch_size": mean("mean_batch_size"),
            "mean_fill_ratio": mean("mean_fill_ratio"),
        })
        print(f"  -> N*({x}) = {n_star}", flush=True)
        pd.DataFrame(rows).to_csv(OUT_DIR / "nstar_frontier.csv", index=False)
        pd.DataFrame(decisions).to_csv(OUT_DIR / "feasibility_decisions.csv", index=False)

    analyze(base_cfg)


def analyze(base_cfg: dict | None = None) -> None:
    if base_cfg is None:
        with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
    cs = base_cfg["capacity_search"]
    fr = pd.read_csv(OUT_DIR / "nstar_frontier.csv").sort_values("x_requirement")
    xs = fr["x_requirement"].astype(float).tolist()
    ns = fr["N_hat_star"].astype(float).tolist()
    bounds = step_area_bounds(xs, ns)
    x_L = float(cs["x_L"])
    x_U = float(cs["x_U"])
    n_ref = float(cs["n_ref"])
    ica_val = ica(xs, ns, x_L=x_L, x_U=x_U, n_ref=n_ref)
    out = {
        "x_L": x_L, "x_U": x_U, "N_ref": n_ref,
        "area_lower": bounds["area_lower"],
        "area_upper": bounds["area_upper"],
        "area_step_estimate": bounds["area_step_estimate"],
        "ICA": ica_val,
        "x_grid": xs, "nstar": ns,
    }
    with open(OUT_DIR / "ica_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n=== N*(x) frontier ===")
    print(fr[["x_requirement", "N_hat_star", "min_rate_at_last_feasible", "Y_at_last_feasible"]].to_string(index=False))
    print(f"\narea_lower={bounds['area_lower']:.3f} area_upper={bounds['area_upper']:.3f} "
          f"area_step={bounds['area_step_estimate']:.3f}  ICA={ica_val:.4f}")

    try:
        _figures(fr)
    except Exception as exc:  # matplotlib optional
        print(f"(figures skipped: {exc})")


def _figures(fr: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(fr["x_requirement"], fr["N_hat_star"], where="post", marker="o")
    ax.set_xlabel("required interactivity x (tok/s/client)")
    ax.set_ylabel("max sustainable concurrency N*(x)")
    ax.set_title("Fig 1: N*(x) capacity frontier (capacity_dpp, pilot)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig1_nstar_frontier.png", dpi=130); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(fr["x_requirement"], fr["mean_gamma"], "o-", color="tab:red", label="mean gamma at boundary")
    ax1.set_xlabel("x"); ax1.set_ylabel("mean gamma", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(fr["x_requirement"], fr["N_hat_star"], "s-", color="tab:blue", label="N*(x)")
    ax2.set_ylabel("N*(x)", color="tab:blue")
    ax1.set_title("Fig 3: gamma mechanism along the frontier")
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig3_gamma_vs_capacity.png", dpi=130); plt.close(fig)
    print(f"wrote fig1, fig3 under {OUT_DIR}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--search", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.analyze_only:
        analyze()
    elif a.pilot:
        search("pilot", force=a.force)
    elif a.search or a.resume:
        search("final", force=a.force and not a.resume)
    else:
        raise SystemExit("pass one of --pilot / --search / --resume / --analyze-only")


if __name__ == "__main__":
    main()
