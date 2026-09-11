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
import glob
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

# Set by search(); which controller policy this sweep evaluates. Every
# baseline is run through the SAME outer capacity search (Section 18).
_ACTIVE_POLICY = "capacity_dpp"


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
    cfg["controller"]["policy"] = _ACTIVE_POLICY
    cfg["controller"]["min_interactivity_tps"] = float(x)
    cfg["controller"]["V"] = float(base_cfg["controller"]["V"])
    cfg["experiment"]["warmup_seconds"] = float(base_cfg["capacity_search"].get("warmup_seconds", 0.0))
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
    cached = None if force else cache.get(x, n, _ACTIVE_POLICY, seed, cfg_hash)
    if cached is not None:
        return cached
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    label = f"{_ACTIVE_POLICY}_n{n}_x{x}_s{seed}"
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
    cache.put(x, n, _ACTIVE_POLICY, seed, cfg_hash, m)
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


def search(mode: str, policy: str = "capacity_dpp", force: bool = False) -> None:
    global _ACTIVE_POLICY
    _ACTIVE_POLICY = policy
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
        n_cap = int(cs.get("n_search_cap", cs["n_ref"]))

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
            "policy": policy,
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
        print(f"  -> [{policy}] N*({x}) = {n_star}", flush=True)
        pd.DataFrame(rows).to_csv(OUT_DIR / f"nstar_frontier_{policy}.csv", index=False)
        pd.DataFrame(decisions).to_csv(OUT_DIR / f"feasibility_decisions_{policy}.csv", index=False)
        if policy == "capacity_dpp":  # keep the un-suffixed names as the primary
            pd.DataFrame(rows).to_csv(OUT_DIR / "nstar_frontier.csv", index=False)
            pd.DataFrame(decisions).to_csv(OUT_DIR / "feasibility_decisions.csv", index=False)

    analyze(base_cfg)


def analyze(base_cfg: dict | None = None) -> None:
    if base_cfg is None:
        with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
    cs = base_cfg["capacity_search"]
    x_L, x_U, n_ref = float(cs["x_L"]), float(cs["x_U"]), float(cs["n_ref"])

    frontier_files = sorted(glob.glob(str(OUT_DIR / "nstar_frontier_*.csv")))
    if not frontier_files and (OUT_DIR / "nstar_frontier.csv").exists():
        frontier_files = [str(OUT_DIR / "nstar_frontier.csv")]

    comparison_rows = []
    per_policy_ica = {}
    for path in frontier_files:
        fr = pd.read_csv(path).sort_values("x_requirement")
        policy = str(fr["policy"].iloc[0]) if "policy" in fr.columns else Path(path).stem.replace("nstar_frontier_", "")
        xs = fr["x_requirement"].astype(float).tolist()
        ns = fr["N_hat_star"].astype(float).tolist()
        bounds = step_area_bounds(xs, ns)
        ica_val = ica(xs, ns, x_L=x_L, x_U=x_U, n_ref=n_ref)
        per_policy_ica[policy] = {"xs": xs, "ns": ns, "ICA": ica_val, **bounds}
        out = {"policy": policy, "x_L": x_L, "x_U": x_U, "N_ref": n_ref,
               "area_lower": bounds["area_lower"], "area_upper": bounds["area_upper"],
               "area_step_estimate": bounds["area_step_estimate"], "ICA": ica_val,
               "x_grid": xs, "nstar": ns}
        with open(OUT_DIR / f"ica_summary_{policy}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        if policy == "capacity_dpp":
            with open(OUT_DIR / "ica_summary.json", "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        comparison_rows.append({"policy": policy, "ICA": ica_val,
                                "area_lower": bounds["area_lower"], "area_upper": bounds["area_upper"],
                                "area_step_estimate": bounds["area_step_estimate"],
                                "x_min": min(xs), "x_max": max(xs),
                                "nstar_at_x_min": ns[0], "nstar_at_x_max": ns[-1]})
        print(f"[{policy}] ICA={ica_val:.4f}  area[{bounds['area_lower']:.1f},{bounds['area_upper']:.1f}]  "
              f"N*: {ns}")

    if comparison_rows:
        cmp_df = pd.DataFrame(comparison_rows).sort_values("ICA", ascending=False)
        cmp_df.to_csv(OUT_DIR / "policy_comparison.csv", index=False)
        print("\n=== policy comparison (by ICA) ===")
        print(cmp_df.to_string(index=False))

    try:
        _figures(per_policy_ica)
    except Exception as exc:  # matplotlib optional
        print(f"(figures skipped: {exc})")


def _figures(per_policy_ica: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig 1: N*(x) frontier(s)
    fig, ax = plt.subplots(figsize=(7, 5))
    for policy, d in sorted(per_policy_ica.items()):
        ax.step(d["xs"], d["ns"], where="post", marker="o", label=f"{policy} (ICA={d['ICA']:.2f})")
    ax.set_xlabel("required interactivity x (tok/s/client)")
    ax.set_ylabel("max sustainable concurrency N*(x)")
    ax.set_title("Fig 1: N*(x) capacity frontier")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_DIR / "fig1_nstar_frontier.png", dpi=130); plt.close(fig)

    # Fig 2: ICA comparison bar
    if len(per_policy_ica) > 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        items = sorted(per_policy_ica.items(), key=lambda kv: kv[1]["ICA"], reverse=True)
        ax.bar([k for k, _ in items], [v["ICA"] for _, v in items], color="tab:blue")
        ax.set_ylabel("ICA (fraction of the common rectangle)")
        ax.set_title("Fig 2: ICA by policy")
        ax.set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(OUT_DIR / "fig2_ica_comparison.png", dpi=130); plt.close(fig)

    # Fig 3: gamma mechanism for capacity_dpp
    cap = None
    for path in sorted(glob.glob(str(OUT_DIR / "nstar_frontier_capacity_dpp.csv"))) or \
            ([str(OUT_DIR / "nstar_frontier.csv")] if (OUT_DIR / "nstar_frontier.csv").exists() else []):
        cap = pd.read_csv(path).sort_values("x_requirement")
    if cap is not None:
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(cap["x_requirement"], cap["mean_gamma"], "o-", color="tab:red", label="mean gamma at boundary")
        ax1.set_xlabel("x"); ax1.set_ylabel("mean gamma", color="tab:red")
        ax2 = ax1.twinx()
        ax2.plot(cap["x_requirement"], cap["N_hat_star"], "s-", color="tab:blue", label="N*(x)")
        ax2.set_ylabel("N*(x)", color="tab:blue")
        ax1.set_title("Fig 3: gamma mechanism along the capacity_dpp frontier")
        fig.tight_layout(); fig.savefig(OUT_DIR / "fig3_gamma_vs_capacity.png", dpi=130); plt.close(fig)
    print(f"wrote figures under {OUT_DIR}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--search", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--policy", default="capacity_dpp",
                   help="single controller policy to sweep")
    p.add_argument("--policies", default=None,
                   help="comma-separated list of policies; each is run through the "
                        "SAME outer capacity search, then policy_comparison.csv is written")
    a = p.parse_args()

    policies = [s.strip() for s in a.policies.split(",")] if a.policies else [a.policy]

    if a.analyze_only:
        analyze()
    elif a.pilot:
        for pol in policies:
            search("pilot", policy=pol, force=a.force)
    elif a.search or a.resume:
        for pol in policies:
            search("final", policy=pol, force=a.force and not a.resume)
    else:
        raise SystemExit("pass one of --pilot / --search / --resume / --analyze-only")


if __name__ == "__main__":
    main()
