"""Outer capacity-search driver.

CAPACITY_AWARE_FRAMEWORK_CODEX_IMPLEMENTATION.md supersedes
CAPACITY_AUC_NSTAR_IMPLEMENTATION.md wherever they disagree (Section 0).
This file now has two paths:

  --pilot           HEURISTIC/SANITY-ONLY. Old classify_feasibility
                     (worst-seed/median) + monotone_scale_search, writing
                     into the OLD mutable results/capacity_frontier/ path.
                     Never produces headline ICA (Section 9.2).

  --search/--resume FINAL mode. classify_feasibility_final (Bonferroni-
                     bootstrap per-client confidence bounds against the
                     EXACT x, Section 9.1), NStarResult (Section 10.1),
                     optional x*(N) cross-check (Section 10.2), dual area
                     bounds (Section 11), all written into an immutable
                     results/capacity_frontier/<run_tag>/ directory
                     (Section 12).

  --analyze-only     rebuild frontier/ICA/figures/RUN_REPORT.md from an
                     existing run directory's candidate cache, no GPU.

Modes:
  --search-axis n    N*(x) (default)
  --search-axis x    x*(N), same candidate runner and classifier
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "legacy" / "src"))

from edge_specsim.capacity import (  # noqa: E402
    CandidateCache,
    CandidateMeasurement,
    NStarResult,
    SearchDecision,
    XStarResult,
    area_from_nstar,
    area_from_xstar,
    classify_feasibility,
    classify_feasibility_final,
    ica,
    monotone_scale_search,
    monotone_scale_search_final,
    step_area_bounds,
    tail_slope,
    xstar_search,
)
from edge_specsim.metrics import (  # noqa: E402
    compute_sustained_service_rates,
    summarize_sustained_service_rates,
)

RESULTS_ROOT = REPO_ROOT / "results" / "capacity_frontier"
# Legacy/pilot mutable path (Section 9.2's heuristic/sanity-only mode).
PILOT_OUT_DIR = RESULTS_ROOT
PILOT_RAW_DIR = PILOT_OUT_DIR / "raw"
PILOT_CACHE_PATH = PILOT_OUT_DIR / "candidate_cache.jsonl"

CANDIDATE_CACHE_SCHEMA_VERSION = 2

_ACTIVE_POLICY = "capacity_dpp"


# =========================================================================
# provenance
# =========================================================================

def _config_hash(cfg: dict) -> str:
    slim = copy.deepcopy(cfg)
    for k in ("output_csv", "seed", "num_clients", "min_interactivity_tps"):
        slim.get("simulation", {}).pop(k, None)
    slim.get("controller", {}).pop("min_interactivity_tps", None)
    payload = json.dumps(slim, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                            capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _environment_info() -> dict:
    info = {
        "python_version": sys.version,
        "hostname": platform.node(),
        "platform": platform.platform(),
    }
    try:
        import torch  # type: ignore
        info["pytorch_version"] = getattr(torch, "__version__", "unknown")
        hip = getattr(torch.version, "hip", None)
        cuda = getattr(torch.version, "cuda", None)
        info["rocm_or_cuda_version"] = hip or cuda or "unknown"
        try:
            info["gpu_models"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception:
            pass
    except Exception:
        pass
    try:
        import vllm  # type: ignore
        info["vllm_version"] = getattr(vllm, "__version__", "unknown")
    except Exception:
        pass
    return info


def _population_config_fingerprint(base_cfg: dict) -> str:
    """Config-level fingerprint of the nested-population identity (seed,
    n_max, questions_per_client). Per-candidate exact client-list
    fingerprints are additionally logged by the simulator on every round
    record (population_fingerprint column)."""
    sim = base_cfg.get("simulation", {})
    payload = json.dumps(
        {
            "seed": sim.get("seed"),
            "nested_population": sim.get("nested_population"),
            "population_n_max": sim.get("population_n_max"),
            "questions_per_client": sim.get("questions_per_client"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class RunContext:
    """Section 12: immutable, provenance-complete run directory."""

    def __init__(self, run_dir: Path, base_cfg: dict, config_hash: str, git_sha: str):
        self.run_dir = run_dir
        self.raw_dir = run_dir / "raw"
        self.figures_dir = run_dir / "figures"
        self.cache_path = run_dir / "candidate_cache.jsonl"
        self.config_hash = config_hash
        self.git_sha = git_sha
        self.base_cfg = base_cfg

    @classmethod
    def create(cls, base_cfg: dict, run_tag: str | None = None) -> "RunContext":
        config_hash = _config_hash(base_cfg)
        git_sha = _git_sha()
        if run_tag is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            run_tag = f"{ts}_{config_hash}_{git_sha[:8]}"
        run_dir = RESULTS_ROOT / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "raw").mkdir(exist_ok=True)
        (run_dir / "figures").mkdir(exist_ok=True)
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(base_cfg, f)
        (run_dir / "config_hash.txt").write_text(config_hash, encoding="utf-8")
        (run_dir / "git_commit.txt").write_text(git_sha, encoding="utf-8")
        with open(run_dir / "environment.json", "w", encoding="utf-8") as f:
            json.dump(_environment_info(), f, indent=2)
        (run_dir / "population_fingerprint.txt").write_text(
            _population_config_fingerprint(base_cfg), encoding="utf-8"
        )
        print(f"[run_tag] {run_tag}", flush=True)
        return cls(run_dir, base_cfg, config_hash, git_sha)

    @classmethod
    def resume(cls, run_tag: str) -> "RunContext":
        run_dir = RESULTS_ROOT / run_tag
        if not run_dir.exists():
            raise FileNotFoundError(
                f"no run directory {run_dir} -- --resume requires an explicit "
                "--run-tag pointing at an existing immutable run (Section 12: "
                "never discover a cache by silently using a global mutable path)"
            )
        base_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        config_hash = (run_dir / "config_hash.txt").read_text(encoding="utf-8").strip()
        git_sha = (run_dir / "git_commit.txt").read_text(encoding="utf-8").strip()
        return cls(run_dir, base_cfg, config_hash, git_sha)


# =========================================================================
# candidate execution
# =========================================================================

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


def measure_candidate(rounds_csv: Path, n: int, x: float, seed: int, meas_s: float) -> CandidateMeasurement:
    """Section 1.1 / 8: `meas_s` -- the EXACT configured measurement
    duration this candidate was run with -- is the denominator, never an
    event-timestamp span. `in_measurement_window` (set by the simulator
    from measurement_start_ms/measurement_end_ms) already excludes warmup
    rows."""
    df = pd.read_csv(rounds_csv, low_memory=False)
    win = df[df["in_measurement_window"].astype(bool)] if "in_measurement_window" in df.columns else df
    if win.empty:
        return CandidateMeasurement(
            scale=n, n_active=n, x_requirement=x, seed=seed, min_rate_tps=0.0,
            per_client_rates={i: 0.0 for i in range(n)},
        )

    rates_df = compute_sustained_service_rates(win, t_measurement_s=meas_s, admitted_client_ids=list(range(n)))
    summ = summarize_sustained_service_rates(rates_df)
    per_client = dict(zip(rates_df["client_id"].astype(int), rates_df["sustained_service_rate_tps"].astype(float)))

    # Workload-exhaustion guard: if a client's own activity span inside the
    # window is much shorter than meas_s, it ran out of assigned prompts
    # (questions_per_client too small for this rate * meas_s) before the
    # measurement interval ended. The fixed-denominator rate is then an
    # UNDERESTIMATE of true sustained capacity, not a real throttling
    # signal -- surfaced loudly here instead of silently producing a
    # spuriously low (and possibly falsely "infeasible") rate.
    exhausted_clients = []
    if "client_receive_ms" in win.columns:
        for cid, g in win.groupby("client_id"):
            span_s = (g["client_receive_ms"].max() - g["client_receive_ms"].min()) / 1000.0
            if span_s < 0.85 * meas_s:
                exhausted_clients.append({"client_id": int(cid), "active_span_s": round(span_s, 1)})
    if exhausted_clients:
        print(f"  !! WORKLOAD EXHAUSTION at n={n} x={x} seed={seed}: "
              f"{len(exhausted_clients)}/{n} client(s) ran out of prompts before the "
              f"{meas_s}s window ended (increase questions_per_client): {exhausted_clients}",
              flush=True)

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
        scale=n, n_active=n, x_requirement=x, seed=seed,
        min_rate_tps=summ["min_sustained_service_rate_tps"],
        per_client_rates=per_client,
        max_z_slope=max_z, max_device_queue_slope=max_dq, server_queue_slope=_slope("server_queue"),
        goodput_tps=summ["mean_sustained_service_rate_tps"] * n,
        mean_gamma=float(win["gamma"].mean()) if "gamma" in win.columns else 0.0,
        mean_batch_size=float(win["verification_batch_size"].mean()) if "verification_batch_size" in win.columns else 0.0,
        mean_fill_ratio=(
            float((win["verification_batch_token_cost"] / win["verification_batch_token_budget"]).mean())
            if {"verification_batch_token_cost", "verification_batch_token_budget"}.issubset(win.columns) else 0.0
        ),
        extra={
            "rate_summary": summ,
            "measurement_duration_s": meas_s,
            "forced_service_fraction": (
                float(win.drop_duplicates("verification_batch_id")["forced_service"].mean())
                if "forced_service" in win.columns else None
            ),
            "mean_uplink_rate_mbps": float(win["realized_uplink_rate_mbps"].mean()) if "realized_uplink_rate_mbps" in win.columns else None,
            "mean_gamma_rate_signal_mbps": float(win["gamma_rate_signal_mbps"].mean()) if "gamma_rate_signal_mbps" in win.columns else None,
            "exhausted_clients": exhausted_clients,
        },
    )


def run_one(base_cfg: dict, n: int, x: float, seed: int, meas_s: float, cfg_hash: str,
            cache: CandidateCache, raw_dir: Path, force: bool) -> CandidateMeasurement:
    cached = None if force else cache.get(x, n, _ACTIVE_POLICY, seed, cfg_hash)
    if cached is not None:
        return cached
    raw_dir.mkdir(parents=True, exist_ok=True)
    label = f"{_ACTIVE_POLICY}_n{n}_x{x}_s{seed}"
    out_csv_abs = raw_dir / f"{label}.csv"
    out_csv_rel = str(out_csv_abs.relative_to(REPO_ROOT))
    cfg = build_run_config(base_cfg, n, x, seed, meas_s, out_csv_rel)
    cfg_path = raw_dir / f"{label}.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    print(f"=== run n={n} x={x} seed={seed} meas={meas_s}s policy={_ACTIVE_POLICY} ===", flush=True)
    r = subprocess.run(
        [sys.executable, "legacy/scripts/run_simulation.py", "--config", str(cfg_path), "--detailed-log"],
        cwd=str(REPO_ROOT), capture_output=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"run_simulation.py failed for {label} (exit {r.returncode})")
    m = measure_candidate(out_csv_abs, n, x, seed, meas_s)
    cache.put(x, n, _ACTIVE_POLICY, seed, cfg_hash, m)
    return m


# =========================================================================
# FINAL mode: exact-SLO classifier + NStarResult + x*(N)
# =========================================================================

def _decision_from_measurements(meas: list[CandidateMeasurement], x: float, n: int, cs: dict) -> SearchDecision:
    r = classify_feasibility_final(
        meas, x, n_clients=n, confidence_delta=float(cs.get("confidence_delta", 0.05)),
    )
    status = "uncertain" if r.status == "heuristic_only" else r.status
    return SearchDecision(n=n, status=status, min_lcb=r.min_lcb, min_ucb=r.min_ucb, seeds_used=r.seeds_used)


def feasibility_at_final(base_cfg: dict, n: int, x: float, meas_s: float, cfg_hash: str,
                          cache: CandidateCache, raw_dir: Path, cs: dict, force: bool,
                          near_boundary: bool = False) -> SearchDecision:
    """Section 9.2: start with `initial_seeds`; escalate on uncertain up to
    `max_seeds`. Near a boundary being locally certified, force at least
    `min_final_seeds` regardless of the preliminary verdict."""
    initial = int(cs.get("initial_seeds", 3))
    max_seeds = int(cs.get("max_seeds", 7))
    min_final = int(cs.get("min_final_seeds", 5)) if near_boundary else initial

    seeds = list(range(1, max(initial, min_final if near_boundary else 0, 1) + 1))
    meas = [run_one(base_cfg, n, x, s, meas_s, cfg_hash, cache, raw_dir, force) for s in seeds]
    decision = _decision_from_measurements(meas, x, n, cs)
    while decision.status == "uncertain" and len(seeds) < max_seeds:
        s = len(seeds) + 1
        seeds.append(s)
        meas.append(run_one(base_cfg, n, x, s, meas_s, cfg_hash, cache, raw_dir, force))
        decision = _decision_from_measurements(meas, x, n, cs)
    return decision


def search_nstar_final(ctx: RunContext, policy: str, x: float, hi_hint: int | None, force: bool) -> NStarResult:
    global _ACTIVE_POLICY
    _ACTIVE_POLICY = policy
    cs = ctx.base_cfg["capacity_search"]
    meas_s = float(cs["measurement_seconds_final"])
    cache = CandidateCache(ctx.cache_path)

    def feas(n: int) -> SearchDecision:
        return feasibility_at_final(ctx.base_cfg, n, x, meas_s, ctx.config_hash, cache, ctx.raw_dir, cs, force)

    result = monotone_scale_search_final(
        feas, lo_feasible=int(cs["n_lo_feasible"]), hi_cap=int(cs["n_search_cap"]),
        hi_hint=hi_hint, x_requirement=x,
    )

    # Section 10.1 step 5: locally certify the boundary with {N*-1, N*, N*+1}
    # using min_final_seeds, if the boundary is exact and not cap-limited.
    if result.exact and not result.hit_search_cap:
        n_star = result.last_confirmed_feasible_n
        for n_check in (n_star - 1, n_star, n_star + 1):
            if n_check < 1 or n_check > int(cs["n_search_cap"]):
                continue
            d = feasibility_at_final(ctx.base_cfg, n_check, x, meas_s, ctx.config_hash, cache, ctx.raw_dir,
                                      cs, force, near_boundary=True)
            result.trace.append(d)
            if d.status == "uncertain" and n_check not in result.uncertain_n:
                result.uncertain_n.append(n_check)
    return result


def search_xstar_final(ctx: RunContext, policy: str, n: int, x_lo: float, x_hi: float, force: bool) -> XStarResult:
    global _ACTIVE_POLICY
    _ACTIVE_POLICY = policy
    cs = ctx.base_cfg["capacity_search"]
    meas_s = float(cs["measurement_seconds_final"])
    cache = CandidateCache(ctx.cache_path)

    def feas(x: float) -> SearchDecision:
        return feasibility_at_final(ctx.base_cfg, n, x, meas_s, ctx.config_hash, cache, ctx.raw_dir, cs, force)

    return xstar_search(feas, n_active=n, x_lo=x_lo, x_hi=x_hi,
                         x_tolerance_tps=float(cs.get("x_tolerance_tps", 0.10)))


def _nstar_row(policy: str, r: NStarResult, seeds_at_boundary: int) -> dict:
    return {
        "policy": policy, "x_requirement": r.x_requirement,
        "last_confirmed_feasible_N": r.last_confirmed_feasible_n,
        "first_confirmed_infeasible_N": r.first_confirmed_infeasible_n,
        "N_lower": r.lower_bound_n, "N_upper": r.upper_bound_n,
        "exact_boundary": r.exact, "hit_search_cap": r.hit_search_cap,
        "boundary_certified": r.exact and not r.hit_search_cap,
        "uncertain_n": ";".join(str(n) for n in r.uncertain_n),
        "seeds_at_boundary": seeds_at_boundary,
    }


def run_search_final(run_tag: str | None, policies: list[str], axis: str, resume: bool, force: bool,
                      config_path: str | None = None) -> None:
    if resume:
        if run_tag is None:
            raise SystemExit("--resume requires --run-tag <existing run directory>")
        ctx = RunContext.resume(run_tag)
    else:
        cfg_path = Path(config_path) if config_path else REPO_ROOT / "configs" / "capacity_frontier_final.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        ctx = RunContext.create(base_cfg, run_tag=run_tag)

    cs = ctx.base_cfg["capacity_search"]
    x_grid = sorted(cs["x_grid_final"])

    for policy in policies:
        if axis == "n":
            rows = []
            hi_hint = None
            for x in x_grid:
                res = search_nstar_final(ctx, policy, x, hi_hint, force)
                seeds_at_boundary = max((d.seeds_used for d in res.trace if d.n == res.last_confirmed_feasible_n), default=0)
                rows.append(_nstar_row(policy, res, seeds_at_boundary))
                hi_hint = max(1, res.lower_bound_n)
                print(f"  -> [{policy}] N*({x}): last_feasible={res.last_confirmed_feasible_n} "
                      f"exact={res.exact} cap={res.hit_search_cap}", flush=True)
                pd.DataFrame(rows).to_csv(ctx.run_dir / f"nstar_frontier_{policy}.csv", index=False)
        else:  # axis == "x"
            n_ref = int(cs["n_ref"])
            rows = []
            for n in range(1, n_ref + 1):
                res = search_xstar_final(ctx, policy, n, x_lo=float(cs["x_L"]), x_hi=float(cs["x_U"]), force=force)
                rows.append({
                    "policy": policy, "N": n,
                    "x_lower_feasible": res.lower_feasible_x,
                    "x_upper_infeasible": res.upper_infeasible_x,
                    "x_estimate": res.estimate_x,
                    "x_tolerance_tps": float(cs.get("x_tolerance_tps", 0.10)),
                    "boundary_certified": res.exact_within_tolerance,
                })
                print(f"  -> [{policy}] x*({n}) = {res.estimate_x:.3f} (certified={res.exact_within_tolerance})",
                      flush=True)
                pd.DataFrame(rows).to_csv(ctx.run_dir / f"xstar_frontier_{policy}.csv", index=False)

    analyze_final(ctx.run_dir.name)


def analyze_final(run_tag: str) -> None:
    ctx = RunContext.resume(run_tag)
    cs = ctx.base_cfg["capacity_search"]
    x_L, x_U, n_ref = float(cs["x_L"]), float(cs["x_U"]), int(cs["n_ref"])

    comparison_rows = []
    report_lines = [
        f"# Capacity-Aware Framework -- Run Report\n",
        f"run_tag: `{run_tag}`  \nconfig_hash: `{ctx.config_hash}`  \ngit_sha: `{ctx.git_sha}`\n",
    ]

    for nstar_path in sorted(glob.glob(str(ctx.run_dir / "nstar_frontier_*.csv"))):
        df = pd.read_csv(nstar_path)
        policy = str(df["policy"].iloc[0])
        results = [
            NStarResult(
                x_requirement=float(row.x_requirement),
                last_confirmed_feasible_n=int(row.last_confirmed_feasible_N),
                first_confirmed_infeasible_n=(None if pd.isna(row.first_confirmed_infeasible_N) else int(row.first_confirmed_infeasible_N)),
                lower_bound_n=int(row.N_lower),
                upper_bound_n=(None if pd.isna(row.N_upper) else int(row.N_upper)),
                exact=bool(row.exact_boundary),
                hit_search_cap=bool(row.hit_search_cap),
            )
            for row in df.itertuples()
        ]
        try:
            area = area_from_nstar(results, x_L=x_L, x_U=x_U, n_ref=n_ref)
        except ValueError as exc:
            print(f"[{policy}] area not computed: {exc}")
            area = {"area_lower": None, "area_upper": None, "ica_lower": None, "ica_upper": None,
                    "any_cap_limited": None, "any_interval_valued": None, "error": str(exc)}
        out = {"policy": policy, "run_tag": run_tag, "x_L": x_L, "x_U": x_U, "n_ref": n_ref, **area}
        with open(ctx.run_dir / f"capacity_area_{policy}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        comparison_rows.append({
            "policy": policy,
            "ica_lower": area["ica_lower"], "ica_upper": area["ica_upper"],
            "any_cap_limited": area["any_cap_limited"], "any_interval_valued": area["any_interval_valued"],
            "x_min": df["x_requirement"].min(), "x_max": df["x_requirement"].max(),
        })
        n_exact = int(df["exact_boundary"].sum())
        n_cap = int(df["hit_search_cap"].sum())
        report_lines.append(
            f"\n## {policy}\n\nICA (lower, upper): {area['ica_lower']}, {area['ica_upper']}  \n"
            f"points: {len(df)} (exact: {n_exact}, cap-limited: {n_cap})\n\n```\n"
            + df.to_string(index=False) + "\n```"
        )
        print(f"[{policy}] ICA in [{area['ica_lower']}, {area['ica_upper']}] "
              f"cap_limited={area['any_cap_limited']} interval_valued={area['any_interval_valued']}")

    if comparison_rows:
        cmp_df = pd.DataFrame(comparison_rows).sort_values("ica_lower", ascending=False)
        cmp_df.to_csv(ctx.run_dir / "policy_comparison.csv", index=False)
        print("\n=== policy comparison (final mode, by ICA lower bound) ===")
        print(cmp_df.to_string(index=False))

    report_lines.append(
        "\n---\n\nThis run used the FINAL exact-SLO classifier "
        "(classify_feasibility_final, Bonferroni-bootstrap confidence bounds) "
        "and NStarResult search semantics. Cap-limited points are reported as "
        "lower bounds, never as N*=N_cap. See CAPACITY_AWARE_FRAMEWORK_CODEX_"
        "IMPLEMENTATION.md for the full contract.\n"
    )
    (ctx.run_dir / "RUN_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nwrote {ctx.run_dir / 'RUN_REPORT.md'}")

    try:
        _figures_final(ctx)
    except Exception as exc:
        print(f"(figures skipped: {exc})")


def _figures_final(ctx: RunContext) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for path in sorted(glob.glob(str(ctx.run_dir / "nstar_frontier_*.csv"))):
        df = pd.read_csv(path).sort_values("x_requirement")
        policy = str(df["policy"].iloc[0])
        xs = df["x_requirement"].tolist()
        ns = df["last_confirmed_feasible_N"].tolist()
        cap = df["hit_search_cap"].tolist()
        ax.step(xs, ns, where="post", marker="o", label=policy)
        for x, n, c in zip(xs, ns, cap):
            if c:
                ax.annotate(r"$\geq$", (x, n), textcoords="offset points", xytext=(0, 6), fontsize=9)
    ax.set_xlabel("required interactivity x (tok/s/client)")
    ax.set_ylabel("N* (last confirmed feasible; arrow = cap-limited lower bound)")
    ax.set_title("N*(x) capacity frontier (final mode)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ctx.figures_dir / "fig1_nstar_frontier.png", dpi=130)
    plt.close(fig)

    cmp_path = ctx.run_dir / "policy_comparison.csv"
    if cmp_path.exists():
        cmp_df = pd.read_csv(cmp_path).sort_values("ica_lower", ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        x_pos = range(len(cmp_df))
        lo = cmp_df["ica_lower"].tolist()
        hi = cmp_df["ica_upper"].tolist()
        ax.bar(x_pos, lo, color="tab:blue", label="ICA lower")
        ax.errorbar(x_pos, lo, yerr=[[0] * len(lo), [h - l for h, l in zip(hi, lo)]],
                     fmt="none", ecolor="black", capsize=4)
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(cmp_df["policy"].tolist(), rotation=20, ha="right")
        ax.set_ylabel("ICA (lower bound, error bar to upper bound)")
        ax.set_title("ICA comparison (final mode)")
        fig.tight_layout()
        fig.savefig(ctx.figures_dir / "fig2_ica_comparison.png", dpi=130)
        plt.close(fig)
    print(f"wrote figures under {ctx.figures_dir}")


# =========================================================================
# PILOT / legacy heuristic path (Section 9.2: sanity-only, no headline ICA)
# =========================================================================

def search_pilot(policy: str, force: bool = False) -> None:
    global _ACTIVE_POLICY
    _ACTIVE_POLICY = policy
    with open(REPO_ROOT / "configs" / "capacity_frontier_base.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    cs = base_cfg["capacity_search"]
    cfg_hash = _config_hash(base_cfg)
    cache = CandidateCache(PILOT_CACHE_PATH)
    x_grid = list(cs["x_grid_pilot"])
    meas_s = float(cs["measurement_seconds_pilot"])
    n_cap = min(int(cs["n_ref"]), 12)

    PILOT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, decisions = [], []
    hi_hint = None
    for x in sorted(x_grid):
        def feas(n: int) -> str:
            seeds = list(range(1, int(cs["initial_seeds"]) + 1))
            meas = [run_one(base_cfg, n, x, s, meas_s, cfg_hash, cache, PILOT_RAW_DIR, force) for s in seeds]
            res = classify_feasibility(meas, x, rate_tolerance_fraction=float(cs["rate_tolerance_fraction"]),
                                        queue_slope_tolerance=float(cs["queue_slope_tolerance"]))
            while res.status == "uncertain" and len(seeds) < int(cs["max_seeds"]):
                s = len(seeds) + 1
                seeds.append(s)
                meas.append(run_one(base_cfg, n, x, s, meas_s, cfg_hash, cache, PILOT_RAW_DIR, force))
                res = classify_feasibility(meas, x, rate_tolerance_fraction=float(cs["rate_tolerance_fraction"]),
                                            queue_slope_tolerance=float(cs["queue_slope_tolerance"]))
            decisions.append({"x": x, "n": n, "status": res.status,
                              "min_rate_mean": sum(m.min_rate_tps for m in meas) / len(meas)})
            return res.status

        n_star, trace = monotone_scale_search(feas, lo_feasible=int(cs["n_lo_feasible"]), hi_cap=n_cap, hi_hint=hi_hint)
        hi_hint = max(1, n_star)
        rows.append({"policy": policy, "x_requirement": x, "N_hat_star": n_star,
                     "hit_search_cap": n_star >= n_cap, "PILOT_SANITY_ONLY": True})
        print(f"  -> [PILOT/{policy}] N*({x}) = {n_star} (heuristic, not headline ICA)", flush=True)
        pd.DataFrame(rows).to_csv(PILOT_OUT_DIR / f"nstar_frontier_{policy}_PILOT.csv", index=False)
        pd.DataFrame(decisions).to_csv(PILOT_OUT_DIR / f"feasibility_decisions_{policy}_PILOT.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", action="store_true", help="heuristic/sanity-only, no headline ICA")
    p.add_argument("--search", action="store_true", help="final mode: new immutable run directory")
    p.add_argument("--resume", action="store_true", help="final mode: continue an existing --run-tag")
    p.add_argument("--analyze-only", action="store_true", help="rebuild reports/figures for --run-tag, no GPU")
    p.add_argument("--run-tag", default=None)
    p.add_argument("--search-axis", choices=["n", "x"], default="n")
    p.add_argument("--force", action="store_true")
    p.add_argument("--policy", default="capacity_dpp")
    p.add_argument("--policies", default=None)
    p.add_argument("--config", default=None,
                   help="override configs/capacity_frontier_final.yaml (e.g. a reduced-grid Gate B/D config)")
    a = p.parse_args()

    policies = [s.strip() for s in a.policies.split(",")] if a.policies else [a.policy]

    if a.analyze_only:
        if not a.run_tag:
            raise SystemExit("--analyze-only requires --run-tag")
        analyze_final(a.run_tag)
    elif a.pilot:
        for pol in policies:
            search_pilot(pol, force=a.force)
    elif a.search or a.resume:
        run_search_final(a.run_tag, policies, a.search_axis, resume=a.resume,
                          force=a.force and not a.resume, config_path=a.config)
    else:
        raise SystemExit("pass one of --pilot / --search / --resume / --analyze-only")


if __name__ == "__main__":
    main()
