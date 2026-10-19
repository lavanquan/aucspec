"""Analysis of the RQ1 minimal pilot (descriptive; CPU only).  Pure functions + a small CLI.

Per run (N, run_seed) from a rounds CSV:  per-client useful-token rates over the EXACT window (absent client = 0),
x_min, mean, aggregate goodput Y, N*x_min, Jain index, verifier utilisation, batch-size and batch-wall statistics, queue wait,
draft/network components, workload lifecycle (INVALID if any client exhausted), virtual/real time ratio, ledger violations.
Across runs: the pre-registered RQ1 rules (see results/AUCSPEC_RQ1_PILOT_REVISED.md section 6).  No statistical
certification is claimed anywhere in this file.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "src"))

from edge_specsim.workload_lifecycle import assess_from_dataframe  # noqa: E402

# pre-registered thresholds (changing them after seeing data invalidates the pilot as a test of RQ1)
PLATEAU_TOL = 0.10          # Y(N_b) <= (1+tol) * Y(N_a)
FLOOR_DROP = 0.65           # x_min(N_b) <= 0.65 * x_min(N_a)   (ideal 1/N gives 0.5 for 8 -> 16)
NO_FALL = 0.90              # x_min(N_b) >= 0.90 * x_min(N_a)   => no concealed fall
UNSATURATED = 1.50          # Y(N_b) >= 1.5 * Y(N_a)            => range too small, inconclusive
MIN_FAIRNESS = 0.80         # x_min / mean >= 0.8 in a homogeneous population
N_A, N_B = 8, 16


def jain(xs):
    s, q = sum(xs), sum(v * v for v in xs)
    return (s * s) / (len(xs) * q) if q > 0 else float("nan")


def run_metrics(df, n: int, *, expected_samples: int | None = None) -> dict:
    start = float(df["experiment_warmup_end_ms"].iloc[0])
    end = float(df["experiment_measurement_end_ms"].iloc[0])
    t_m = (end - start) / 1000.0
    win = df[(df["client_receive_ms"] >= start) & (df["client_receive_ms"] <= end)]
    tokens = win.groupby("client_id")["committed_token_count"].sum().to_dict()
    rates = {c: float(tokens.get(c, 0)) / t_m for c in range(n)}
    vals = list(rates.values())
    work = assess_from_dataframe(df, admitted_client_ids=range(n), measurement_start_ms=start, measurement_end_ms=end,
                                 expected_samples_per_client=expected_samples)
    batches = win.drop_duplicates("verification_batch_id")
    util = float(batches["verification_batch_wall_ms"].sum()) / 1000.0 / t_m if len(batches) else 0.0
    out = {
        "N": n, "window_s": t_m, "rates": rates, "x_min": min(vals), "x_mean": statistics.fmean(vals),
        "Y": sum(vals), "N_x_min": n * min(vals), "jain": jain(vals),
        "x_min_over_mean": (min(vals) / statistics.fmean(vals)) if statistics.fmean(vals) > 0 else float("nan"),
        "verifier_utilisation": util,
        "batch_size_mean": float(win["verification_batch_size"].mean()) if len(win) else float("nan"),
        "batch_size_max": int(win["verification_batch_size"].max()) if len(win) else 0,
        "batch_wall_ms_mean": float(batches["verification_batch_wall_ms"].mean()) if len(batches) else float("nan"),
        "queue_wait_ms_mean": float((win["batch_start_ms"] - win["server_arrival_ms"]).mean()) if len(win) else float("nan"),
        "draft_ms_mean": float(win["draft_latency_ms"].mean()) if len(win) else float("nan"),
        "round_ms_mean": float(win["virtual_round_latency_ms"].mean()) if len(win) else float("nan"),
        "rounds_per_client_min": int(win.groupby("client_id").size().reindex(range(n), fill_value=0).min()),
        "invalid": bool(work.invalid or not work.verifiable),
        "invalid_reason": (f"exhausted clients {work.exhausted_clients}" if work.invalid else
                           (f"unverifiable clients {work.unknown_clients}" if not work.verifiable else "")),
        "real_over_virtual": (float(df["wall_clock_time_ms"].max()) / float(df["client_receive_ms"].max())
                              if "wall_clock_time_ms" in df.columns and df["client_receive_ms"].max() > 0 else float("nan")),
    }
    return out


def _agg(runs: list[dict], key: str) -> dict:
    v = [r[key] for r in runs]
    return {"mean": statistics.fmean(v), "min": min(v), "max": max(v)}


def rq1_verdict(runs_by_n: dict[int, list[dict]]) -> dict:
    """Apply the pre-registered rules to runs grouped by N (INVALID runs are excluded and counted)."""
    clean = {n: [r for r in rs if not r["invalid"]] for n, rs in runs_by_n.items()}
    invalid = {n: sum(r["invalid"] for r in rs) for n, rs in runs_by_n.items() if any(r["invalid"] for r in rs)}
    out: dict = {"invalid_runs_by_N": invalid, "per_N": {}}
    for n, rs in sorted(clean.items()):
        if rs:
            out["per_N"][n] = {"seeds": len(rs), "Y": _agg(rs, "Y"), "x_min": _agg(rs, "x_min"),
                               "N_x_min": _agg(rs, "N_x_min"), "x_min_over_mean": _agg(rs, "x_min_over_mean"),
                               "utilisation": _agg(rs, "verifier_utilisation"), "batch_size": _agg(rs, "batch_size_mean")}
    if N_A not in out["per_N"] or N_B not in out["per_N"]:
        out["verdict"] = "INCONCLUSIVE: runs at N=8 and N=16 are required (missing or INVALID)"
        return out
    a, b = out["per_N"][N_A], out["per_N"][N_B]
    y_ratio = b["Y"]["mean"] / a["Y"]["mean"]
    x_ratio = b["x_min"]["mean"] / a["x_min"]["mean"]
    # seed-consistency: the per-seed ratios must satisfy the rule, not only the means
    per_seed = []
    for ra, rb in zip(clean[N_A], clean[N_B]):
        per_seed.append((rb["Y"] / ra["Y"], rb["x_min"] / ra["x_min"]))
    plateau = y_ratio <= 1 + PLATEAU_TOL and all(y <= 1 + PLATEAU_TOL + 0.05 for y, _ in per_seed)
    floor_falls = x_ratio <= FLOOR_DROP and all(x <= FLOOR_DROP + 0.05 for _, x in per_seed)
    fair = all(out["per_N"][n]["x_min_over_mean"]["mean"] >= MIN_FAIRNESS for n in (N_A, N_B))
    out["N8_to_N16"] = {"Y_ratio": y_ratio, "x_min_ratio": x_ratio, "per_seed_(Y,x_min)": per_seed,
                        "plateau": plateau, "floor_falls": floor_falls, "homogeneous_fair": fair}
    if y_ratio >= UNSATURATED:
        out["verdict"] = "INCONCLUSIVE: goodput still rising between N=8 and N=16 (capacity not reached; extend N)"
    elif y_ratio < 1.0 - PLATEAU_TOL:
        out["verdict"] = "PATHOLOGY: goodput falls with N (diagnose contention/overhead before any capacity claim)"
    elif plateau and floor_falls and fair:
        out["verdict"] = "SUPPORTS RQ1: aggregate goodput plateaus while the per-client floor falls (concealed capacity loss)"
    elif plateau and x_ratio >= NO_FALL:
        out["verdict"] = "FALSIFIES the concealment claim in this range: goodput plateaus but the per-client floor does not fall"
    else:
        out["verdict"] = "MIXED: partial support; report the numbers, do not summarise as support"
    return out


def main():
    import pandas as pd

    spec = json.loads(Path(sys.argv[1]).read_text())      # [{"csv": ..., "n": ..., "seed": ..., "ledger": ...}, ...]
    runs_by_n: dict[int, list[dict]] = {}
    for item in spec:
        df = pd.read_csv(item["csv"], low_memory=False)
        m = run_metrics(df, int(item["n"]))
        m["seed"] = item["seed"]
        if item.get("ledger"):
            chk = json.loads(Path(item["ledger"]).read_text())
            m["ledger_violations"] = chk["n_violations"]
            if chk["n_violations"]:
                m["invalid"], m["invalid_reason"] = True, f"{chk['n_violations']} causality violation(s)"
        runs_by_n.setdefault(int(item["n"]), []).append(m)
    res = rq1_verdict(runs_by_n)
    print(json.dumps(res, indent=1, default=str))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps({"runs": runs_by_n, "verdict": res}, indent=1, default=str))


if __name__ == "__main__":
    main()
