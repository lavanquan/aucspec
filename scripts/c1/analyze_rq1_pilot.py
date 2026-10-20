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


def exact_occupancy(df, start_ms: float, end_ms: float) -> dict:
    """C1.4: exact virtual verifier occupancy.  Every UNIQUE verifier service interval [batch_start, verify_finish] is intersected with
    the measurement window; overlap_b = max(0, min(finish_b, end) - max(start_b, start)); utilisation = sum(overlap) / window.
    All batches overlapping the window are used (not only batches reached through in-window client receives).  Also reports whether any
    two service intervals overlap each other (serialisation violation) - then the sum would exceed the true busy time."""
    b = df.drop_duplicates("verification_batch_id")[["verification_batch_id", "batch_start_ms", "verify_finish_ms"]].sort_values("batch_start_ms")
    s, f = b["batch_start_ms"].to_numpy(float), b["verify_finish_ms"].to_numpy(float)
    ov = [max(0.0, min(fi, end_ms) - max(si, start_ms)) for si, fi in zip(s, f)]
    dur = end_ms - start_ms
    serial_ok = bool(all(s[i + 1] >= f[i] - 1e-6 for i in range(len(s) - 1)))
    return {"utilisation": float(sum(ov)) / dur if dur > 0 else float("nan"), "n_batches_overlapping_window": int(sum(o > 0 for o in ov)),
            "busy_ms": float(sum(ov)), "serialisation_ok": serial_ok}


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
    util_legacy = float(batches["verification_batch_wall_ms"].sum()) / 1000.0 / t_m if len(batches) else 0.0   # superseded (C1.4)
    occ = exact_occupancy(df, start, end) if "verify_finish_ms" in df.columns else None
    util = occ["utilisation"] if occ is not None else util_legacy
    out = {
        "N": n, "window_s": t_m, "rates": rates, "x_min": min(vals), "x_mean": statistics.fmean(vals),
        "Y": sum(vals), "N_x_min": n * min(vals), "jain": jain(vals),
        "x_min_over_mean": (min(vals) / statistics.fmean(vals)) if statistics.fmean(vals) > 0 else float("nan"),
        "verifier_utilisation": util,
        "verifier_utilisation_legacy": util_legacy,
        "verifier_serialisation_ok": None if occ is None else occ["serialisation_ok"],
        "verifier_batches_overlapping_window": None if occ is None else occ["n_batches_overlapping_window"],
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


REQUIRED_SEEDS = (1, 2, 3)      # pre-registered valid seed pairs needed for a substantive verdict


def _by_seed(runs: list[dict], n: int) -> dict:
    out: dict = {}
    for r in runs:
        if "seed" not in r:
            raise ValueError(f"run at N={n} has no run-seed; seeds must be explicit")
        if r["seed"] in out:
            raise ValueError(f"duplicate run record for N={n}, seed={r['seed']}")
        out[r["seed"]] = r
    return out


def rq1_verdict(runs_by_n: dict[int, list[dict]], required_seeds=REQUIRED_SEEDS) -> dict:
    """Apply the pre-registered rules. N=8 and N=16 runs are paired by EXPLICIT run-seed (never positionally);
    duplicates raise; missing/INVALID runs are listed. A substantive verdict needs a valid pair for EVERY required
    seed; otherwise the numbers are kept as descriptive diagnostics under an INCONCLUSIVE(partial) verdict."""
    by = {n: _by_seed(rs, n) for n, rs in runs_by_n.items()}
    invalid = {n: sorted(s for s, r in d.items() if r["invalid"]) for n, d in by.items()}
    invalid = {n: v for n, v in invalid.items() if v}
    clean = {n: {s: r for s, r in d.items() if not r["invalid"]} for n, d in by.items()}
    out: dict = {"invalid_seeds_by_N": invalid, "per_N": {}, "required_seeds": list(required_seeds)}
    for n, d in sorted(clean.items()):
        rs = list(d.values())
        if rs:
            out["per_N"][n] = {"seeds": sorted(d), "Y": _agg(rs, "Y"), "x_min": _agg(rs, "x_min"),
                               "N_x_min": _agg(rs, "N_x_min"), "x_min_over_mean": _agg(rs, "x_min_over_mean"),
                               "utilisation": _agg(rs, "verifier_utilisation"), "batch_size": _agg(rs, "batch_size_mean")}
    a, b = clean.get(N_A, {}), clean.get(N_B, {})
    pairs = sorted(set(a) & set(b))
    missing = [s for s in required_seeds if s not in pairs]
    per_seed = {s: {"Y_ratio": b[s]["Y"] / a[s]["Y"], "x_min_ratio": b[s]["x_min"] / a[s]["x_min"]} for s in pairs}
    out["N8_to_N16"] = {"valid_pairs": pairs, "missing_or_invalid_required_seeds": missing, "per_seed": per_seed}
    if not pairs:
        out["verdict"] = "INCONCLUSIVE(partial): no valid N=8/N=16 seed pair"
        return out
    y_ratio = statistics.fmean(v["Y_ratio"] for v in per_seed.values())
    x_ratio = statistics.fmean(v["x_min_ratio"] for v in per_seed.values())
    plateau = y_ratio <= 1 + PLATEAU_TOL and all(v["Y_ratio"] <= 1 + PLATEAU_TOL + 0.05 for v in per_seed.values())
    floor_falls = x_ratio <= FLOOR_DROP and all(v["x_min_ratio"] <= FLOOR_DROP + 0.05 for v in per_seed.values())
    fair = all(statistics.fmean(r["x_min_over_mean"] for r in clean[n].values()) >= MIN_FAIRNESS for n in (N_A, N_B))
    out["N8_to_N16"].update({"Y_ratio_mean": y_ratio, "x_min_ratio_mean": x_ratio, "plateau": plateau,
                             "floor_falls": floor_falls, "homogeneous_fair": fair})
    if missing:
        out["verdict"] = (f"INCONCLUSIVE(partial): valid pairs for seeds {pairs}, required {list(required_seeds)} "
                          f"(missing/INVALID {missing}); numbers are descriptive diagnostics only")
    elif y_ratio >= UNSATURATED:
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
