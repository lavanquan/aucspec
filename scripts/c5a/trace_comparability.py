"""C5-A: which existing GPU traces are scientifically comparable (catalog, cap, seed, gamma, runtime H, protocol) and what partition do the in-window batches show?
usage: trace_comparability.py <out.json>   (reads results/c2/batches.csv.gz, results/c2/rounds_replay.csv.gz, results/c2/rq1_evidence_inventory.json)"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

META = {  # from the committed C1 reports (results/AUCSPEC_RQ1_*.md, results/c2/rq1_evidence_inventory.json)
    "RQ1_PILOT_4a7e866": dict(catalog="GSM8K-test (hash 628857be)", serving="batch_cap_32 (cap never binds for N<=16)", protocol="original gate"),
    "RQ1_EXT_28907a6": dict(catalog="GSM8K-train (hash 02e79304)", serving="batch_cap_32 (cap never binds for N<=16)", protocol="original gate; N=24 OOM, no data"),
    "RQ1_CAP16_5a37f03": dict(catalog="GSM8K-train (hash 02e79304)", serving="batch_cap_16", protocol="original gate; N=24 FAILED it (3.07)"),
    "RQ1_PROBE_2d58666": dict(catalog="GSM8K-train (hash 02e79304)", serving="batch_cap_16", protocol="exploratory_v2; N=28, 32 FAIL the original gate (3.21, 3.28)"),
}


def main(out):
    b = pd.read_csv(ROOT / "results" / "c2" / "batches.csv.gz")
    inv = json.loads((ROOT / "results" / "c2" / "rq1_evidence_inventory.json").read_text())
    byN = {(p["tag"], p["N"]): p for p in inv["points"]}
    res = {"common": {"run_seed": 1, "population_seed": 20261020, "gamma": 2, "runtime": "H = V1-batchwall-tritonfa33 (vLLM 0.9.1 V0, Triton FA, prefix caching off, chunked prefill off)", "verification": "V1 greedy, batch_wall, FCFS"}, "runs": []}
    for run in sorted(b.run.unique()):
        for n in sorted(b[b.run == run].N.unique()):
            bb = b[(b.run == run) & (b.N == n)]
            w0, w1 = float(bb.experiment_warmup_end_ms.iloc[0]), float(bb.experiment_measurement_end_ms.iloc[0])
            bw = bb[(bb.verify_finish_ms >= w0) & (bb.batch_start_ms <= w1)]
            hist = {int(k): int(v) for k, v in bw.verification_batch_size.value_counts().sort_index().items()}
            cap = None if pd.isna(bb.verifier_max_batch_size.iloc[0]) else int(bb.verifier_max_batch_size.iloc[0])
            B = cap if cap else 32
            part = []
            r_ = int(n)
            while r_ > 0:
                part.append(min(B, r_))
                r_ -= B
            e = byN.get((run.replace("RQ1_", "RQ1_"), n))
            res["runs"].append({"run": run, "N": int(n), **META.get(run, {}), "effective_cap": B, "in_window_batch_size_histogram": hist,
                                "greedy_partition_prediction": sorted(part, reverse=True), "mean_batch_size": float(bw.verification_batch_size.mean()),
                                "n_batches_in_window": int(len(bw))})
    # cross-launch replicate pairs (same catalog, seed, gamma, H; cap not binding)
    inv_pts = {(p["campaign"], p["N"]): p for p in inv["points"]}
    res["cross_launch_replicates_same_conditions"] = [
        {"N": n, "C1.2-extension": {k: inv_pts[("C1.2-extension", n)][k] for k in ("x_min", "Y")}, "C1.3-cap16": {k: inv_pts[("C1.3-cap16", n)][k] for k in ("x_min", "Y")},
         "x_min_rel_diff_pct": 100 * (inv_pts[("C1.2-extension", n)]["x_min"] - inv_pts[("C1.3-cap16", n)]["x_min"]) / inv_pts[("C1.3-cap16", n)]["x_min"]} for n in (8, 16)]
    p8, p12 = inv_pts[("C1.1-pilot", 8)], inv_pts[("C1.1-pilot", 12)]
    res["pilot_N8_vs_N12"] = {"x_min_N8": p8["x_min"], "x_min_N12": p12["x_min"], "rel_diff_pct": 100 * (p12["x_min"] - p8["x_min"]) / p8["x_min"],
                              "note": "same catalog (test), same cap; difference is of the order of the cross-launch replicate difference; N=8 was the first run of its job with 45 slow batches"}
    Path(out).write_text(json.dumps(res, indent=1, default=int))
    for r in res["runs"]:
        print(r["run"], r["N"], "cap", r["effective_cap"], r["in_window_batch_size_histogram"], "greedy", r["greedy_partition_prediction"])
    print(res["cross_launch_replicates_same_conditions"])
    print(res["pilot_N8_vs_N12"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "trace_comparability.json")
