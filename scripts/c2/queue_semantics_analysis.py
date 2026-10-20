"""C2 Phase B: what do the logged virtual queues actually measure?  Uses the compact real-trace tables (loaded C1 runs, fixed gamma=2 policy, so the
queue trajectories are those of that policy - this is NOT a replay of Capacity-DPP).  usage: queue_semantics_analysis.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
r = pd.read_csv(ROOT / "results" / "c2" / "rounds_replay.csv.gz")
b = pd.read_csv(ROOT / "results" / "c2" / "batches.csv.gz")
out = {"label": "real C1 traces under fixed gamma=2 (cap-16 runs: server_queue_before = causal decision-time state)"}
rows = []
for (run, n), g in r.groupby(["run", "N"]):
    if "cap16" not in run.lower() and "CAP16" not in run and "PROBE" not in run:
        continue
    bb = b[(b["run"] == run) & (b["N"] == n)].sort_values("batch_start_ms")
    g = g.sort_values("draft_start_ms")
    # most recent completed batch (finish <= decision time) -> its measured wall time
    fin = bb["verify_finish_ms"].to_numpy()
    wall = bb["verification_batch_wall_ms"].to_numpy()
    idx = np.searchsorted(fin, g["draft_start_ms"].to_numpy(), side="right") - 1
    ok = idx >= 0
    last_wall = wall[idx[ok]]
    qs = g["server_queue_before"].to_numpy()[ok]
    A = np.column_stack([np.ones(ok.sum()), last_wall])
    beta, *_ = np.linalg.lstsq(A, qs, rcond=None)
    pred = A @ beta
    r2 = 1 - ((qs - pred) ** 2).sum() / ((qs - qs.mean()) ** 2).sum()
    start, end = bb["experiment_warmup_end_ms"].iloc[0], bb["experiment_measurement_end_ms"].iloc[0]
    s = bb["batch_start_ms"].to_numpy()
    f = bb["verify_finish_ms"].to_numpy()
    util = float(np.maximum(0, np.minimum(f, end) - np.maximum(s, start)).sum() / (end - start))
    w = g[(g["client_receive_ms"] >= start) & (g["client_receive_ms"] <= end)]
    rows.append({"run": run, "N": int(n), "verifier_busy": util, "Qs_decision_mean_ms": float(w["server_queue_before"].mean()), "Qs_decision_p95_ms": float(w["server_queue_before"].quantile(0.95)),
                 "Qs_vs_last_batch_wall_R2": float(r2), "Qs_slope_per_ms_of_last_wall": float(beta[1]), "Qs_intercept_ms": float(beta[0]),
                 "mean_queue_wait_ms": float((w["batch_start_ms"] - w["server_arrival_ms"]).mean()),
                 "Z_mean": float(w["z_queue"].mean()), "Z_p95": float(w["z_queue"].quantile(0.95)), "Z_max": float(w["z_queue"].max()),
                 "Qi_mean_ms": float(w["device_queue"].mean()), "Qi_max_ms": float(w["device_queue"].max()),
                 "Qi_positive_fraction": float((w["device_queue"] > 0).mean()), "round_ms_mean": float((w["client_receive_ms"] - w["draft_start_ms"]).mean())})
out["per_run"] = rows
Path(ROOT / "results" / "c2" / "queue_semantics_real_traces.json").write_text(json.dumps(out, indent=1))
for x in rows:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})
