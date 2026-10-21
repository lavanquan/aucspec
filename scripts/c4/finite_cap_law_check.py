"""C4 Path-1 evidence: does a saturated finite-batch-cap law with full-context cost, FITTED ONLY ON N<=16 RUNS, predict the held-out N=24,28,32 goodput?
Law (saturated verifier, batches of the observed cohort sizes):  cycle R(N) = K_N * a + sum_i d_i,   d_i = b * (ctx_i + gamma),   Y(N) = N * phi_bar / R(N)
(K_N = number of cap-limited batches per cycle; phi_bar = measured mean useful tokens per round at that N; ctx_bar measured).  In-sample fit on batches of
runs with N<=16 (GSM8K-train cap16/ext); prediction for N>=24 uses ONLY (a, b) from those fits plus the measured ctx_bar, phi_bar of the target run.
This is a partial out-of-sample check on ONE seed/catalog; (a, b) come from loaded runs.  CPU only.  usage: finite_cap_law_check.py <out.json>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main(out):
    b = pd.read_csv(ROOT / "results" / "c2" / "batches.csv.gz")
    r = pd.read_csv(ROOT / "results" / "c2" / "rounds_replay.csv.gz")
    train = b[(b.N <= 16) & (b.run.isin(["RQ1_CAP16_5a37f03", "RQ1_EXT_28907a6"]))]
    X = np.column_stack([np.ones(len(train)), train.verification_batch_processed_tokens.to_numpy(float)])
    y = train.verification_batch_wall_ms.to_numpy(float)
    (a, bb), *_ = np.linalg.lstsq(X, y, rcond=None)
    res = {"label": "Finite-cap saturated capacity law; (a,b) fitted on N<=16 only; targets N=24,28,32 held out (one seed, GSM8K-train).", "fit": {"a_ms": float(a), "b_ms_per_token": float(bb), "n_batches": int(len(train))}, "points": []}
    for run, n, cohorts in (("RQ1_CAP16_5a37f03", 24, [16, 8]), ("RQ1_PROBE_2d58666", 28, [14, 14]), ("RQ1_PROBE_2d58666", 32, [16, 16]), ("RQ1_CAP16_5a37f03", 16, [16]), ("RQ1_CAP16_5a37f03", 8, [8])):
        g = r[(r.run == run) & (r.N == n)]
        bt = b[(b.run == run) & (b.N == n)]
        w0, w1 = float(bt.experiment_warmup_end_ms.iloc[0]), float(bt.experiment_measurement_end_ms.iloc[0])
        gw = g[(g.client_receive_ms >= w0) & (g.client_receive_ms <= w1)]
        y_meas = float(gw.committed_token_count.sum() / ((w1 - w0) / 1000.0))
        x_min = float((gw.groupby("client_id").committed_token_count.sum() / ((w1 - w0) / 1000.0)).min())
        phi = float(gw.committed_token_count.mean())
        btw = bt[(bt.verify_finish_ms >= w0) & (bt.batch_start_ms <= w1)]
        proc_per_req = float((btw.verification_batch_processed_tokens.sum()) / btw.verification_batch_size.sum())
        d = bb * proc_per_req
        R = len(cohorts) * a + n * d
        saturated = res_sat = (n >= 24)
        y_pred = n * phi / (R / 1000.0)
        res["points"].append({"run": run, "N": n, "cohorts": cohorts, "saturated_regime": bool(saturated), "Y_measured": y_meas, "Y_predicted": float(y_pred), "rel_error_pct": float(100 * (y_pred - y_meas) / y_meas),
                              "x_mean_pred": float(y_pred / n), "x_min_measured": x_min, "phi_bar": phi, "processed_per_request": proc_per_req, "cycle_ms": float(R),
                              "note": "held-out N" if n >= 24 else "N<=16 is NOT saturated: device+RTT+hold W add to the cycle, so the saturated law over-predicts (shown for contrast)"})
    Path(out).write_text(json.dumps(res, indent=1))
    for p in res["points"]:
        print(p["N"], p["cohorts"], "Y meas %.1f pred %.1f err %+.1f%%" % (p["Y_measured"], p["Y_predicted"], p["rel_error_pct"]))
    print(res["fit"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "c4" / "finite_cap_law_check.json"))
