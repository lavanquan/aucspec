"""C5-A: Theorem-A style prediction (max-plus cycle R = max{T1+T2, D+W+max(T1,T2)}, K<=2) vs the valid GPU traces, constants fitted on N<=16 only.
Inputs measured at the TARGET N (not predicted): processed tokens per request (ctx), mean useful tokens per round (phi), and the return delay D (finish -> next
arrival, measured in the same run).  Fitted on N<=16: (a, b) of wall = a + b * processed_tokens.  W = 150 ms (batch_wait_ms of the C1 config).
Same catalog (GSM8K-train), same run-seed, same gamma, same runtime H: held out in N only; NOT independent evidence (one seed, one catalog).  usage: heldout_check.py <out.json>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
W_MS = 150.0


def main(out):
    b = pd.read_csv(ROOT / "results" / "c2" / "batches.csv.gz")
    r = pd.read_csv(ROOT / "results" / "c2" / "rounds_replay.csv.gz")
    train = b[(b.N <= 16) & (b.run.isin(["RQ1_CAP16_5a37f03", "RQ1_EXT_28907a6"]))]
    X = np.column_stack([np.ones(len(train)), train.verification_batch_processed_tokens.to_numpy(float)])
    (a, bb), *_ = np.linalg.lstsq(X, train.verification_batch_wall_ms.to_numpy(float), rcond=None)
    res = {"fit_N_le_16": {"a_ms": float(a), "b_ms_per_token": float(bb), "n_batches": int(len(train))}, "W_ms": W_MS, "points": []}
    for run, n, B, role in (("RQ1_CAP16_5a37f03", 8, 16, "in-sample"), ("RQ1_CAP16_5a37f03", 16, 16, "in-sample"), ("RQ1_CAP16_5a37f03", 24, 16, "held-out in N"),
                            ("RQ1_PROBE_2d58666", 28, 16, "held-out in N"), ("RQ1_PROBE_2d58666", 32, 16, "held-out in N")):
        g = r[(r.run == run) & (r.N == n)].sort_values(["client_id", "server_arrival_ms"])
        bt = b[(b.run == run) & (b.N == n)]
        w0, w1 = float(bt.experiment_warmup_end_ms.iloc[0]), float(bt.experiment_measurement_end_ms.iloc[0])
        gw = g[(g.client_receive_ms >= w0) & (g.client_receive_ms <= w1)]
        y_meas = float(gw.committed_token_count.sum() / ((w1 - w0) / 1000.0))
        phi = float(gw.committed_token_count.mean())
        btw = bt[(bt.verify_finish_ms >= w0) & (bt.batch_start_ms <= w1)]
        proc = float(btw.verification_batch_processed_tokens.sum() / btw.verification_batch_size.sum())
        d = bb * proc
        part = [min(B, n)] if n <= B else [B, n - B]
        T = [a + p * d for p in part]
        gg = g.copy()
        gg["next"] = gg.groupby("client_id").server_arrival_ms.shift(-1)
        D = float((gg["next"] - gg["verify_finish_ms"]).median())
        R = max(sum(T), D + W_MS + max(T)) if len(T) > 1 else D + W_MS + T[0]
        y_pred = n * phi / (R / 1000.0)
        # structure-only check: replace the fitted cost model by the MEASURED mean wall time of each batch size in this run (tests the max-plus structure, not the cost model)
        wall_by_size = btw.groupby("verification_batch_size").verification_batch_wall_ms.mean().to_dict()
        Tm = [float(wall_by_size[p]) for p in part]
        R_struct = max(sum(Tm), D + W_MS + max(Tm)) if len(Tm) > 1 else D + W_MS + Tm[0]
        res["points"].append({"run": run, "N": n, "role": role, "partition": part, "D_ms_measured": D, "T_ms": T, "R_pred_ms": float(R), "R_meas_ms": float(n * phi / y_meas * 1000.0),
                              "Y_meas": y_meas, "Y_pred": float(y_pred), "rel_err_pct": float(100 * (y_pred - y_meas) / y_meas),
                              "T_measured_ms": Tm, "R_struct_ms": float(R_struct), "R_struct_err_pct": float(100 * (R_struct - n * phi / y_meas * 1000.0) / (n * phi / y_meas * 1000.0)), "limiting_term": "server (T1+T2)" if len(T) > 1 and sum(T) >= D + W_MS + max(T) else "client (D+W+Tmax)"})
    Path(out).write_text(json.dumps(res, indent=1))
    print(res["fit_N_le_16"])
    for p in res["points"]:
        print(p["N"], p["role"], p["partition"], "D %.1f" % p["D_ms_measured"], "R pred %.0f meas %.0f" % (p["R_pred_ms"], p["R_meas_ms"]), "Y err %+.1f%%" % p["rel_err_pct"], "| structure-only R %.0f (err %+.1f%%)" % (p["R_struct_ms"], p["R_struct_err_pct"]), p["limiting_term"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "heldout_check.json")
