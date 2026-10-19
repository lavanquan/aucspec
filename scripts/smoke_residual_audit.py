"""Where does the loaded-run batch time exceed the isolated calibration? (existing logs, CPU only; no cause is assumed)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

cm = json.load(open(sys.argv[1]))["coefficients_ms"]
out = {}
for label, p in (a.split("=", 1) for a in sys.argv[3:]):
    d = pd.read_csv(p, usecols=["verification_batch_id", "verification_batch_size", "verification_batch_wall_ms",
                                "verification_batch_context_tokens_sum", "verification_batch_token_cost",
                                "batch_start_ms", "in_measurement_window", "n_active"], low_memory=False)
    d = d.drop_duplicates("verification_batch_id").sort_values("batch_start_ms")
    B = d.verification_batch_size.to_numpy(float)
    pred = cm["theta0_ms"] + cm["per_request_const_ms"] * B + cm["per_context_token_ms"] * d.verification_batch_context_tokens_sum.to_numpy(float) \
        + cm["per_draft_token_ms"] * (d.verification_batch_token_cost.to_numpy(float) - B)
    res = d.verification_batch_wall_ms.to_numpy(float) - pred
    t = d.batch_start_ms.to_numpy(float)
    thirds = np.array_split(np.arange(len(d)), 3)
    na = d["n_active"].to_numpy()
    out[label] = {
        "n_batches": int(len(d)), "residual_mean_ms": float(res.mean()), "residual_median_ms": float(np.median(res)),
        "residual_quantiles_5_25_75_95": [float(x) for x in np.percentile(res, [5, 25, 75, 95])],
        "residual_by_time_third_mean_ms": [float(res[i].mean()) for i in thirds],
        "corr_residual_vs_virtual_time": float(np.corrcoef(res, t)[0, 1]),
        "corr_residual_vs_batch_size": float(np.corrcoef(res, B)[0, 1]),
        "n_active_values": sorted(set(int(x) for x in na)),
    }
print(json.dumps(out, indent=1))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
