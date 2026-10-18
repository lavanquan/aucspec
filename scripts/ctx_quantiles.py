"""Verifier context-length quantiles from canonical smoke traces (per-round context incl. generated history)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1])
out = {}
for name, rel in {
    "bw_n10_all_rounds": "bw_smoke60_r2_20261019/raw/capacity_dpp_n10_x3.0_s1.csv",
    "v1ctl_n10_all_rounds": "v1ctl_smoke60_r2_20261019/raw/capacity_dpp_n10_x3.0_s1.csv",
    "bw_n1_all_rounds": "bw_smoke60_r2_20261019/raw/capacity_dpp_n1_x3.0_s1.csv",
}.items():
    d = pd.read_csv(R / rel, usecols=["client_id", "sample_id", "round_id", "context_token_ids", "in_measurement_window"],
                    low_memory=False)
    n = d["context_token_ids"].astype(str).str.count(",") + 1
    n_hist = None
    q = {f"p{k}": float(np.percentile(n, k)) for k in (1, 5, 10, 25, 50, 75, 90, 95, 99, 100)}
    out[name] = {"n_rounds": int(len(n)), "n_samples": int(d.groupby(["client_id", "sample_id"]).ngroups), **q,
                 "mean": float(n.mean()),
                 "in_window": {f"p{k}": float(np.percentile(n[d["in_measurement_window"] == True], k)) for k in (10, 50, 90)}}  # noqa: E712
print(json.dumps(out, indent=1))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
