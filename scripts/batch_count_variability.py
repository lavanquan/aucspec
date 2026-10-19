"""Batch-count variability across certified V1 N=10 runs (existing logs, CPU only)."""
import json
import sys
from pathlib import Path

import pandas as pd

R = Path(sys.argv[1])
rows = []
for p in sorted(R.glob("capacity_dpp_n*_x3.0_s*.csv")):
    d = pd.read_csv(p, usecols=["verification_batch_id", "client_id", "sample_id", "committed_token_count"], low_memory=False)
    n = int(p.name.split("_n")[1].split("_")[0])
    rows.append({"file": p.name, "n_clients": n, "batches": int(d.verification_batch_id.nunique()),
                 "requests": int(len(d)), "samples": int(d.groupby(["client_id", "sample_id"]).ngroups),
                 "committed": int(d.committed_token_count.sum())})
df = pd.DataFrame(rows)
print(df.groupby("n_clients").agg(runs=("batches", "size"), batches_min=("batches", "min"), batches_max=("batches", "max"),
                                  batches_mean=("batches", "mean"), batches_std=("batches", "std"),
                                  requests_mean=("requests", "mean")).to_string())
n10 = df[df.n_clients == 10].sort_values("batches")
print(n10[["file", "batches", "requests", "samples", "committed"]].to_string(index=False))
Path(sys.argv[2]).write_text(df.to_json(orient="records", indent=1))
