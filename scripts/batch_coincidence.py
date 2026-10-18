"""Why do the batch-wall and V1-control N=10 smoke runs both contain 7,460 batches? (CPU, read-only)"""
import json
import sys
from pathlib import Path

import pandas as pd

R = Path(sys.argv[1])
cols = ["client_id", "sample_id", "round_id", "verification_batch_id", "verification_batch_size", "useful_tokens",
        "committed_token_count", "generated_for_sample", "in_measurement_window", "in_warmup_window", "gamma",
        "batch_start_ms", "verify_finish_ms", "server_arrival_ms", "batch_ready_ms"]
res = {}
dfs = {}
for name, rel in {"bw": "bw_smoke60_r2_20261019", "v1": "v1ctl_smoke60_r2_20261019"}.items():
    d = pd.read_csv(R / rel / "raw" / "capacity_dpp_n10_x3.0_s1.csv", usecols=cols, low_memory=False)
    dfs[name] = d
    b = d.drop_duplicates("verification_batch_id")
    res[name] = {
        "rows_requests": int(len(d)), "batches": int(b.verification_batch_id.nunique()),
        "samples": int(d.groupby(["client_id", "sample_id"]).ngroups),
        "sum_committed_tokens": int(d.committed_token_count.sum()), "sum_useful_tokens": int(d.useful_tokens.sum()),
        "rounds_per_sample_mean": float(d.groupby(["client_id", "sample_id"]).size().mean()),
        "batch_size_hist": {int(k): int(v) for k, v in b.verification_batch_size.value_counts().sort_index().items()},
        "measurement_rows": int((d.in_measurement_window == True).sum()),  # noqa: E712
        "batches_with_measurement_row": int(d[d.in_measurement_window == True].verification_batch_id.nunique()),  # noqa: E712
        "last_batch_id": int(d.verification_batch_id.max()),
        "virtual_end_ms": float(d.verify_finish_ms.max()),
    }
# are the two row sets identical in membership/sizes?
a, c = dfs["bw"], dfs["v1"]
res["same_batch_size_sequence"] = bool(
    a.drop_duplicates("verification_batch_id").verification_batch_size.tolist()
    == c.drop_duplicates("verification_batch_id").verification_batch_size.tolist())
# how do batches relate to request count? distribution of requests per batch id vs rounds
for name, d in dfs.items():
    per_sample = d.groupby(["client_id", "sample_id"]).agg(rounds=("round_id", "size"), tok=("committed_token_count", "sum"))
    res[name]["tokens_per_sample_mean"] = float(per_sample.tok.mean())
    res[name]["per_client_rows"] = d.groupby("client_id").size().tolist()
    res[name]["per_client_samples"] = d.groupby("client_id").sample_id.nunique().tolist()
# ids of samples (same questions in both runs?)
res["same_sample_sets"] = bool(sorted(set(map(tuple, a[["client_id", "sample_id"]].drop_duplicates().values.tolist())))
                                == sorted(set(map(tuple, c[["client_id", "sample_id"]].drop_duplicates().values.tolist()))))
print(json.dumps(res, indent=1))
Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
