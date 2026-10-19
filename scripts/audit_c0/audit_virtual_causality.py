"""Virtual-time causality diagnostics from one simulator round CSV (CPU, read-only).

(1) Batch formation: with batch_wait_ms W, a work-conserving single verifier should serve every request whose
    virtual server arrival is <= batch_ready of an earlier batch in that batch (while capacity remains).
    "Missed arrivals" are requests that arrived (virtual) before an earlier batch became ready but were served later.
(2) Controller state: for each depth decision, did the server-queue state already include a batch whose virtual
    finish lies AFTER the decision's virtual time?  (needs the decision_* columns written by the causality logging.)
"""
import json
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
out = {"rows": int(len(df))}
b = df.sort_values("verification_batch_id")
batches = b.groupby("verification_batch_id").agg(
    size=("verification_batch_size", "first"), ready=("batch_ready_ms", "first"), start=("batch_start_ms", "first"),
    finish=("verify_finish_ms", "first"), tokcost=("verification_batch_token_cost", "first"), budget=("verification_batch_token_budget", "first"))
out["n_batches"] = int(len(batches))
arr = b[["verification_batch_id", "server_arrival_ms", "client_id"]].to_numpy()
ids = batches.index.to_numpy()
ready = batches["ready"].to_numpy()
size = batches["size"].to_numpy()
by_batch = {i: g["server_arrival_ms"].to_numpy() for i, g in b.groupby("verification_batch_id")}
missed = 0
missed_batches = 0
late_gap = []
for k, i in enumerate(ids):
    # requests served in strictly later batches that had arrived before this batch became ready
    later = np.concatenate([by_batch[j] for j in ids[k + 1: k + 40]]) if k + 1 < len(ids) else np.array([])
    m = later[later <= ready[k] + 1e-6]
    if len(m) and size[k] < 32:
        missed += len(m)
        missed_batches += 1
        late_gap.extend((ready[k] - m).tolist())
out["missed_arrivals"] = {"requests": int(missed), "batches_affected": int(missed_batches),
                          "fraction_of_batches": float(missed_batches / max(1, len(ids))),
                          "mean_early_by_ms": float(np.mean(late_gap)) if late_gap else 0.0}
# overtaking: a request served earlier although it arrived later than a request served in a later batch
first_arr = np.array([by_batch[i].min() for i in ids])
out["batches_whose_earliest_arrival_is_later_than_a_following_batch"] = int((np.diff(first_arr) < -1e-6).sum())
cols = {"decision_virtual_ms", "decision_server_state_max_finish_ms", "stamped_server_state_max_finish_ms"}
if cols.issubset(df.columns):
    d = df["decision_virtual_ms"].to_numpy(float)
    f = df["decision_server_state_max_finish_ms"].to_numpy(float)
    s = df["stamped_server_state_max_finish_ms"].to_numpy(float)
    ahead = f - d
    out["controller_state_causality"] = {
        "decisions": int(len(df)),
        "state_includes_future_batch(finish>decision_time)": int((ahead > 1e-6).sum()),
        "fraction": float((ahead > 1e-6).mean()),
        "future_by_ms_mean_when_violated": float(ahead[ahead > 1e-6].mean()) if (ahead > 1e-6).any() else 0.0,
        "future_by_ms_p95": float(np.percentile(ahead[ahead > 1e-6], 95)) if (ahead > 1e-6).any() else 0.0,
        "stamped_state_future_fraction": float(((s - d) > 1e-6).mean()),
    }
    sq = df[["decision_server_state_max_finish_ms"]]
    out["note"] = "server-queue price is zero in verifier_price_mode=zero, so this causality gap does not currently change any decision"
print(json.dumps(out, indent=1))
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(json.dumps(out, indent=1))
