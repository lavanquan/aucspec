import json
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
out = {"rows": int(len(df))}
g = df.groupby("client_id")
c = g.agg(first=("draft_start_ms", "min"), last=("client_receive_ms", "max"), tokens=("committed_token_count", "sum"), rounds=("round_id", "size"))
c["span_s"] = (c["last"] - c["first"]) / 1000.0
c["rate_tps"] = c["tokens"] / c["span_s"]
out["per_client"] = {"span_s_min": float(c.span_s.min()), "span_s_max": float(c.span_s.max()), "tokens": c.tokens.tolist(),
                     "rate_tps_min": float(c.rate_tps.min()), "rate_tps_max": float(c.rate_tps.max()), "rate_tps_mean": float(c.rate_tps.mean())}
cols = ["draft_latency_ms", "rtt_ms", "upload_ms", "download_ms", "virtual_round_latency_ms", "wall_round_latency_ms"]
w = df.copy()
w["queue_wait_ms"] = w["batch_start_ms"] - w["server_arrival_ms"]
w["service_ms"] = w["verify_finish_ms"] - w["batch_start_ms"]
out["mean_ms"] = {k: float(w[k].mean()) for k in cols + ["queue_wait_ms", "service_ms"]}
out["tokens_per_round_mean"] = float(w["committed_token_count"].mean())
out["window_ms"] = [float(w["experiment_warmup_end_ms"].iloc[0]), float(w["experiment_measurement_end_ms"].iloc[0])]
win = w[w["in_measurement_window"].astype(bool)]
out["window_rows"] = int(len(win))
out["window_tokens_per_client"] = win.groupby("client_id")["committed_token_count"].sum().tolist()
# rate by time slice (is the rate stationary within the run?)
t = w["client_receive_ms"] / 1000.0
for a, b in ((0, 5), (5, 10), (10, 15), (15, 25)):
    s = w[(t >= a) & (t < b)]
    out[f"agg_tok_per_s[{a},{b})"] = float(s["committed_token_count"].sum() / (b - a))
out["max_virtual_time_s"] = float(w["client_receive_ms"].max() / 1000.0)
print(json.dumps(out, indent=1))
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(json.dumps(out, indent=1))
