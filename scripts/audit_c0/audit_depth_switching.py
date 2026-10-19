import json
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False).sort_values(["client_id", "round_id"])
g = df.groupby("client_id")
df["prev_device_queue"] = g["device_queue"].shift(1)
df["prev_selected"] = g["selected_gamma"].shift(1)
d = df.dropna(subset=["prev_device_queue"])
out = {"rows": int(len(d))}
pos = d["prev_device_queue"] > 1e-9
out["P(selected=0 | prev device_queue>0)"] = float((d.loc[pos, "selected_gamma"] == 0).mean())
out["P(selected=0 | prev device_queue==0)"] = float((d.loc[~pos, "selected_gamma"] == 0).mean())
out["prev_device_queue_threshold_scan"] = {
    str(t): float((d.loc[d["prev_device_queue"] > t, "selected_gamma"] == 0).mean()) if (d["prev_device_queue"] > t).any() else None
    for t in (0, 5, 10, 20, 40)}
tr = pd.crosstab(d["prev_selected"], d["selected_gamma"])
out["transition_counts(prev->selected)"] = {str(k): {str(c): int(v) for c, v in r.items()} for k, r in tr.iterrows()}
# per-client sequence example
ex = df[df.client_id == df.client_id.iloc[0]].head(24)
out["example_client_sequence"] = list(zip(ex["round_id"].astype(int).tolist(), ex["selected_gamma"].astype(int).tolist(),
                                          ex["device_queue"].round(1).tolist(), ex["z_queue"].round(1).tolist()))
print(json.dumps(out, indent=1))
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(json.dumps(out, indent=1))
