"""Which control signals are actually non-zero / decision-relevant in a Capacity-DPP run? (CPU, read-only)"""
import json
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], low_memory=False)
w = df
out = {"rows": int(len(w)), "policy": sorted(set(w["controller_policy"].astype(str)))}


def stats(c):
    v = w[c].astype(float)
    return {"mean": float(v.mean()), "max": float(v.max()), "frac_positive": float((v > 1e-9).mean())}


for c in ("z_queue", "device_queue", "server_queue", "lambda_price"):
    out[c] = stats(c)
out["selected_gamma_counts"] = {int(k): int(v) for k, v in w["selected_gamma"].value_counts().sort_index().items()}
out["gamma_counts"] = {int(k): int(v) for k, v in w["gamma"].value_counts().sort_index().items()}
out["theta_f_stamped_nonzero_frac"] = float((w["stamped_theta_f"].astype(float) > 0).mean()) if "stamped_theta_f" in w else None
# depth vs device price
dq = w["device_queue"].astype(float) > 1e-9
out["mean_selected_gamma_when_device_queue>0"] = float(w.loc[dq, "selected_gamma"].mean()) if dq.any() else None
out["mean_selected_gamma_when_device_queue==0"] = float(w.loc[~dq, "selected_gamma"].mean())
zq = w["z_queue"].astype(float)
out["mean_selected_gamma_by_z_queue_tercile"] = {str(q): float(g["selected_gamma"].mean()) for q, g in w.groupby(pd.qcut(zq.rank(method="first"), 3, labels=["low", "mid", "high"]))}
out["corr_selected_gamma_vs_alpha_hat"] = float(np.corrcoef(w["selected_gamma"].astype(float), w["alpha_hat"].astype(float))[0, 1]) if w["alpha_hat"].std() > 0 else None
out["controller_forced_exploration_frac"] = float(w["controller_forced_exploration"].astype(bool).mean()) if "controller_forced_exploration" in w else None
out["forced_service_frac_batches"] = float(w.drop_duplicates("verification_batch_id")["forced_service"].astype(bool).mean())
b = w.drop_duplicates("verification_batch_id")
out["batch_size_mean"] = float(b["verification_batch_size"].mean())
out["batch_token_cost_vs_budget_max_fill"] = float((b["verification_batch_token_cost"] / b["verification_batch_token_budget"]).max())
out["batch_hit_max_batch_size_32_frac"] = float((b["verification_batch_size"] >= 32).mean())
out["gamma_rate_signal_age_min"] = float(w["gamma_rate_signal_age_rounds"].min())
out["uplink_allocator"] = sorted(set(w["uplink_allocator"].astype(str)))
out["uplink_rate_mbps_cv_across_clients"] = float(w.groupby("client_id")["realized_uplink_rate_mbps"].mean().std() / w["realized_uplink_rate_mbps"].mean())
out["x_requirement"] = float(w["x_requirement"].iloc[0])
print(json.dumps(out, indent=1))
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(json.dumps(out, indent=1))
