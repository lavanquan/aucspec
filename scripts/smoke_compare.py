"""Diagnostic comparison of smoke runs (non-certification). Usage: smoke_compare.py <label=run_dir> ..."""
import json
import sys
from pathlib import Path

import pandas as pd

rows = {}
for spec in sys.argv[1:]:
    label, run = spec.split("=", 1)
    run = Path(run)
    for part in sorted((run / "cache_parts").glob("*.jsonl")):
        rec = json.loads(part.read_text().splitlines()[0])
        m = rec["measurement"]
        n = m["n_active"]
        csv = run / "raw" / f"capacity_dpp_n{n}_x3.0_s1.csv"
        df = pd.read_csv(csv, usecols=["in_measurement_window", "verification_batch_id", "verification_batch_size",
                                       "virtual_round_latency_ms", "gamma", "useful_tokens", "gamma_rate_signal_age_rounds",
                                       "server_queue", "verification_batch_wall_ms", "verification_batch_service_ms",
                                       "batch_start_ms", "verify_finish_ms", "draft_start_ms"])
        w = df[df["in_measurement_window"] == True]  # noqa: E712
        b = w.drop_duplicates("verification_batch_id")
        span = b["verify_finish_ms"].max() - b["batch_start_ms"].min()
        rows[f"{label} N={n}"] = {
            "min_rate_tps": m["min_rate_tps"], "goodput_tps": m["goodput_tps"],
            "median_client_rate": sorted(m["per_client_rates"].values())[len(m["per_client_rates"]) // 2],
            "mean_gamma": m["mean_gamma"], "mean_batch_size": float(b["verification_batch_size"].mean()),
            "mean_round_latency_ms": float(w["virtual_round_latency_ms"].mean()),
            "mean_batch_wall_ms": float(b["verification_batch_wall_ms"].mean()),
            "mean_modeled_service_ms": float(b["verification_batch_service_ms"].mean()),
            "verifier_busy_fraction(modeled)": float(b["verification_batch_service_ms"].sum() / span),
            "rate_signal_age_min": float(df["gamma_rate_signal_age_rounds"].min()),
            "rate_signal_age_frac_zero": float((df["gamma_rate_signal_age_rounds"] == 0).mean()),
            "measurement_rounds": int(len(w)), "exhausted": rec["measurement"]["extra"]["exhausted_clients"],
        }
print(json.dumps(rows, indent=2))
