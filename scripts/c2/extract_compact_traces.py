"""Extract COMPACT per-batch and per-round tables from existing raw C1 traces (read-only; CPU).  usage:
  extract_compact_traces.py <scratch_root> <out_dir>
Per-batch table: one row per unique verifier batch of every run (size, context sums, processed tokens, gammas, measured wall, start/finish).
Per-round replay table (high-load cap-16 runs only): the controller state logged for each round."""
import glob
import re
import sys
from pathlib import Path

import pandas as pd

TAGS = {"RQ1_PILOT_4a7e866": ("pilot_test_cap32", "GSM8K-test"), "RQ1_EXT_28907a6": ("ext_train_cap32", "GSM8K-train"),
        "RQ1_CAP16_5a37f03": ("cap16", "GSM8K-train"), "RQ1_PROBE_2d58666": ("cap16_probe", "GSM8K-train")}
BATCH_COLS = ["verification_batch_id", "verification_batch_size", "verification_batch_wall_ms", "verification_batch_processed_tokens",
              "verification_batch_context_tokens_sum", "verification_batch_context_tokens_max", "batch_start_ms", "verify_finish_ms",
              "experiment_warmup_end_ms", "experiment_measurement_end_ms", "verifier_max_batch_size"]
ROUND_COLS = ["client_id", "round_id", "sample_id", "draft_start_ms", "client_receive_ms", "gamma", "selected_gamma", "committed_token_count",
              "accepted_length", "alpha_hat", "z_queue", "device_queue", "server_queue_before", "server_queue_after", "verification_batch_id",
              "verification_batch_size", "server_arrival_ms", "batch_start_ms", "verify_finish_ms", "draft_latency_ms", "upload_ms",
              "download_ms", "uplink_rate_mbps", "rtt_ms", "uplink_bandwidth_mhz"]


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    batches, rounds = [], []
    for tag, (label, cat) in TAGS.items():
        for csv in sorted(glob.glob(str(root / tag / "raw" / "fixed_gamma_n*_x3.0_s*.csv"))):
            n = int(re.search(r"_n(\d+)_", csv).group(1))
            df = pd.read_csv(csv, low_memory=False)
            cols = [c for c in BATCH_COLS if c in df.columns]
            b = df.drop_duplicates("verification_batch_id")[cols].copy()
            g = df.groupby("verification_batch_id")
            b["gamma_sum"] = g["gamma"].sum().reindex(b["verification_batch_id"]).to_numpy()
            b["positions_sum"] = (g["gamma"].sum() + g["gamma"].count()).reindex(b["verification_batch_id"]).to_numpy()
            b["run"], b["serving"], b["catalog"], b["N"] = tag, label, cat, n
            batches.append(b)
            if label in ("cap16", "cap16_probe") or (label == "ext_train_cap32" and n in (12, 16)):
                r = df[[c for c in ROUND_COLS if c in df.columns]].copy()
                r["run"], r["N"] = tag, n
                rounds.append(r)
    pd.concat(batches).to_csv(out / "batches.csv.gz", index=False, compression="gzip")
    pd.concat(rounds).to_csv(out / "rounds_replay.csv.gz", index=False, compression="gzip")
    print(len(pd.concat(batches)), "batches;", len(pd.concat(rounds)), "round rows")


if __name__ == "__main__":
    main()
