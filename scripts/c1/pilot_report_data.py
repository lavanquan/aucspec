"""Compact per-run report data for the RQ1 pilot (CPU).  usage: pilot_report_data.py <scratch_root> <tag> <out.json> seed[,seed..]"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import analyze_rq1_pilot as an  # noqa: E402
import pilot_gate as pg  # noqa: E402

PROFILE_MS_PER_TOKEN = 22.2
EXPECT_MAX_BATCH = int(__import__("os").environ.get("EXPECT_MAX_BATCH", "0")) or None


def one(csv: Path, n: int, seed: int) -> dict:
    df = pd.read_csv(csv, low_memory=False)
    m = an.run_metrics(df, n)
    start, end = float(df["experiment_warmup_end_ms"].iloc[0]), float(df["experiment_measurement_end_ms"].iloc[0])
    win = df[(df["client_receive_ms"] >= start) & (df["client_receive_ms"] <= end)]
    b = win.drop_duplicates("verification_batch_id")
    m["seed"] = seed
    m["batch_size_hist"] = {int(k): int(v) for k, v in b["verification_batch_size"].value_counts().sort_index().items()}
    m["batch_wall_ms"] = {"mean": float(b["verification_batch_wall_ms"].mean()), "p50": float(b["verification_batch_wall_ms"].median()),
                          "p90": float(b["verification_batch_wall_ms"].quantile(0.9)), "max": float(b["verification_batch_wall_ms"].max())}
    spec = win[win["gamma"] > 0]
    m["draft"] = {
        "virtual_ms_per_token": float((spec["draft_latency_ms"] - 10.0).sum() / spec["gamma"].sum()),   # fixed 10 ms removed
        "physical_gpu_ms_per_token": float(spec["draft_measured_latency_ms"].sum() / spec["gamma"].sum()),
        "physical_wall_ms_per_token": float(spec["draft_measured_wall_latency_ms"].sum() / spec["gamma"].sum()),
        "profile_ms_per_token": PROFILE_MS_PER_TOKEN,
    }
    m["draft"]["physical_gpu_rel_dev_vs_profile"] = m["draft"]["physical_gpu_ms_per_token"] / PROFILE_MS_PER_TOKEN - 1
    qw = (win["batch_start_ms"] - win["server_arrival_ms"])
    pc = qw.groupby(win["client_id"]).mean()
    m["queue_wait_ms"] = {"mean": float(qw.mean()), "p50": float(qw.median()), "p90": float(qw.quantile(0.9)), "max": float(qw.max()),
                          "per_client_mean_min": float(pc.min()), "per_client_mean_max": float(pc.max())}
    try:
        import numpy as _np
        x, y = b["verification_batch_processed_tokens"].to_numpy(float), b["verification_batch_wall_ms"].to_numpy(float)
        a1, a0 = _np.polyfit(x, y, 1)
        r = float(_np.corrcoef(x, y)[0, 1])
        m["wall_vs_processed_tokens"] = {"slope_ms_per_token": float(a1), "intercept_ms": float(a0), "r": r,
                                         "processed_tokens_mean": float(x.mean()), "processed_tokens_max": float(x.max()),
                                         "context_tokens_max_mean": float(b["verification_batch_context_tokens_max"].mean())}
    except Exception:
        m["wall_vs_processed_tokens"] = None
    # cohort view: consecutive batches that start back-to-back (start == previous finish) are one cohort split by the cap
    bb = b.sort_values("batch_start_ms")
    fin = (bb["batch_start_ms"] + bb["verification_batch_wall_ms"]).to_numpy()
    st = bb["batch_start_ms"].to_numpy()
    back = int(((st[1:] - fin[:-1]) < 1.0).sum()) if len(bb) > 1 else 0
    m["back_to_back_batches"] = back
    m["verifier_max_batch_size"] = int(win["verifier_max_batch_size"].iloc[0]) if "verifier_max_batch_size" in win.columns and len(win) else None
    m["rounds_per_client"] = win.groupby("client_id").size().reindex(range(n), fill_value=0).tolist()
    m["tokens_per_round_mean"] = float(win["committed_token_count"].mean())
    m["gamma_values"] = sorted(int(g) for g in df["gamma"].unique())
    last = df.sort_values("client_receive_ms").groupby("client_id").tail(1)
    m["lifecycle"] = {"max_client_virtual_end_s": float(last["client_receive_ms"].max()) / 1000, "min_client_virtual_end_s": float(last["client_receive_ms"].min()) / 1000,
                      "any_work_complete": bool(df["client_work_complete"].any()),
                      "samples_assigned": int(df["client_samples_assigned"].iloc[0]),
                      "mean_tokens_per_finished_sample": float(df[df["sample_finished"]].groupby(["client_id", "sample_id"])["generated_for_sample"].max().mean()) if df["sample_finished"].any() else None,
                      "eos_finished_samples": int(df["reached_eos"].sum())}
    m["wall_clock_s"] = float(df["wall_clock_time_ms"].max()) / 1000
    m["virtual_max_s"] = float(df["client_receive_ms"].max()) / 1000
    chk = json.loads((Path(str(csv).replace(".csv", "_artifacts")) / "causality_check.json").read_text())
    m["causality"] = {"n_events": chk["n_events"], "n_violations": chk["n_violations"], "causal_virtual_time": chk["causal_virtual_time"]}
    ok, why = pg.gate(csv, n, Path(str(csv).replace(".csv", "_artifacts")) / "causality_check.json", EXPECT_MAX_BATCH)
    m["gate_ok"], m["gate_reasons"] = ok, why
    m["batches_in_window"] = int(len(b))
    m["requests_in_window"] = int(len(win))
    m["runtime_fallback_events"] = None
    return m


if __name__ == "__main__":
    root, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]
    seeds = [int(s) for s in sys.argv[4].split(",")]
    runs = []
    for seed in seeds:
        for n in (1, 2, 4, 8, 12, 16, 24, 32):
            p = pg.csv_path(root, tag, n, seed)
            if p.exists():
                runs.append(one(p, n, seed))
    by = {}
    for r in runs:
        by.setdefault(r["N"], []).append(r)
    res = {"runs": runs, "verdict": an.rq1_verdict(by)}
    Path(out).write_text(json.dumps(res, indent=1, default=str))
    for r in runs:
        print(r["N"], "x_min %.3f mean %.3f Y %.2f Nxmin %.2f fair %.2f util %.2f bs %.2f/%d wall %.0f r/v %.2f ledger %d gate %s rounds_min %d draft_gpu_dev %+.1f%% eos %d" % (
            r["x_min"], r["x_mean"], r["Y"], r["N_x_min"], r["x_min_over_mean"], r["verifier_utilisation"], r["batch_size_mean"], r["batch_size_max"],
            r["batch_wall_ms"]["mean"], r["real_over_virtual"], r["causality"]["n_violations"], r["gate_ok"], r["rounds_per_client_min"],
            100 * r["draft"]["physical_gpu_rel_dev_vs_profile"], r["lifecycle"]["eos_finished_samples"]))
    print(res["verdict"]["verdict"])
