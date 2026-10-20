"""CPU diagnostics of verifier batch-wall tails from raw pilot traces (C1.2 stage C).
usage: analyze_slow_batches.py <scratch_root> <tag> <seed> <out.json> [N ...]
Per N: distribution of per-batch wall time, slow batches (> max(600 ms, 2.0 x median)), their clustering in virtual time /
batch order / warm-up boundary, and whether they differ in context composition (processed tokens, context sum/max, batch size).
Only descriptive statistics; no causal claim."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import pilot_gate as pg  # noqa: E402


def per_batch(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, low_memory=False)
    cols = ["verification_batch_id", "verification_batch_wall_ms", "batch_start_ms", "verification_batch_size",
            "verification_batch_processed_tokens", "verification_batch_context_tokens_sum", "verification_batch_context_tokens_max",
            "experiment_warmup_end_ms", "experiment_measurement_end_ms"]
    b = df[cols].drop_duplicates("verification_batch_id").sort_values("verification_batch_id").reset_index(drop=True)
    return b


def analyze(b: pd.DataFrame) -> dict:
    w = b["verification_batch_wall_ms"].to_numpy()
    med = float(np.median(w))
    thr = max(600.0, 2.0 * med)
    slow = b["verification_batch_wall_ms"] > thr
    t = b["batch_start_ms"] / 1000.0
    ws, we = float(b["experiment_warmup_end_ms"].iloc[0]) / 1000, float(b["experiment_measurement_end_ms"].iloc[0]) / 1000
    out = {"n_batches": int(len(b)), "median_ms": med, "mean_ms": float(w.mean()), "p90_ms": float(np.quantile(w, 0.9)),
           "p99_ms": float(np.quantile(w, 0.99)), "max_ms": float(w.max()), "slow_threshold_ms": thr, "n_slow": int(slow.sum())}
    if slow.sum():
        sl = b[slow]
        ids = sl["verification_batch_id"].to_numpy()
        out["slow_in_warmup"] = int((sl["batch_start_ms"] / 1000 < ws).sum())
        out["slow_in_window"] = int(((sl["batch_start_ms"] / 1000 >= ws) & (sl["batch_start_ms"] / 1000 <= we)).sum())
        out["slow_after_window"] = int((sl["batch_start_ms"] / 1000 > we).sum())
        out["slow_virtual_time_s"] = [round(float(x), 1) for x in sl["batch_start_ms"] / 1000][:60]
        out["slow_batch_ids"] = [int(x) for x in ids][:60]
        # run-length clustering in batch order (consecutive slow batches)
        runs, cur = [], 1
        for a, c in zip(ids[:-1], ids[1:]):
            if c == a + 1:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        out["slow_run_lengths"] = runs
        out["lag1_autocorr_of_slow_flag"] = float(np.corrcoef(slow[:-1].astype(float), slow[1:].astype(float))[0, 1]) if len(b) > 3 else None
        for col in ("verification_batch_processed_tokens", "verification_batch_context_tokens_sum", "verification_batch_context_tokens_max", "verification_batch_size"):
            out[f"mean_{col}_slow_vs_fast"] = [float(sl[col].mean()), float(b[~slow][col].mean())]
        # relation of wall time to processed tokens among fast batches (linear fit) and residual of slow ones
        x, y = b["verification_batch_processed_tokens"].to_numpy(float), w
        k = ~slow.to_numpy()
        if k.sum() > 5:
            a1, a0 = np.polyfit(x[k], y[k], 1)
            out["fast_fit_ms_per_token_intercept"] = [float(a1), float(a0)]
            out["slow_over_fit_ratio_median"] = float(np.median(y[~k] / (a1 * x[~k] + a0)))
    # first-vs-later halves of the run
    half = len(b) // 2
    out["median_first_half_ms"] = float(np.median(w[:half]))
    out["median_second_half_ms"] = float(np.median(w[half:]))
    return out


if __name__ == "__main__":
    root, tag, seed, outp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    ns = [int(x) for x in sys.argv[5:]] or [1, 2, 4, 8, 12, 16]
    res = {}
    for n in ns:
        p = pg.csv_path(root, tag, n, seed)
        if p.exists():
            res[n] = analyze(per_batch(p))
    Path(outp).write_text(json.dumps(res, indent=1))
    for n, r in res.items():
        print(n, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, list)})
