"""C3: compact summaries of mechanism_validation.py output (means over run-seeds, paired differences vs a reference arm with t-intervals).
usage: summarize_validation.py <validation.json> <summary.json> [print]"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}      # two-sided 95 % t, df = n-1 (n = number of paired run-seeds, <= 6)
KEYS = ("x_min", "x_mean", "Y", "jain", "util", "batch_size_mean", "queue_wait_ms_mean", "queue_wait_ms_max", "wait_spread_ms", "gamma_mean", "z_mean", "z_max",
        "round_ms_mean", "ledger_violations", "empty_batches", "skips_before_service_max", "cap_violations", "token_budget_violations", "max_processed_tokens_seen",
        "oversize_singletons", "forced_service_fraction", "cap_binding_fraction", "x_client0", "x_p10")


def cell_key(r):
    t = r["task"]
    extra = {k: v for k, v in t.items() if k not in ("arm", "seed")}
    return json.dumps(extra, sort_keys=True, default=str), t["arm"]


def agg(rows):
    out = {}
    for k in KEYS:
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            out[k] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
    out["n_seeds"] = len(rows)
    if rows and rows[0].get("c3_depth_log"):
        out["frac_not_gamma_max"] = float(np.mean([r["c3_depth_log"]["frac_not_gamma_max"] for r in rows]))
    if rows and rows[0].get("open_loop_selector"):
        out["open_loop_selector"] = {k: float(np.mean([r["open_loop_selector"][k] for r in rows])) for k in rows[0]["open_loop_selector"]}
    return out


def paired(rows_a, rows_b, key):
    """b - a, paired by run-seed (same acceptance draws).  Returns mean, 95% t-interval, k of n seeds with b > a."""
    da = {r["task"]["seed"]: r[key] for r in rows_a if r.get(key) is not None}
    db = {r["task"]["seed"]: r[key] for r in rows_b if r.get(key) is not None}
    seeds = sorted(set(da) & set(db))
    d = np.array([db[s] - da[s] for s in seeds], float)
    n = len(d)
    if n < 2:
        return None
    sd = float(d.std(ddof=1))
    half = T975.get(n - 1, 1.96) * sd / math.sqrt(n)
    return {"mean": float(d.mean()), "ci95": [float(d.mean() - half), float(d.mean() + half)], "n": n, "b_gt_a": int((d > 0).sum())}


def summarize(path):
    data = json.loads(Path(path).read_text())
    out = {"label": data.get("label", ""), "experiments": {}}
    for name in ("sched", "depth", "combined", "adversarial"):
        if name not in data:
            continue
        cells = defaultdict(lambda: defaultdict(list))
        for r in data[name]:
            ck, arm = cell_key(r)
            cells[ck][arm].append(r)
        out["experiments"][name] = []
        for ck, arms in cells.items():
            rec = {"cell": json.loads(ck), "arms": {a: agg(rows) for a, rows in arms.items()}}
            out["experiments"][name].append(rec)
    return out, data


def pair_table(data, name, ref, others, keys=("x_min", "Y", "jain", "queue_wait_ms_max")):
    cells = defaultdict(lambda: defaultdict(list))
    for r in data[name]:
        ck, arm = cell_key(r)
        cells[ck][arm].append(r)
    rows = []
    for ck, arms in cells.items():
        if ref not in arms:
            continue
        for o in others:
            if o in arms:
                rows.append({"cell": json.loads(ck), "ref": ref, "arm": o, **{k: paired(arms[ref], arms[o], k) for k in keys}})
    return rows


if __name__ == "__main__":
    summ, data = summarize(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(summ, indent=1, default=float))
    if len(sys.argv) > 3:
        for name, cells in summ["experiments"].items():
            print("==", name)
            for c in cells:
                print(c["cell"])
                for a, m in c["arms"].items():
                    print("   ", a, {k: round(m[k]["mean"], 3) for k in ("x_min", "Y", "jain", "queue_wait_ms_max", "batch_size_mean", "z_max") if k in m},
                          "viol", {k: m[k]["max"] for k in ("ledger_violations", "empty_batches", "cap_violations", "token_budget_violations") if k in m})
