"""C3.1 Task D: distribution of paired treatment-minus-baseline changes across independent populations (population = sampling unit).
usage: robustness_analysis.py <robust.json> <summary.json>"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

T975 = {19: 2.093, 9: 2.262}


def dist(d):
    d = np.asarray(d, float)
    n = len(d)
    if n == 0:
        return None
    half = T975.get(n - 1, 1.96) * d.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": float(d.mean()), "sd": float(d.std(ddof=1)) if n > 1 else 0.0, "ci95_mean": [float(d.mean() - half), float(d.mean() + half)],
            "p10": float(np.quantile(d, 0.1)), "p50": float(np.quantile(d, 0.5)), "p90": float(np.quantile(d, 0.9)), "frac_gt0": float((d > 0).mean()), "frac_lt0": float((d < 0).mean())}


def main(src, dst):
    data = json.loads(Path(src).read_text())["robust"]
    cells = defaultdict(lambda: defaultdict(dict))
    for r in data:
        t = r["task"]
        key = (r["tag"], t["n"], t["cap"])
        cells[key][t["pop_seed"]][t["arm"]] = r
    out = {"label": "Paired changes across independent populations (population = sampling unit, one paired run each; CPU model, NOT GPU). T = deficit-first (density tie-break), "
                    "A = deficit-first with arrival tie-break (ablation), B = work-conserving FCFS.", "cells": {}, "integrity": {}}
    viol = defaultdict(int)
    for r in data:
        viol["runs"] += 1
        viol["ledger"] += int(r["ledger_violations"] or 0)
        viol["invalid"] += int(bool(r["invalid"]))
        viol["empty_batches"] += r["empty_batches"]
        viol["cap_violations"] += r["cap_violations"]
        viol["token_budget_violations"] += r["token_budget_violations"]
        viol["skip_bound_violations"] += int(r["skips_before_service_max"] > r["task"].get("max_skips", 2) + -(-(r["task"]["n"] - 1) // r["task"]["cap"]))
    out["integrity"] = dict(viol)
    for (tag, n, cap), pops in sorted(cells.items()):
        rows = {"T": defaultdict(list), "A": defaultdict(list), "binding": defaultdict(list), "B": defaultdict(list)}
        for ps, arms in sorted(pops.items()):
            if "fix4+PF" not in arms:
                continue
            B = arms["fix4+PF"]
            for lab, arm in (("T", "fix4+DSz"), ("A", "fix4+DSa")):
                if arm not in arms:
                    continue
                T = arms[arm]
                for k in ("x_min", "Y", "jain", "slo_deficit_sum", "frac_clients_below_slo", "skipped_service_fraction", "queue_wait_ms_max", "x_p10"):
                    rows[lab][k].append(T[k] - B[k])
                rows[lab]["x_min_rel"].append((T["x_min"] - B["x_min"]) / B["x_min"])
                rows[lab]["Y_rel"].append((T["Y"] - B["Y"]) / B["Y"])
            for k in ("x_min", "Y", "jain", "slo_deficit_sum", "x_min_over_mean"):
                if k in B:
                    rows["B"][k].append(B[k])
            rows["B"]["x_min_over_mean"].append(B["x_min"] / B["x_mean"])
            bb = B.get("binding_batches")
            if bb:
                for k, v in bb.items():
                    rows["binding"][k].append(v)
        out["cells"][f"{tag}|N={n}|B_max={cap}"] = {
            "baseline_levels": {k: float(np.mean(v)) for k, v in rows["B"].items()},
            "T_minus_B": {k: dist(v) for k, v in rows["T"].items()},
            "A_minus_B": {k: dist(v) for k, v in rows["A"].items()},
            "B_binding_batches_replay": {k: float(np.mean(v)) for k, v in rows["binding"].items()}}
    Path(dst).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["integrity"]))
    for key, c in out["cells"].items():
        t, a, b = c["T_minus_B"], c["A_minus_B"], c["baseline_levels"]
        bd = c["B_binding_batches_replay"]
        print(f"{key:44s} Bx_min {b['x_min']:.2f} (x_min/mean {b['x_min_over_mean']:.3f}) | T-B x_min {t['x_min']['mean']:+.3f} [{t['x_min']['ci95_mean'][0]:+.3f},{t['x_min']['ci95_mean'][1]:+.3f}] "
              f"rel {100 * t['x_min_rel']['mean']:+.1f}% >0:{t['x_min']['frac_gt0']:.2f} | Y {100 * t['Y_rel']['mean']:+.2f}% | jain {t['jain']['mean']:+.4f} | slo {t['slo_deficit_sum']['mean']:+.2f} | "
              f"A-B x_min {(a.get('x_min') or {'mean': float('nan')})['mean']:+.3f} | bind {bd.get('fraction_of_batches', 0):.2f} T!=B {bd.get('pair_T_vs_B', 0):.2f} T!=T0(deficit effect) {bd.get('pair_T_vs_T0_(deficit_effect)', 0):.2f} T!=A {bd.get('pair_T_vs_A', 0):.2f} "
              f"A!=B {bd.get('pair_A_vs_B', 0):.2f} swapped {bd.get('mean_swapped_T_vs_B', 0):.2f} posdef {bd.get('frac_any_positive_deficit', 0):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
