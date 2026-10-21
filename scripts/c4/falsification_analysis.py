"""C4 Task A: paired comparison of four ranks on the same work-conserving path.  FCFS (PF) is the reference; AGE, ZON, ZEFF(=DSa) are compared with it.
usage: falsification_analysis.py <c4.json> <summary.json>"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RANKS = {"PF": "FCFS", "AGE": "AGE", "ZON": "ZONLY", "DSa": "ZEFF"}
METRICS = ("x_min", "Y", "n_clients_below_slo", "slo_deficit_sum", "queue_wait_ms_mean", "queue_wait_ms_max", "skipped_service_fraction", "jain", "x_p10")


def dist(d):
    d = np.asarray(d, float)
    n = len(d)
    t = {19: 2.093}.get(n - 1, 1.96)
    h = t * d.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return {"mean": float(d.mean()), "ci95": [float(d.mean() - h), float(d.mean() + h)], "p10": float(np.quantile(d, 0.1)), "p90": float(np.quantile(d, 0.9)),
            "frac_gt0": float((d > 1e-12).mean()), "frac_lt0": float((d < -1e-12).mean()), "n": n}


def main(src, dst):
    data = json.loads(Path(src).read_text())["c4"]
    cells = defaultdict(lambda: defaultdict(dict))
    for r in data:
        t = r["task"]
        g, rank = t["arm"].split("+")
        cells[(r["tag"], t["n"], t["cap"])][t["pop_seed"]][rank] = r
    out = {"label": "C4 scheduler falsification (CPU model). Reference = FCFS on the identical work-conserving path; population = sampling unit (20 per cell); gamma fixed.", "cells": {}}
    viol = defaultdict(int)
    for r in data:
        viol["runs"] += 1
        viol["ledger"] += int(r["ledger_violations"] or 0)
        viol["invalid"] += int(bool(r["invalid"]))
        viol["empty_batches"] += r["empty_batches"]
        viol["cap_violations"] += r["cap_violations"]
        viol["token_budget_violations"] += r["token_budget_violations"]
    out["integrity"] = dict(viol)
    for key, pops in sorted(cells.items()):
        tag, n, cap = key
        c = {"baseline": {}, "delta_vs_FCFS": {}, "attribution_on_FCFS_eligible_sets": {}, "all_clients_meet_SLO_populations": {}}
        base = {m: [] for m in METRICS}
        for ps, arms in sorted(pops.items()):
            if "PF" in arms:
                for m in METRICS:
                    base[m].append(arms["PF"][m])
        c["baseline"] = {m: float(np.mean(v)) for m, v in base.items()}
        for rk in ("AGE", "ZON", "DSa"):
            dd = {m: [] for m in METRICS}
            rel = []
            relY = []
            for ps, arms in sorted(pops.items()):
                if "PF" in arms and rk in arms:
                    for m in METRICS:
                        dd[m].append(arms[rk][m] - arms["PF"][m])
                    rel.append((arms[rk]["x_min"] - arms["PF"]["x_min"]) / arms["PF"]["x_min"])
                    relY.append((arms[rk]["Y"] - arms["PF"]["Y"]) / arms["PF"]["Y"])
            c["delta_vs_FCFS"][RANKS[rk]] = {**{m: dist(v) for m, v in dd.items()}, "x_min_rel": dist(rel), "Y_rel": dist(relY)}
        for rk in ("PF", "AGE", "ZON", "DSa"):
            c["all_clients_meet_SLO_populations"][RANKS[rk]] = float(np.mean([arms[rk]["n_clients_below_slo"] == 0 for arms in pops.values() if rk in arms]))
        att = [arms["PF"]["attribution"] for arms in pops.values() if "PF" in arms and arms["PF"].get("attribution")]
        if att:
            keys = ["binding", "zeff_ne_fcfs_frac", "age_ne_fcfs_frac", "zonly_ne_fcfs_frac", "zeff_ne_age_frac", "zeff_ne_zonly_frac", "frac_batches_any_positive_stamped_Z"]
            c["attribution_on_FCFS_eligible_sets"] = {k: float(np.mean([a[k] for a in att])) for k in keys}
            for k in ("among_changed_explained_by_age", "among_changed_explained_by_zonly", "among_changed_explained_by_neither"):
                v = [a[k] for a in att if a.get(k) is not None]
                c["attribution_on_FCFS_eligible_sets"][k] = float(np.mean(v)) if v else None
        out["cells"][f"{tag}|N={n}|B_max={cap}"] = c
    Path(dst).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["integrity"]))
    for k, c in out["cells"].items():
        b = c["baseline"]
        line = f"{k:40s} FCFS x_min {b['x_min']:.2f} below-SLO {b['n_clients_below_slo']:.1f}"
        for rk in ("AGE", "ZONLY", "ZEFF"):
            d = c["delta_vs_FCFS"][rk]
            line += f" | {rk}: dx {d['x_min']['mean']:+.3f} ({100 * d['x_min_rel']['mean']:+.1f}%) dBelow {d['n_clients_below_slo']['mean']:+.2f} dY {100 * d['Y_rel']['mean']:+.2f}%"
        a = c["attribution_on_FCFS_eligible_sets"]
        if a:
            line += f" || Zeff!=FCFS {a['zeff_ne_fcfs_frac']:.2f} age!=FCFS {a['age_ne_fcfs_frac']:.2f} zon!=FCFS {a['zonly_ne_fcfs_frac']:.2f} expl.age {a.get('among_changed_explained_by_age')} expl.Z {a.get('among_changed_explained_by_zonly')}"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
