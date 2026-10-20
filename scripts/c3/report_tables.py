"""C3: build the compact committed summary (results/c3/validation_summary.json) and print readable tables from the raw validation output.
usage: report_tables.py <validation_all.json> <summary_out.json>"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_validation import T975, agg, cell_key, paired  # noqa: E402

KEYS = ("x_min", "Y", "jain", "queue_wait_ms_max", "x_p10")


def group(rows):
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ck, arm = cell_key(r)
        cells[ck][arm].append(r)
    return cells


def fmt(p):
    if p is None:
        return "n/a"
    return f"{p['mean']:+.3f} [{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}] {p['b_gt_a']}/{p['n']}"


def compare(rows, pairs):
    out = []
    for ck, arms in group(rows).items():
        for ref, other in pairs:
            if ref in arms and other in arms:
                out.append({"cell": json.loads(ck), "ref": ref, "arm": other, **{k: paired(arms[ref], arms[other], k) for k in KEYS}})
    return out


def main(src, dst):
    data = json.loads(Path(src).read_text())
    summ = {"label": data["label"], "n_runs": {k: len(v) for k, v in data.items() if isinstance(v, list)}}
    allrows = [r for k, v in data.items() if isinstance(v, list) for r in v]
    summ["integrity_totals_over_all_runs"] = {
        "runs": len(allrows), "ledger_violations_total": int(sum(r["ledger_violations"] or 0 for r in allrows)),
        "invalid_runs": int(sum(bool(r["invalid"]) for r in allrows)), "empty_batches_total": int(sum(r["empty_batches"] for r in allrows)),
        "cap_violations_total": int(sum(r["cap_violations"] for r in allrows)),
        "token_budget_violations_total_c3_and_PF_arms": int(sum(r["token_budget_violations"] for r in allrows if "legacy" not in r["task"]["arm"])),
        "max_skips_before_service_seen": int(max(r["skips_before_service_max"] for r in allrows)),
        "forced_service_fraction_max": float(max((r["forced_service_fraction"] or 0.0) for r in allrows))}
    cmp_sched = []
    for g in (2, 4):
        cmp_sched += compare(data["sched"], [(f"fix{g}+legacy", f"fix{g}+PF"), (f"fix{g}+PF", f"fix{g}+DS"), (f"fix{g}+PF", f"fix{g}+DSz")])
    summ["sched_paired_differences"] = cmp_sched
    summ["sched_cells"] = [{"cell": json.loads(ck), "arms": {a: agg(rows) for a, rows in arms.items()}} for ck, arms in group(data["sched"]).items()]
    summ["depth_paired_differences"] = compare(data["depth"], [("fix4+PF", "depth+PF"), ("depth+PF", "depthNP+PF"), ("fix2+PF", "depth+PF"), ("fix2+PF", "fix4+PF")])
    summ["depth_cells"] = [{"cell": json.loads(ck), "arms": {a: agg(rows) for a, rows in arms.items()}} for ck, arms in group(data["depth"]).items()]
    summ["combined_paired_differences"] = compare(data["combined"], [("fix4+PF", "fix4+DSz"), ("fix4+PF", "depth+PF"), ("fix4+PF", "depth+DSz"), ("fix4+DSz", "depth+DSz")])
    summ["combined_cells"] = [{"cell": json.loads(ck), "arms": {a: agg(rows) for a, rows in arms.items()}} for ck, arms in group(data["combined"]).items()]
    sens = defaultdict(list)
    for r in data["sens"]:
        sens[r["tag"]].append(r)
    summ["sensitivity"] = {tag: {"arm": rows[0]["task"]["arm"], **agg(rows)} for tag, rows in sorted(sens.items())}
    adv = defaultdict(lambda: defaultdict(list))
    for r in data["adversarial"]:
        adv[r["tag"]][r["task"]["arm"]].append(r)
    summ["adversarial"] = {tag: {arm: agg(rows) for arm, rows in arms.items()} for tag, arms in sorted(adv.items())}
    # ---- open-loop replay vs closed loop for the depth rule
    ol = defaultdict(dict)
    for r in data.get("openloop", []):
        kind, rest = r["tag"].split("_", 2)[1], r["tag"].split("_", 2)[2]
        ol[rest].setdefault(kind, []).append(r)
    summ["open_loop_vs_closed_loop_depth"] = {}
    for rest, d in sorted(ol.items()):
        row = {}
        if "shadow" in d:
            row["open_loop_frac_would_differ_from_gamma4"] = float(np.mean([r["shadow_depth"]["frac_would_differ_from_acted"] for r in d["shadow"]]))
            row["fixed_gamma4_x_min"] = float(np.mean([r["x_min"] for r in d["shadow"]]))
            row["fixed_gamma4_Y"] = float(np.mean([r["Y"] for r in d["shadow"]]))
        if "closed" in d:
            row["closed_loop_frac_not_gamma4"] = float(np.mean([r["c3_depth_log"]["frac_not_gamma_max"] for r in d["closed"]]))
            row["closed_loop_x_min"] = float(np.mean([r["x_min"] for r in d["closed"]]))
            row["closed_loop_Y"] = float(np.mean([r["Y"] for r in d["closed"]]))
        summ["open_loop_vs_closed_loop_depth"][rest] = row
    # ---- interactivity capacity N_c(x) (first crossing, conservative) per arm, population, B_max, run-seed
    cap = defaultdict(lambda: defaultdict(dict))
    for r in data.get("capacity", []):
        t = r["task"]
        cap[(t["pop"], t["cap"], t["arm"])][t["seed"]][t["n"]] = r["x_min"]
    floors = (1.0, 1.5, 2.0, 2.5, 3.0)

    def n_c(curve, x):
        best = 0
        for n in sorted(curve):
            if curve[n] >= x:
                best = n
            else:
                break
        return best

    summ["capacity"] = {}
    for (pop, b, arm), per_seed in sorted(cap.items()):
        for x in floors:
            vals = {sd: n_c(c, x) for sd, c in per_seed.items()}
            summ["capacity"].setdefault(f"{pop}|B={b}|x>={x}", {})[arm] = {"N_c_by_seed": vals, "N_c_mean": float(np.mean(list(vals.values())))}
    # paired differences vs fix4+PF
    summ["capacity_paired_vs_fix4+PF"] = {}
    for key, arms in summ["capacity"].items():
        ref = arms.get("fix4+PF")
        if not ref:
            continue
        summ["capacity_paired_vs_fix4+PF"][key] = {}
        for a_, v in arms.items():
            if a_ == "fix4+PF":
                continue
            d = np.array([v["N_c_by_seed"][sd] - ref["N_c_by_seed"][sd] for sd in sorted(ref["N_c_by_seed"]) if sd in v["N_c_by_seed"]], float)
            n_ = len(d)
            half = T975.get(n_ - 1, 1.96) * (d.std(ddof=1) / np.sqrt(n_) if n_ > 1 else 0.0)
            summ["capacity_paired_vs_fix4+PF"][key][a_] = {"mean_diff": float(d.mean()), "ci95": [float(d.mean() - half), float(d.mean() + half)], "n": n_,
                                                          "arm_gt_ref": int((d > 0).sum()), "arm_lt_ref": int((d < 0).sum())}
    # skip-bound check: a request eligible-but-unselected for k batches is served within max_skips + ceil((N-1)/B) batches (starvation guard)
    viol = 0
    worst = 0
    for r in allrows:
        if "legacy" in r["task"]["arm"]:
            continue
        n_, b_ = r["task"]["n"], r["task"].get("cap", 16)
        bound = r["task"].get("max_skips", 2) + -(-(n_ - 1) // b_)
        worst = max(worst, r["skips_before_service_max"] - bound)
        viol += r["skips_before_service_max"] > bound
    summ["integrity_totals_over_all_runs"]["skip_bound_violations"] = int(viol)
    summ["integrity_totals_over_all_runs"]["skip_bound_worst_margin"] = int(worst)
    Path(dst).write_text(json.dumps(summ, indent=1, default=float))

    # ---- readable tables
    print("integrity", json.dumps(summ["integrity_totals_over_all_runs"]))
    for title, rows in (("PF vs legacy / DS vs PF / DSz vs PF (gamma 2 and 4)", cmp_sched),):
        print("==", title)
        for r in rows:
            c = r["cell"]
            print(f"{c['pop']:13s} N={c['n']:2d} B={c['cap']:2d} {r['ref']:>10s} -> {r['arm']:10s} x_min {fmt(r['x_min'])} | Y {fmt(r['Y'])} | jain {fmt(r['jain'])}")
    print("== depth")
    for r in summ["depth_paired_differences"]:
        c = r["cell"]
        print(f"{c['pop']:13s} N={c['n']:2d} B={c['cap']:2d} {r['ref']:>10s} -> {r['arm']:10s} x_min {fmt(r['x_min'])} | Y {fmt(r['Y'])}")
    print("== combined")
    for r in summ["combined_paired_differences"]:
        c = r["cell"]
        print(f"{c['pop']:13s} N={c['n']:2d} B={c['cap']:2d} {r['ref']:>10s} -> {r['arm']:10s} x_min {fmt(r['x_min'])} | Y {fmt(r['Y'])} | jain {fmt(r['jain'])}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
