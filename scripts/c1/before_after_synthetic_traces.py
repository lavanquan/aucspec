"""Before/after synthetic traces for the C1 repairs (CPU only).  Writes results/c1_20261020/before_after_traces.json."""
import asyncio
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "src"))
sys.path.insert(0, str(ROOT / "legacy" / "tests"))

import pandas as pd  # noqa: E402

from causality_harness import Spec, reference_batches, run_scenario  # noqa: E402
from edge_specsim import capacity as legacy  # noqa: E402
from edge_specsim.capacity import NStarResult, SearchDecision  # noqa: E402
from edge_specsim.capacity_certified import (  # noqa: E402
    FEASIBLE, INFEASIBLE, UNCERTAIN, area_from_boundaries, certified_n_search, summarize_probes)
from edge_specsim.population_seeds import MasterPopulation  # noqa: E402
from edge_specsim.workload_lifecycle import assess_from_dataframe  # noqa: E402

OUT = {}

# ---- (a) exhaustion: the recorded B6 smoke ---------------------------------------------------------------------
df = pd.read_csv(ROOT / "legacy" / "tests" / "data_b6_smoke_compact.csv")
win = df[df["in_measurement_window"].astype(bool)]
legacy_flag = []
for cid, g in win.groupby("client_id"):
    span = (g["client_receive_ms"].max() - g["client_receive_ms"].min()) / 1000.0
    if span < 0.85 * 20.0:
        legacy_flag.append({"client": int(cid), "in_window_span_s": round(span, 2)})
a = assess_from_dataframe(df, admitted_client_ids=range(8), measurement_start_ms=5000.0, measurement_end_ms=25000.0,
                          expected_samples_per_client=6)
OUT["exhaustion_b6_smoke"] = {
    "before_span_heuristic_flags_exhausted": legacy_flag,
    "before_verdict": "INVALID (5+ of 8 clients 'exhausted')",
    "after_lifecycle": [{"client": c.client_id, "status": c.status, "last_event_virtual_s": round(c.last_event_ms / 1000, 1),
                         "in_flight_at_end": c.in_flight_at_end} for c in a.clients],
    "after_verdict": "INVALID" if a.invalid else "valid (no client exhausted)",
}

# ---- (b) population identity ------------------------------------------------------------------------------------
CFG = {"rtt_ms_range": [15.0, 100.0], "uplink_mbps_range": [5.0, 50.0], "draft_tokens_per_second_range": [20.0, 60.0]}


def legacy_client0(seed):
    random.seed(seed)
    tps = random.uniform(20.0, 60.0)
    random.uniform(5.0, 20.0)            # draft_fixed_latency
    rtt = random.uniform(*CFG["rtt_ms_range"])
    return {"seed": seed, "client0_rtt_ms": round(rtt, 2), "client0_draft_tps": round(tps, 2)}


pop = {s: MasterPopulation(CFG, 7, 16) for s in (1, 2, 3)}      # run-seed 1,2,3 -> the same master (seed not an input)
OUT["population"] = {
    "before_client0_per_run_seed": [legacy_client0(s) for s in (1, 2, 3, 4, 5)],
    "after_client0_per_run_seed": [{"run_seed": s, "client0_rtt_ms": round(p.clients[0].rtt_ms, 2),
                                    "client0_draft_tps": round(p.clients[0].draft_tokens_per_second_base, 2),
                                    "fingerprint_N8": p.fingerprint(8)} for s, p in pop.items()],
    "after_prefix_fingerprints_equal_across_N": MasterPopulation(CFG, 7, 8).fingerprint(8) == MasterPopulation(CFG, 7, 16).fingerprint(8),
}

# ---- (c) causality ---------------------------------------------------------------------------------------------------
def rand_specs(rng):
    n = rng.randint(3, 6)
    return [Spec(i, rng.choice([10.0, 20.0, 40.0]), [rng.choice([5.0, 30.0, 80.0, 200.0]) + rng.random() for _ in range(3)],
                 [rng.random() * 0.05 for _ in range(3)]) for i in range(n)]


rng = random.Random(11)
rows = []
for t in range(30):
    specs = rand_specs(rng)
    ref = reference_batches(specs)
    r = {}
    for mode in ("legacy_real_order", "causal"):
        res = asyncio.run(run_scenario(specs, causal=(mode == "causal")))
        r[mode] = {
            "d1_violations": sum(v.startswith("D1") for v in res["violations"]),
            "d2_violations": sum(v.startswith("D2") for v in res["violations"]),
            "batches_equal_to_virtual_reference": [b["members"] for b in res["batches"]] == [b["members"] for b in ref],
        }
    rows.append(r)
OUT["causality"] = {
    mode: {
        "scenarios": len(rows),
        "scenarios_with_d1_violation": sum(r[mode]["d1_violations"] > 0 for r in rows),
        "scenarios_with_d2_violation": sum(r[mode]["d2_violations"] > 0 for r in rows),
        "scenarios_matching_virtual_reference": sum(r[mode]["batches_equal_to_virtual_reference"] for r in rows),
        "total_d1_violations": sum(r[mode]["d1_violations"] for r in rows),
    } for mode in ("legacy_real_order", "causal")
}

# targeted D2 scenario: client 1 decides at virtual t=5 but its real evaluation follows a batch finishing at ~100 ms
d2_specs = [Spec(0, 20.0, [10.0, 10.0], [0.0]), Spec(1, 20.0, [10.0, 10.0], [0.0])]
d2 = {}
for mode in ("legacy_real_order", "causal"):
    res = asyncio.run(run_scenario(d2_specs, causal=(mode == "causal"), pre_decision_real_delay_s={1: 0.1}))
    d2[mode] = {"d2_violations": [v for v in res["violations"] if v.startswith("D2")],
                "max_visible_minus_decision_ts": max(d["visible_ts"] - d["t"] for d in res["decisions"])}
OUT["causality_d2_targeted"] = d2

# ---- (d) boundary logic ---------------------------------------------------------------------------------------------
tab = {1: FEASIBLE, 2: FEASIBLE, 3: FEASIBLE, 4: UNCERTAIN, 5: UNCERTAIN, 6: UNCERTAIN}
dec = lambda n: tab.get(n, INFEASIBLE)
old = legacy.monotone_scale_search_final(lambda n: SearchDecision(n=n, status=dec(n), min_lcb=0.0, min_ucb=0.0, seeds_used=5),
                                         hi_cap=16, x_requirement=3.0)
new = certified_n_search(dec, hi_cap=16, x_requirement=3.0)
OUT["boundary_uncertain_midpoint"] = {
    "truth": "feasible up to N=6 is consistent with the data (N=4..6 gray); first certified infeasible N=7",
    "before": {"lower": old.lower_bound_n, "upper_bound_n": old.upper_bound_n, "claimed_exact": old.exact,
               "note": "upper bound = the UNCERTAIN N=4 (would exclude N=5,6 which may be feasible)"},
    "after": {"lower": new.lower_bound_n, "upper_conditional_monotone": new.upper_bound_monotone, "status": new.status,
              "gray": new.gray_ns},
}
nonmono = summarize_probes(3.0, {1: FEASIBLE, 2: FEASIBLE, 3: FEASIBLE, 4: INFEASIBLE, 6: FEASIBLE, 9: INFEASIBLE}, 16)
OUT["boundary_nonmonotone_counterexample"] = {
    "lower_bound_unconditional": nonmono.lower_bound_n, "upper_bound_monotone": nonmono.upper_bound_monotone,
    "violations": nonmono.monotone_violations, "status": nonmono.status,
}

n_of_x = lambda x: math.floor(24.0 / x + 1e-12)
xs = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
res = [summarize_probes(x, {n: (FEASIBLE if n <= n_of_x(x) else INFEASIBLE) for n in range(1, n_of_x(x) + 3)}, 64) for x in xs]
steps = 400000
h = 11.0 / steps
truth = sum(min(n_of_x(1.0 + (i + 0.5) * h), 24) * h for i in range(steps))
a_new = area_from_boundaries(res, x_L=1.0, x_U=12.0, n_ref=24)
a_old = legacy.area_from_nstar([NStarResult(x_requirement=x, last_confirmed_feasible_n=n_of_x(x),
                                            first_confirmed_infeasible_n=n_of_x(x) + 1, lower_bound_n=n_of_x(x),
                                            upper_bound_n=n_of_x(x) + 1, exact=True, hit_search_cap=False) for x in xs],
                               x_L=1.0, x_U=12.0, n_ref=24)
OUT["ica_staircase"] = {
    "true_area": truth,
    "before": {"area_lower": a_old["area_lower"], "area_upper": a_old["area_upper"],
               "lower_bound_valid": a_old["area_lower"] <= truth + 1e-3},
    "after": {"area_lower": a_new["area_lower"], "area_upper": a_new["area_upper"],
              "lower_bound_valid": a_new["area_lower"] <= truth + 1e-3,
              "upper_bound_valid": a_new["area_upper"] >= truth - 1e-3,
              "discretisation_gap": a_new["discretisation_gap"]},
}

path = ROOT / "results" / "c1_20261020" / "before_after_traces.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(OUT, indent=1, default=str))
print(json.dumps(OUT, indent=1, default=str))
