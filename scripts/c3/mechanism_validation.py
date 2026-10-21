"""C3 Stage D: CPU mechanism validation on the real simulator / queue / batcher code (closed-loop) with a CPU physical-verifier stand-in.

usage: mechanism_validation.py <experiment> <out.json> [workers]       experiment in {sched, depth, combined, sens, adversarial, openloop, all}

* NOT a GPU result.  The verifier's service time is the CPU world's TRUTH (a + b*ctx + b_gamma*gamma, +-8 % noise), distinct from the cost model
  the controller assumes; acceptance outcomes are coupled across arms (same token position, same draw).
* Populations are STATIC and matched across arms; run-seeds only change the acceptance draws, service-time noise and arrival jitter.
* Settings were fixed before looking at outcomes; nothing here is tuned toward a win.  Negative outcomes are reported as such.
"""
import itertools
import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _init():
    for p in (ROOT / "legacy" / "src", ROOT / "legacy" / "tests", ROOT / "scripts" / "c1", ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def run_task(task: dict) -> dict:
    _init()
    import numpy as np

    import c3_scenarios as sc

    tag = task.pop("_tag", None)
    sim = sc.run(sc.task_to_spec(dict(task)))
    m = sc.metrics(sim)
    log = getattr(sim, "batch_log", [])
    multi = [r for r in log if r["n_eligible"] > 1]
    out = {"tag": tag, "task": task, **{k: v for k, v in m.items() if k not in ("rates_sorted",)}}
    out["x_min_rates"] = m["rates_sorted"][:3]
    out["empty_batches"] = int(sum(1 for r in log if not r["chosen"]))
    out["skips_before_service_max"] = int(max((max(r.get("skips_before_service", [0]) or [0]) for r in log), default=0))
    out["oversize_singletons"] = int(sum(1 for r in log if len(r["chosen"]) == 1 and r.get("chosen_processed_tokens", 0) > 4096))
    out["token_budget_violations"] = int(sum(1 for r in log if len(r["chosen"]) > 1 and r.get("chosen_processed_tokens", 0) > task.get("token_budget_processed", 4096)))
    out["cap_violations"] = int(sum(1 for r in log if len(r["chosen"]) > r["cap"]))
    # open-loop selector replay on THE LOGGED ELIGIBLE SETS of this run (same state, different selector)
    ol = {}
    for alt in ("fcfs", "c3_fcfs", "c3_deficit"):
        if multi and alt in multi[0]:
            ol[alt + "_vs_chosen_member_set_differs"] = float(np.mean([set(r[alt]) != set(r["chosen"]) for r in multi]))
    if ol:
        out["open_loop_selector"] = ol
        out["cap_binding_fraction"] = float(np.mean([r["n_eligible"] > r["cap"] for r in log])) if log else None
    # C3.1: membership differences on the batches where the cap BINDS (selection is non-trivial), measured on this run's own logged eligible sets
    binding = [r for r in log if r["n_eligible"] > r["cap"]]
    if binding and "T0_density_only" in binding[0]:
        out["binding_batches"] = {
            "n": len(binding), "fraction_of_batches": len(binding) / max(1, len(log)),
            "frac_any_positive_deficit": float(np.mean([any(v > 1e-9 for v in r["z_eff"].values()) for r in binding])),
            **{f"member_set_differs_from_chosen__{k}": float(np.mean([set(r[k]) != set(r["chosen"]) for r in binding]))
               for k in ("fcfs", "c3_fcfs", "c3_deficit", "T0_density_only", "T_deficit_arrival", "T_deficit_density")},
            "mean_members_swapped_vs_c3_fcfs": float(np.mean([len(set(r["c3_deficit"]) - set(r["c3_fcfs"])) for r in binding])),
            # pairwise on the SAME eligible set: B = c3_fcfs, T = deficit-first/density tie-break, A = deficit-first/arrival tie-break, T0 = density only (no deficit)
            "pair_T_vs_B": float(np.mean([set(r["T_deficit_density"]) != set(r["c3_fcfs"]) for r in binding])),
            "pair_T_vs_T0_(deficit_effect)": float(np.mean([set(r["T_deficit_density"]) != set(r["T0_density_only"]) for r in binding])),
            "pair_A_vs_B": float(np.mean([set(r["T_deficit_arrival"]) != set(r["c3_fcfs"]) for r in binding])),
            "pair_T_vs_A": float(np.mean([set(r["T_deficit_density"]) != set(r["T_deficit_arrival"]) for r in binding])),
            "pair_T0_vs_B": float(np.mean([set(r["T0_density_only"]) != set(r["c3_fcfs"]) for r in binding])),
            "mean_swapped_T_vs_B": float(np.mean([len(set(r["T_deficit_density"]) - set(r["c3_fcfs"])) for r in binding])),
            "mean_swapped_A_vs_B": float(np.mean([len(set(r["T_deficit_arrival"]) - set(r["c3_fcfs"])) for r in binding]))}
    ctrl = sim.controller
    if getattr(ctrl, "shadow_log", None):
        w = np.array([r["would"] for r in ctrl.shadow_log])
        out["shadow_depth"] = {"n": int(len(w)), "frac_would_differ_from_acted": float(np.mean(w != np.array([r["acted"] for r in ctrl.shadow_log]))),
                               "would_dist": {int(k): float(np.mean(w == k)) for k in sorted(set(w.tolist()))}}
    if hasattr(ctrl, "c3_log") and ctrl.c3_log:
        g = np.array([r["gamma"] for r in ctrl.c3_log])
        out["c3_depth_log"] = {"n": int(len(g)), "gamma_dist": {int(k): float(np.mean(g == k)) for k in sorted(set(g.tolist()))},
                               "frac_not_gamma_max": float(np.mean(g != 4)), "mean_u_hat_ms": float(np.mean([r["u_hat"] for r in ctrl.c3_log])),
                               "mean_goodput_tok_per_ms": float(np.mean([r["goodput"] for r in ctrl.c3_log]))}
    return out


def pool_map(tasks, workers):
    if workers <= 1:
        _init()
        return [run_task(dict(t)) for t in tasks]
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        return pool.map(run_task, [dict(t) for t in tasks], chunksize=4)


GRID = [(24, 16), (32, 16), (48, 16), (24, 8), (32, 8)]          # (N, B_max); B_max=16 is the measured memory limit, 8 a tighter physical limit
POPS = ("homog_sync", "hetero_sync", "hetero_async")
SEEDS = (1, 2, 3, 4, 5)


def tasks_sched():
    arms = [f"fix{g}+{s}" for g in (2, 4) for s in ("legacy", "PF", "DS", "DSz")]
    return [dict(arm=a, n=n, cap=b, pop=p, seed=s, _tag="sched") for p in POPS for (n, b) in GRID for a in arms for s in SEEDS]


def tasks_depth():
    arms = ["fix2+PF", "fix4+PF", "depth+PF", "depthNP+PF"]
    return [dict(arm=a, n=n, cap=b, pop=p, seed=s, _tag="depth") for p in POPS for (n, b) in GRID for a in arms for s in SEEDS]


def tasks_combined():
    arms = ["fix4+PF", "fix4+DSz", "depth+PF", "depth+DSz"]
    return [dict(arm=a, n=n, cap=b, pop=p, seed=s, _tag="combined") for p in POPS for (n, b) in GRID for a in arms for s in SEEDS]


def tasks_sens():
    t = []
    base = dict(n=24, cap=8, pop="hetero_async")
    # (1) truth per-candidate-token verifier cost x controller's assumed value (depth arms) - misspecified and well specified
    for truth in (0.0, 0.19, 1.9, 5.5, 19.0):
        for assumed in (0.19, 1.9, 5.5, 19.0):
            for arm in ("depth+PF",):
                for s in SEEDS[:3]:
                    t.append(dict(base, arm=arm, seed=s, truth_b_gamma=truth, cost_bg=assumed, _tag=f"sens_bg_truth{truth}_assumed{assumed}"))
        for arm in ("fix2+PF", "fix4+PF"):
            for s in SEEDS[:3]:
                t.append(dict(base, arm=arm, seed=s, truth_b_gamma=truth, _tag=f"sens_bg_truth{truth}_{arm}"))
    # (2) cost predictor error for the scheduler (density ranking and the optional predicted-time cap): assumed (a,b) scaled
    for fa, fb in itertools.product((0.25, 1.0, 4.0), (0.25, 1.0, 4.0)):
        for arm in ("fix4+DS", "fix4+DSz"):
            for s in SEEDS[:3]:
                t.append(dict(base, arm=arm, seed=s, cost_a=52.0 * fa, cost_b=0.19 * fb, _tag=f"sens_cost_a{fa}_b{fb}"))
    # (3) normalisation scales
    for V in (5.0, 50.0, 500.0):
        for arm in ("fix4+DS", "fix4+DSz", "fix4+PF"):
            for s in SEEDS[:3]:
                t.append(dict(base, arm=arm, seed=s, V=V, _tag=f"sens_V{V}"))
    for tr in (13.0, 52.0, 208.0, 832.0):
        for s in SEEDS[:3]:
            t.append(dict(base, arm="depth+PF", seed=s, t_ref_ms=tr, cost_bg=5.5, truth_b_gamma=5.5, _tag=f"sens_tref{tr}"))
    for ms in (0, 2, 8, 10 ** 6):
        for s in SEEDS[:3]:
            t.append(dict(base, arm="fix4+DSz", seed=s, max_skips=ms, _tag=f"sens_maxskips{ms}"))
    # (4) delayed state updates: the depth rule reads a snapshot that is `lag` ms old (fault injection), well specified b_gamma = 5.5
    for lag in (0.0, 250.0, 1000.0, 4000.0):
        for s_ in SEEDS[:3]:
            t.append(dict(base, arm="depth+PF", seed=s_, cost_bg=5.5, truth_b_gamma=5.5, read_lag_ms=lag, _tag=f"sens_readlag{lag}"))
    for cap_ms in (float("inf"), 900.0, 600.0):
        for arm in ("fix4+PF", "fix4+DSz"):
            for s in SEEDS[:3]:
                t.append(dict(base, arm=arm, seed=s, service_time_cap_ms=cap_ms, _tag=f"sens_timecap{cap_ms}"))
    return t


def tasks_adversarial():
    t = []
    # (1) persistent overload
    for arm in ("fix4+PF", "fix4+DS", "fix4+DSz"):
        for s in SEEDS[:3]:
            t.append(dict(arm=arm, n=64, cap=8, pop="hetero_async", seed=s, _tag="adv_overload"))
    # (2) weak client accumulating deficit: long context + low alpha client among strong ones (handled in worker via population override below)
    for arm in ("fix4+PF", "fix4+DS", "fix4+DSz"):
        for s in SEEDS[:3]:
            t.append(dict(arm=arm, n=40, cap=8, pop="weak_client", seed=s, _tag="adv_weak_client"))
    # (3) heterogeneous context lengths up to 1500 tokens: the processed-token budget binds
    for arm in ("fix4+PF", "fix4+DS", "fix4+DSz"):
        for s in SEEDS[:3]:
            t.append(dict(arm=arm, n=24, cap=16, pop="hetero_async", seed=s, ctx_max=1500, _tag="adv_long_context"))
    # (4) correlated arrivals: no offsets, no jitter, identical hardware (all clients return together)
    for arm in ("fix4+legacy", "fix4+PF", "fix4+DSz"):
        for s in SEEDS[:3]:
            t.append(dict(arm=arm, n=32, cap=8, pop="homog_sync", seed=s, _tag="adv_correlated"))
    # (5) empty / partial batches: very few clients
    for n in (1, 3):
        for arm in ("fix4+PF", "fix4+DSz", "depth+DSz"):
            for s in SEEDS[:3]:
                t.append(dict(arm=arm, n=n, cap=16, pop="hetero_async", seed=s, _tag=f"adv_few_clients_N{n}"))
    # (6) time slots with no arrivals: long staggered idle starts (clients appear 0..20 s apart)
    for arm in ("fix4+PF", "fix4+DSz", "depth+DSz"):
        for s in SEEDS[:3]:
            t.append(dict(arm=arm, n=16, cap=16, pop="hetero_async", seed=s, start_spread_ms=20000.0, _tag="adv_idle_slots"))
    return t


def tasks_openloop():
    """Open-loop replay (fixed gamma=4 acts; the depth rule only logs) vs closed loop (the depth rule acts), same populations/assumed costs."""
    t = []
    for pop in ("homog_sync", "hetero_async"):
        for (n, b) in ((24, 8), (24, 16)):
            for assumed in (0.19, 1.9, 5.5, 19.0):
                for truth in (0.19, 5.5):
                    for s in SEEDS[:3]:
                        base = dict(n=n, cap=b, pop=pop, seed=s, cost_bg=assumed, truth_b_gamma=truth)
                        t.append(dict(base, arm="fix4+PF", shadow_depth=True, _tag=f"ol_shadow_{pop}_N{n}_B{b}_assumed{assumed}_truth{truth}"))
                        t.append(dict(base, arm="depth+PF", _tag=f"ol_closed_{pop}_N{n}_B{b}_assumed{assumed}_truth{truth}"))
    return t


def tasks_capacity():
    """Interactivity capacity under each scheduling policy (CPU model): x_min(N) sweeps; N_c(x) = largest N with x_min >= x."""
    arms = ["fix4+legacy", "fix4+PF", "fix4+DS", "fix4+DSz", "fix2+PF", "fix2+DSz"]
    return [dict(arm=a, n=n, cap=b, pop=p, seed=s, _tag="capacity") for p in ("homog_sync", "hetero_async") for b in (8, 16)
            for n in range(8, 69, 4) for a in arms for s in SEEDS]


ROBUST_CELLS = [(24, 16), (32, 16), (24, 8), (32, 8), (48, 16)]       # first two = negative control (original high-load region, B_max=16)
ROBUST_FAMILIES = ("measured", "measured_jitter", "wide_async")
POP_SEEDS = tuple(range(1, 21))


def tasks_robust():
    """Independent populations (population seed = the sampling unit, one paired run each); algorithms fixed; ranges grounded in results/c3_1/trace_conditions.json
    for 'measured*'; 'wide_async' is the C3 assumption-driven family kept for contrast.  Arms: B = c3_fcfs, T = c3_deficit deficit-first, DSa = ablation."""
    t = []
    for fam in ROBUST_FAMILIES:
        for (n, b) in ROBUST_CELLS:
            for ps in POP_SEEDS:
                for arm in ("fix4+PF", "fix4+DSz", "fix4+DSa"):
                    t.append(dict(arm=arm, n=n, cap=b, pop=fam, seed=ps, pop_seed=ps, _tag=f"robust_{fam}"))
    for sd in (0.05, 0.10, 0.15):                                   # dose-response: how much acceptance spread does a benefit need?
        for (n, b) in ((24, 8), (32, 8), (48, 16)):
            for ps in POP_SEEDS:
                for arm in ("fix4+PF", "fix4+DSz"):
                    t.append(dict(arm=arm, n=n, cap=b, pop="measured", seed=ps, pop_seed=ps, alpha_sd=sd, _tag=f"dose_sd{sd}"))
    return t


def tasks_robust_g2():
    """Replication of the measured-grounded cells at the depth actually run on the GPU (gamma = 2): B = fix2+PF, A = fix2+DSa (alpha-free pilot treatment)."""
    return [dict(arm=arm, n=n, cap=b, pop=fam, seed=ps, pop_seed=ps, _tag=f"robustg2_{fam}") for fam in ("measured", "measured_jitter") for (n, b) in ROBUST_CELLS
            for ps in POP_SEEDS for arm in ("fix2+PF", "fix2+DSa")]


TASK_SETS = {"sched": tasks_sched, "depth": tasks_depth, "combined": tasks_combined, "sens": tasks_sens, "adversarial": tasks_adversarial, "openloop": tasks_openloop, "capacity": tasks_capacity, "robust": tasks_robust, "robust_g2": tasks_robust_g2}


def main():
    exp, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    names = list(TASK_SETS) if exp == "all" else [exp]
    res = {"label": "CPU closed-loop mechanism validation (model-derived; NOT a GPU capacity result). Matched static populations, 3-5 run-seeds, "
                    "fixed settings."}
    for name in names:
        tasks = TASK_SETS[name]()
        res[name] = pool_map(tasks, workers)
        print("done", name, len(tasks), flush=True)
    Path(out).write_text(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
