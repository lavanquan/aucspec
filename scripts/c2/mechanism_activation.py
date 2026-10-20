"""C2 Phase E: mechanism-activation experiments on the real simulator code with CPU stand-in models (see legacy/tests/closed_loop_harness.py).
Open-loop questions (same logged state, does the mechanism change the ACTION?) are kept apart from closed-loop questions (does the trajectory
change?).  No parameter is tuned toward a desired outcome: every setting below is fixed in advance and reported whatever the result.
usage: mechanism_activation.py <out.json>"""
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "tests"))
sys.path.insert(0, str(ROOT / "scripts" / "c1"))

import numpy as np  # noqa: E402

import closed_loop_harness as ch  # noqa: E402

NS = (16, 24, 32)


def het_alphas(n):
    return [0.60 + 0.35 * ch._u("alpha", i) for i in range(n)]


def run(spec):
    with contextlib.redirect_stdout(io.StringIO()):
        sim = ch.run_sim(spec)
    return sim


def flip_rates(log, window=None):
    keys = ["no_z", "no_qi", "no_qs", "benefit_only", "qs_x0.1", "qs_x10", "qi_x0.1", "qi_x10", "stale_drained"]
    rows = [r for r in log if window is None or window[0] <= r["t"] <= window[1]]
    out = {"n_decisions": len(rows)}
    for k in keys:
        out[k] = float(np.mean([r[k] != r["full"] for r in rows])) if rows else None
    out["gamma_full_dist"] = {int(g): float(np.mean([r["full"] == g for r in rows])) for g in sorted({r["full"] for r in rows})}
    out["z_mean"] = float(np.mean([r["z"] for r in rows]))
    out["z_p95"] = float(np.quantile([r["z"] for r in rows], 0.95))
    out["qi_mean"] = float(np.mean([r["qi"] for r in rows]))
    out["qi_max"] = float(max(r["qi"] for r in rows))
    out["qs_mean"] = float(np.mean([r["qs"] for r in rows]))
    out["qs_max"] = float(max(r["qs"] for r in rows))
    out["qi_age_slots_mean"] = float(np.mean([r["qi_age_slots"] for r in rows]))
    out["qs_age_slots_mean"] = float(np.mean([r["qs_age_slots"] for r in rows]))
    return out


def outcome(sim):
    s = ch.summarize(sim)
    waits = {}
    for r in sim.records:
        waits.setdefault(r["client_id"], []).append(r["batch_start_ms"] - r["server_arrival_ms"])
    pc = [float(np.mean(v)) for v in waits.values()]
    keys = ("x_min", "x_mean", "Y", "fair", "util", "batch_size_mean", "gamma_mean", "gamma_dist", "z_max", "qi_max", "qs_max", "round_ms_mean",
            "ledger_violations")
    out = {k: s[k] for k in keys}
    out["queue_wait_spread_ms"] = float(max(pc) - min(pc))
    return out


def e1_production(res):
    """Production regime: fixed-budget zero price (theta_f = 0), capacity_dpp + capacity_knapsack, cap 16."""
    res["E1_production"] = {}
    for pop in ("homog", "hetero"):
        for n in NS:
            al = None if pop == "homog" else het_alphas(n)
            hw = pop == "hetero"
            sim = run(ch.Spec(n=n, policy="capacity_dpp", alphas=al, hetero_hw=hw))
            fx = run(ch.Spec(n=n, policy="fixed_gamma", fixed_gamma=2, alphas=al, hetero_hw=hw))
            g4 = run(ch.Spec(n=n, policy="fixed_gamma", fixed_gamma=4, alphas=al, hetero_hw=hw))
            res["E1_production"][f"{pop}_N{n}"] = {
                "open_loop_flip_rates": flip_rates(sim.controller.decision_log, (sim.measurement_start_ms, sim.measurement_end_ms)),
                "closed_loop_capacity_dpp": outcome(sim), "closed_loop_fixed_gamma2": outcome(fx), "closed_loop_fixed_gamma4": outcome(g4)}


def e2_hypothetical_price(res):
    """OFFLINE activation of the server price (theta_f in ms per verifier position); NOT the canonical campaign.
    Variants separate WHERE the price acts: A = depth scoring + knapsack (as implemented), B = depth only (FCFS batching),
    C = knapsack only (depth unpriced), D = no price (production)."""
    res["E2_hypothetical_theta_f"] = {}
    for theta in (1.0, 5.0, 20.0):
        for n in (24, 32):
            al = het_alphas(n)
            base = dict(n=n, policy="capacity_dpp", theta_f=theta, alphas=al, hetero_hw=True)
            sim = run(ch.Spec(**base))
            cell = {"open_loop_flip_rates": flip_rates(sim.controller.decision_log, (sim.measurement_start_ms, sim.measurement_end_ms)),
                    "A_gamma_and_knapsack_priced": outcome(sim),
                    "B_gamma_priced_only_fcfs": outcome(run(ch.Spec(**base, scheduler="fcfs"))),
                    "C_knapsack_priced_only": outcome(run(ch.Spec(**base, ablate=frozenset({"qs"}))))}
            kn = [r for r in sim.batch_log]
            cell["A_forced_single_request_batches_fraction"] = float(np.mean([len(r["chosen"]) == 1 for r in kn])) if kn else None
            res["E2_hypothetical_theta_f"][f"theta{theta:g}_N{n}"] = cell
    res["E2_hypothetical_theta_f"]["D_production_no_price"] = {
        f"N{n}": outcome(run(ch.Spec(n=n, policy="capacity_dpp", alphas=het_alphas(n), hetero_hw=True))) for n in (24, 32)}


def e3_batch_selection(res):
    """Does capacity_knapsack ever differ from FCFS?  Open-loop on the logged eligible sets; closed-loop with the same gamma policy."""
    res["E3_batch_selection"] = {}
    for n in (24, 32):
        for x_req, label, hw in ((3.0, "hetero_hw_x3.0", True), (4.5, "hetero_hw_x4.5_Z_grows", True), (3.0, "homog_hw_hetero_alpha_x3.0", False),
                                 (4.5, "homog_hw_hetero_alpha_x4.5_Z_grows", False)):
            al = het_alphas(n)
            sim = run(ch.Spec(n=n, policy="fixed_gamma", fixed_gamma=2, scheduler="capacity_knapsack", x_req=x_req, alphas=al, hetero_hw=hw))
            log = sim.batch_log
            multi = [r for r in log if r["n_eligible"] > 1]
            d_fcfs = [set(r["capacity_knapsack"]) != set(r["fcfs"]) for r in multi]
            d_thr = [set(r["capacity_knapsack"]) != set(r["max_throughput"]) for r in multi]
            res["E3_batch_selection"][f"N{n}_{label}"] = {
                "batches": len(log), "cap_binds_fraction": float(np.mean([r["n_eligible"] > r["cap"] for r in log])),
                "token_budget_binds_fraction": float(np.mean([r["eligible_cost"] > r["budget"] for r in log])),
                "knapsack_vs_fcfs_different_member_set": float(np.mean(d_fcfs)) if d_fcfs else None,
                "knapsack_vs_max_throughput_different_member_set": float(np.mean(d_thr)) if d_thr else None,
                "closed_loop_knapsack": outcome(sim),
                "closed_loop_fcfs": outcome(run(ch.Spec(n=n, policy="fixed_gamma", fixed_gamma=2, scheduler="fcfs", x_req=x_req, alphas=al, hetero_hw=hw)))}


def e4_wireless(res):
    sim = ch.build(ch.Spec(n=24, hetero_hw=True, causal=False, log_batches=False))
    sim.causal_virtual_time = False
    rng = np.random.default_rng(3)
    out = {}
    for payload_name, payload in (("greedy_delta_proposal_gamma2_96B", 96), ("full_vocab_logprobs_gamma2_1.2MB", 2 * 151936 * 4 + 96)):
        per_alloc = {}
        for alloc in ("equal", "proportional", "queue-weighted", "square-root"):
            sim.cfg["clients"]["shared_wireless"] = dict(sim.cfg["clients"]["shared_wireless"], uplink_allocator=alloc)
            ups = []
            for c in sim.clients:
                c.gamma, c.device_queue, c.z_queue = 2, float(rng.random() * 30), float(rng.random() * 30)
                c.current_sample = c.current_sample or c.prompt_queue[0]
            for c in sim.clients:
                bw, rate, snr, se, _ = sim._allocate_shared_bandwidth(c, "uplink", 50000.0, payload)
                ups.append(sim._transfer_time_ms(payload, rate, c.packet_loss))
            per_alloc[alloc] = np.array(ups)
        base = per_alloc["proportional"]
        out[payload_name] = {a: {"upload_ms_mean": float(v.mean()), "upload_ms_max": float(v.max()),
                                 "max_abs_diff_vs_proportional_ms": float(np.abs(v - base).max())} for a, v in per_alloc.items()}
        out[payload_name]["round_ms_reference"] = 800.0
        out[payload_name]["max_alloc_effect_as_fraction_of_round"] = float(max(np.abs(v - base).max() for v in per_alloc.values()) / 800.0)
    out["rate_for_10pct_of_round_mbps"] = {"greedy_96B": 96 * 8 / (0.1 * 0.8) / 1e6, "full_vocab_1.2MB": (2 * 151936 * 4 + 96) * 8 / (0.1 * 0.8) / 1e6}
    res["E4_wireless"] = out


def e5_future_information(res):
    """Decisions must not depend on real completion order.  causal=True vs the historical real-order mode."""
    res["E5_future_information"] = {}
    for causal in (True, False):
        logs = []
        for delay in (0.0, 0.002, 0.004):
            sim = run(ch.Spec(n=16, policy="capacity_dpp", alphas=het_alphas(16), hetero_hw=True, causal=causal, delay_s=delay, seed=1,
                              window_s=40.0, warmup_s=10.0, margin_s=3.0))
            logs.append({(r["client"], round(r["t"], 3)): r["full"] for r in sim.controller.decision_log})
        keys = [set(x) for x in logs]
        common = keys[0] & keys[1] & keys[2]
        mism = sum(1 for k in common if not (logs[0][k] == logs[1][k] == logs[2][k]))
        res["E5_future_information"]["causal" if causal else "real_order"] = {
            "decisions_compared": len(common), "gamma_mismatches_across_real_delay_seeds": mism,
            "decision_keys_identical_across_runs": keys[0] == keys[1] == keys[2],
            "decisions_per_run": [len(x) for x in logs]}


def main():
    res = {"note": "CPU closed-loop harness on the real simulator code; verifier = fitted physical service-time model; acceptance Bernoulli "
                   "(coupled across policies); one run-seed per cell; fixed settings, none tuned toward an outcome."}
    for fn in (e1_production, e2_hypothetical_price, e3_batch_selection, e4_wireless, e5_future_information):
        fn(res)
        print("done", fn.__name__, flush=True)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
