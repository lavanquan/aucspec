"""C2: CPU mechanism-activation tests on the real simulator code (closed-loop harness with a fitted physical verifier model).
Each test pins a MEASURED fact about the implemented controller (including negative facts such as 'this mechanism never changes an action');
none of them asserts that the controller is good.  Settings were fixed before looking at outcomes."""
import io
import contextlib
import random

import numpy as np
import pytest

import closed_loop_harness as ch
from edge_specsim.controller import OnlineController
from edge_specsim.models import ClientProfile


def quiet_run(spec):
    with contextlib.redirect_stdout(io.StringIO()):
        return ch.run_sim(spec)


def het_alphas(n):
    return [0.60 + 0.35 * ch._u("alpha", i) for i in range(n)]


SHORT = dict(window_s=60.0, warmup_s=10.0, margin_s=4.0)


def flips(sim, key):
    rows = [r for r in sim.controller.decision_log]
    return float(np.mean([r[key] != r["full"] for r in rows]))


# ------------------------------------------------------------------ harness exactness and fidelity
def test_ablation_harness_with_no_ablation_equals_production_scores():
    Ablated = ch.ablated_controller_class()
    kw = dict(gamma_choices=[0, 1, 2, 3, 4], V=50.0, min_tps=3.0, server_price_scale=0.02, policy="capacity_dpp", proposal_vocab_size=151936,
              proposal_distribution_payload="delta_proposal")
    prod, abl = OnlineController(**kw), Ablated(**kw)
    rng = random.Random(0)
    sim = ch.build(ch.Spec(n=4, log_batches=False))
    client = sim.clients[0]
    for _ in range(200):
        client.z_queue, client.device_queue = rng.random() * 80, rng.random() * 50
        for c in (prod, abl):
            c.server_queue = rng.random() * 800 if c is prod else prod.server_queue
            c.current_verifier_theta_f_ms_per_token = prod.current_verifier_theta_f_ms_per_token = rng.choice([0.0, 1.0, 10.0])
        client.alpha_ucb = client.alpha_hat = 0.5 + 0.49 * rng.random()
        for g in range(5):
            prof = client.positional_acceptance_profile(g, True)
            assert abl.capacity_dpp_score(client, g, prof) == pytest.approx(prod.capacity_dpp_score(client, g, prof), abs=1e-9)


def test_closed_loop_harness_reproduces_the_measured_fixed_gamma2_curve_at_n16():
    """Fidelity anchor (real C1.3 trace, cap 16, fixed gamma 2, FCFS): x_min 3.425, Y 56.43, busy 0.684."""
    sim = quiet_run(ch.Spec(n=16, policy="fixed_gamma", scheduler="fcfs"))
    s = ch.summarize(sim)
    assert s["x_min"] == pytest.approx(3.425, rel=0.06) and s["Y"] == pytest.approx(56.43, rel=0.06) and s["util"] == pytest.approx(0.684, abs=0.05)


def test_liveness_regression_n24_cap16_fcfs_completes():
    """C2 found a latent hang: the deadlock check peeked the head cutoff and stole an item from the asyncio queue the batcher was blocked on."""
    sim = quiet_run(ch.Spec(n=24, policy="fixed_gamma", scheduler="fcfs", window_s=20.0, warmup_s=5.0, margin_s=2.0))
    assert len(sim.records) > 24 * 5 and sim.coordinator.ledger.check_causality() == []


# ------------------------------------------------------------------ what each mechanism changes (production regime: theta_f = 0)
def test_production_regime_homogeneous_every_queue_term_is_inactive_and_depth_is_gamma_max():
    sim = quiet_run(ch.Spec(n=24, policy="capacity_dpp", **SHORT))
    for key in ("no_z", "no_qi", "no_qs", "benefit_only"):
        assert flips(sim, key) == 0.0
    assert {r["full"] for r in sim.controller.decision_log} == {4}                  # argmax phi = gamma_max
    fx4 = quiet_run(ch.Spec(n=24, policy="fixed_gamma", fixed_gamma=4, **SHORT))
    assert ch.summarize(sim)["x_min"] == pytest.approx(ch.summarize(fx4)["x_min"], rel=1e-9)   # identical trajectory to constant gamma=4


def test_server_price_is_inactive_in_the_canonical_zero_price_configuration_even_though_Qs_is_positive():
    sim = quiet_run(ch.Spec(n=16, policy="capacity_dpp", alphas=het_alphas(16), hetero_hw=True, **SHORT))
    qs = [r["qs"] for r in sim.controller.decision_log]
    assert np.mean(qs) > 100.0 and flips(sim, "no_qs") == 0.0                        # positive state, zero effect


def test_device_queue_changes_actions_only_through_a_stale_read_and_hurts_in_this_regime():
    sim = quiet_run(ch.Spec(n=24, policy="capacity_dpp", alphas=het_alphas(24), hetero_hw=True, **SHORT))
    f_qi, f_stale = flips(sim, "no_qi"), flips(sim, "stale_drained")
    assert f_qi > 0.1                                                                # the term does change many actions ...
    assert f_stale == pytest.approx(f_qi, abs=1e-12)                                 # ... and drains away entirely once idle slots are applied
    assert np.mean([r["qi_age_slots"] for r in sim.controller.decision_log]) > 3.0   # the read is several slots stale
    fx2 = quiet_run(ch.Spec(n=24, policy="fixed_gamma", fixed_gamma=2, alphas=het_alphas(24), hetero_hw=True, **SHORT))
    assert ch.summarize(sim)["x_min"] < ch.summarize(fx2)["x_min"]                   # worse floor than constant gamma=2 in this harness regime


def test_unit_scale_changes_decisions_when_the_server_price_is_active():
    sim = quiet_run(ch.Spec(n=24, policy="capacity_dpp", theta_f=1.0, alphas=het_alphas(24), hetero_hw=True, **SHORT))
    assert flips(sim, "qs_x0.1") > 0.2                                               # a 10x smaller price scale flips most decisions (x10 is already saturated at gamma=0)


def test_activating_the_server_price_with_ms_units_collapses_batching_to_single_request_batches():
    sim = quiet_run(ch.Spec(n=24, policy="capacity_dpp", theta_f=5.0, alphas=het_alphas(24), hetero_hw=True, **SHORT))
    forced = np.mean([len(r["chosen"]) == 1 for r in sim.batch_log])
    assert forced > 0.9 and ch.summarize(sim)["Y"] < 20.0                            # every capacity_dpp_value < 0 -> forced service of one request


def test_batch_selection_never_differs_from_fcfs_because_the_cap_stops_binding_after_the_first_round():
    sim = quiet_run(ch.Spec(n=24, policy="fixed_gamma", fixed_gamma=2, scheduler="capacity_knapsack", alphas=het_alphas(24), **SHORT))
    multi = [r for r in sim.batch_log if r["n_eligible"] > 1]
    assert np.mean([r["n_eligible"] > r["cap"] for r in sim.batch_log]) < 0.05
    assert all(set(r["capacity_knapsack"]) == set(r["fcfs"]) for r in multi)
    assert np.mean([r["eligible_cost"] > r["budget"] for r in sim.batch_log]) == 0.0  # the logical token budget never binds either


# ------------------------------------------------------------------ wireless
def test_wireless_allocator_is_irrelevant_for_greedy_payloads_and_decisive_for_full_distribution_uploads():
    sim = ch.build(ch.Spec(n=24, hetero_hw=True, causal=False, log_batches=False))
    rng = np.random.default_rng(3)
    res = {}
    for name, payload in (("greedy", 96), ("full", 2 * 151936 * 4 + 96)):
        per = {}
        for alloc in ("proportional", "queue-weighted", "square-root"):
            sim.cfg["clients"]["shared_wireless"] = dict(sim.cfg["clients"]["shared_wireless"], uplink_allocator=alloc)
            for c in sim.clients:
                c.gamma, c.device_queue, c.z_queue = 2, float(rng.random() * 30), float(rng.random() * 30)
                c.current_sample = c.current_sample or c.prompt_queue[0]
            per[alloc] = np.array([sim._transfer_time_ms(payload, sim._allocate_shared_bandwidth(c, "uplink", 50000.0, payload)[1], c.packet_loss)
                                   for c in sim.clients])
        res[name] = max(np.abs(per[a] - per["proportional"]).max() for a in per)
    assert res["greedy"] < 0.01 * 800.0          # < 1 % of a ~800 ms round
    assert res["full"] > 800.0                   # seconds: communication-limited regime exists only for full-distribution uploads


# ------------------------------------------------------------------ future information
def test_decisions_do_not_depend_on_real_completion_order_in_causal_mode_but_do_in_real_order_mode():
    def decisions(causal, delay):
        sim = quiet_run(ch.Spec(n=12, policy="capacity_dpp", alphas=het_alphas(12), hetero_hw=True, causal=causal, delay_s=delay,
                                window_s=20.0, warmup_s=5.0, margin_s=2.0))
        return {(r["client"], round(r["t"], 3)): r["full"] for r in sim.controller.decision_log}

    c0, c1, c2 = decisions(True, 0.0), decisions(True, 0.002), decisions(True, 0.004)
    assert c0 == c1 == c2
    r0, r1, r2 = decisions(False, 0.0), decisions(False, 0.002), decisions(False, 0.004)
    assert not (set(r0) == set(r1) == set(r2) and r0 == r1 == r2)


def test_selection_is_structurally_inert_even_with_a_tiny_cap_because_eligibility_is_limited_to_the_head_arrival_window():
    n = 32
    plens = [int(80 + 700 * ch._u("pl", i)) for i in range(n)]
    sim = quiet_run(ch.Spec(n=n, policy="fixed_gamma", fixed_gamma=2, scheduler="capacity_knapsack", hetero_hw=True, alphas=het_alphas(n),
                            max_batch=4, prompt_lens=plens, **SHORT))
    multi = [r for r in sim.batch_log if r["n_eligible"] > 1]
    assert np.mean([r["n_eligible"] > r["cap"] for r in sim.batch_log]) < 0.05          # saturated verifier, cap 4, yet the cap almost never binds
    assert all(set(r["capacity_knapsack"]) == set(r["fcfs"]) for r in multi)
    assert ch.summarize(sim)["util"] > 0.99


def test_evidence_inventory_keeps_every_old_verdict_and_claims_no_statistics():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    import importlib.util
    spec = importlib.util.spec_from_file_location("inv", root / "scripts" / "c2" / "build_evidence_inventory.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.main()
    inv = json.loads((root / "results" / "c2" / "rq1_evidence_inventory.json").read_text())
    pts = {(p["campaign"], p["N"]): p for p in inv["points"]}
    assert pts[("C1.3-cap16", 24)]["original_gate"] == "FAIL" and pts[("C1.3-cap16", 24)]["validity_checks_exploratory_v2"] == "pass"
    assert pts[("C1.4-probe", 32)]["original_gate"] == "FAIL"
    assert all(p["statistical_claim_supported"] is False for p in inv["points"])
    assert ("C1.2-extension", 24) not in pts                                           # the OOM run produced no measurement
