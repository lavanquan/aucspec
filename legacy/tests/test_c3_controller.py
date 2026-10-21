"""C3 scientific-integrity tests for the opt-in minimal capacity-aware controller (mechanisms A and B).

These pin INTEGRITY properties (causality, conservation, dimensional consistency, constraint enforcement, determinism, liveness) and MEASURED facts about
which mechanism acts and why (including negative facts).  None asserts that a mechanism wins; CPU closed-loop numbers are model-derived, never GPU results."""
import math
import types

import pytest

import c3_scenarios as sc
import closed_loop_harness as ch
from edge_specsim.c3_control import (C3Controller, C3Params, CostModel, WorkLedger, depth_score, effective_deficit, select_c3_deficit,
                                     select_c3_fcfs)

SHORT = dict(window_s=40.0, warmup_s=8.0, margin_s=3.0)


# ----------------------------------------------------------------------------------------------------------------------
# Stage A: conservation and causal-read invariants on a simple deterministic event trace
# ----------------------------------------------------------------------------------------------------------------------
def _trace():
    led = WorkLedger()
    led.arrive(0.0, 1, 10.0)
    led.arrive(5.0, 2, 20.0)
    led.arrive(30.0, 3, 5.0)
    led.serve(5.0, 87.0, 52.0, [1, 2])         # 52 + 30 = 82 ms of service
    led.serve(87.0, 144.0, 52.0, [3])          # 52 + 5
    return led


def test_each_request_enters_arrivals_once_and_departures_once_no_double_counting():
    led = _trace()
    assert led.A(1e9) == pytest.approx(35.0) and led.D(1e9) == pytest.approx(35.0)       # sum of demands, counted once each
    assert led.A(4.9) == pytest.approx(10.0) and led.D(86.9) == 0.0 and led.D(87.0) == pytest.approx(30.0)
    assert all(led.A(t) + 1e-9 >= led.D(t) for t in range(0, 200, 3))


def test_unfinished_work_is_nonnegative_drains_at_unit_rate_while_busy_and_vanishes_when_all_served():
    led = _trace()
    ts = [t / 2.0 for t in range(0, 320)]
    u = [led.unfinished_work(t) for t in ts]
    assert min(u) >= 0.0 and led.unfinished_work(144.0) == 0.0
    # inside a service interval with no new arrival the remaining work falls with slope -1 (verifier busy = unit service rate)
    assert led.unfinished_work(40.0) - led.unfinished_work(50.0) == pytest.approx(10.0)
    assert led.unfinished_work(5.0) == pytest.approx(82.0)                                # waiting demand turned into in-service remaining (overhead charged once)


def test_overlapping_service_is_rejected():
    led = WorkLedger()
    led.serve(0.0, 100.0, 52.0, [1])
    with pytest.raises(ValueError):
        led.serve(50.0, 120.0, 52.0, [2])


def _meta(finish, wall=100.0, useful=10, leftover=0.0, start=None, predicted=None):
    start = finish - wall if start is None else start
    return types.SimpleNamespace(verify_finish_ms=finish, batch_wall_ms=wall, batch_useful_tokens=useful, waiting_work_after_ms=leftover,
                                 batch_start_ms=start, predicted_service_ms=wall if predicted is None else predicted)


def _controller(**kw):
    return C3Controller(gamma_choices=[0, 1, 2, 3, 4], V=50.0, min_tps=3.0, server_price_scale=0.02, policy="c3_depth", seed=0, slot_ms=100.0,
                        min_exploration_rounds=0, exploration_epsilon=0.0, proposal_vocab_size=151936, proposal_distribution_payload="delta_proposal", **kw)


def test_snapshot_read_is_a_function_of_published_batches_started_or_finished_at_or_before_t_only():
    full, prefix = _controller(), _controller()
    for c in (full, prefix):
        c.observe_batch_meta(_meta(100.0, useful=10, leftover=80.0))                 # [0, 100]
        c.observe_batch_meta(_meta(200.0, useful=20, leftover=40.0))                 # [100, 200]
    full.observe_batch_meta(_meta(300.0, useful=999, leftover=500.0))               # [200, 300]: a batch that has not STARTED at the reads below
    for t in (50.0, 100.0, 150.0, 199.999):
        assert full.snapshot_at(t) == prefix.snapshot_at(t)                          # the future batch never leaks
    assert full.snapshot_at(199.999)["version"] == 1 and full.snapshot_at(200.0)["version"] == 2
    assert full.snapshot_at(200.0)["started"] == 3                                     # batch 3 started AT 200 and is therefore visible at t=200 ...
    assert full.snapshot_at(199.999)["started"] == 2                                   # ... but not a microsecond earlier
    assert prefix.snapshot_at(250.0) == prefix.snapshot_at(250.0)
    assert full.snapshot_at(-1.0)["version"] == 0 and full.snapshot_at(-1.0)["u_hat_ms"] == 0.0


def test_demand_estimate_uses_predicted_remaining_service_plus_leftover_and_never_the_realized_finish():
    c = _controller()
    c.observe_batch_meta(_meta(finish=400.0, wall=400.0, start=0.0, predicted=250.0, leftover=80.0))     # realized 400 ms, predicted 250 ms
    snap = lambda t: c.snapshot_at(t)["u_hat_ms"]                                                           # noqa: E731
    assert snap(0.0) == pytest.approx(250.0 + 80.0)
    assert snap(100.0) == pytest.approx(150.0 + 80.0)
    assert snap(300.0) == pytest.approx(80.0)                                                               # predicted service over; leftover still waiting
    c2 = _controller()
    c2.observe_batch_meta(_meta(finish=400.0, wall=400.0, start=0.0, predicted=250.0, leftover=80.0))
    c2._c3_hist[0] = c2._c3_hist[0][:1] + (9999.0,) + c2._c3_hist[0][2:]                                    # corrupt the realized wall time
    assert c2.snapshot_at(100.0)["u_hat_ms"] == snap(100.0)                                                 # demand read is independent of it


def test_idle_slots_zero_the_demand_estimate_only_through_the_published_trace_not_through_wall_clock():
    c = _controller()
    c.observe_batch_meta(_meta(finish=100.0, wall=100.0, start=0.0, predicted=100.0, leftover=0.0))
    assert [c.snapshot_at(t)["u_hat_ms"] for t in (50.0, 100.0, 1e6)] == pytest.approx([50.0, 0.0, 0.0])


def test_snapshots_must_be_published_in_finish_order():
    c = _controller()
    c.observe_batch_meta(_meta(200.0))
    with pytest.raises(ValueError):
        c.observe_batch_meta(_meta(100.0))


# ----------------------------------------------------------------------------------------------------------------------
# dimensional consistency of the depth score
# ----------------------------------------------------------------------------------------------------------------------
def test_depth_score_is_invariant_to_the_time_unit():
    s_ms = depth_score(1.4, 3.1, 3, mu_tok_per_ms=0.12, b_gamma_ms=0.19, xhat_tok_per_ms=0.003, c_dev_ms=22.2)
    k = 1000.0                                                                      # ms -> s : rates x k, durations / k
    s_s = depth_score(1.4, 3.1, 3, mu_tok_per_ms=0.12 * k, b_gamma_ms=0.19 / k, xhat_tok_per_ms=0.003 * k, c_dev_ms=22.2 / k)
    assert s_ms == pytest.approx(s_s, rel=1e-12)


def test_depth_score_terms_are_tokens_and_deficit_weight_cancels_in_the_own_latency_trade_off():
    # with no verifier price, scaling the deficit weight omega scales the whole score: the argmax over gamma cannot depend on omega
    def argmax(omega):
        return max(range(5), key=lambda g: depth_score(omega, (1 - 0.8 ** (g + 1)) / 0.2 if g else 1.0, g, 0.0, 0.19, 0.003, 22.2))
    assert len({argmax(w) for w in (1.0, 2.0, 10.0, 100.0)}) == 1


# ----------------------------------------------------------------------------------------------------------------------
# Mechanism A: constraint enforcement, liveness, starvation bound (synthetic requests)
# ----------------------------------------------------------------------------------------------------------------------
def _req(cid, arrival=0.0, ctx=100, gamma=2, z=0.0, V=50.0, decision=0.0, rid=0, useful=2.0):
    return types.SimpleNamespace(client_id=cid, arrival_time=arrival, request_id=rid, context_token_ids=[0] * ctx, candidate_token_ids=[0] * gamma,
                                 gamma=gamma, weight=V + z, decision_time_ms=decision, expected_useful_tokens=useful, c3_skips=0)


def test_selection_enforces_batch_size_processed_token_budget_and_predicted_time_cap():
    reqs = [_req(i, ctx=300) for i in range(10)]
    p = C3Params(max_batch_size=4, processed_token_budget=10 ** 6)
    assert len(select_c3_deficit(list(reqs), 1000.0, p)) == 4
    p = C3Params(max_batch_size=16, processed_token_budget=1000)                      # 302 tokens each -> 3 fit
    assert len(select_c3_deficit(list(reqs), 1000.0, p)) == 3
    p = C3Params(max_batch_size=16, processed_token_budget=10 ** 6, service_time_cap_ms=52.0 + 2 * (0.19 * 300 + 0.19 * 2) + 1.0)
    assert len(select_c3_fcfs(list(reqs), 1000.0, p)) == 2                            # predicted time cap binds at 2 requests


def test_oversize_singleton_fails_closed_instead_of_bypassing_the_processed_token_budget():
    from edge_specsim.c3_control import ConstraintInfeasible

    big = _req(0, ctx=5000)
    for sel in (select_c3_deficit, select_c3_fcfs):
        with pytest.raises(ConstraintInfeasible):
            sel([big], 0.0, C3Params(processed_token_budget=4096))
    with pytest.raises(ConstraintInfeasible):                                               # one inadmissible request poisons the whole selection, even with admissible peers
        select_c3_deficit([_req(1, ctx=100), big], 0.0, C3Params(processed_token_budget=4096))


def test_singleton_whose_predicted_service_exceeds_the_time_cap_fails_closed():
    from edge_specsim.c3_control import ConstraintInfeasible

    r = _req(0, ctx=300, gamma=2)                                                          # predicted 52 + 0.19*300 + 0.19*2 = 109.4 ms
    with pytest.raises(ConstraintInfeasible):
        select_c3_fcfs([r], 0.0, C3Params(service_time_cap_ms=100.0))
    assert select_c3_fcfs([r], 0.0, C3Params(service_time_cap_ms=110.0)) == [r]


def test_hard_limits_are_inclusive_at_the_boundary_and_liveness_holds_for_admissible_workloads():
    exact = _req(0, ctx=4094, gamma=2)                                                      # 4096 processed tokens == budget: admissible
    assert select_c3_deficit([exact], 0.0, C3Params(processed_token_budget=4096)) == [exact]
    reqs = [_req(i, ctx=3000) for i in range(5)]                                            # each admissible alone, no two fit together
    for sel in (select_c3_deficit, select_c3_fcfs):
        assert len(sel(list(reqs), 0.0, C3Params(processed_token_budget=4096))) == 1        # non-empty: liveness without any bypass
    assert select_c3_deficit([], 0.0, C3Params()) == []


def test_submit_preflight_rejects_an_inadmissible_request_before_it_is_queued():
    import asyncio

    from edge_specsim.c3_control import ConstraintInfeasible
    from edge_specsim.models import DraftResult
    from edge_specsim.verification_batcher import VerificationBatcher

    class T:
        pass

    b = VerificationBatcher(T(), max_batch_size=4, scheduler_name="c3_fcfs", service_time_model="batch_wall", c3_params=C3Params(processed_token_budget=100))
    draft = DraftResult(0, [1, 2], [None, None], "", 1.0, 0, distribution_payload="delta_proposal")
    with pytest.raises(ConstraintInfeasible):
        asyncio.run(b.submit(0, [0] * 200, draft, 10, 0.0))
    assert not b.server_queue and not b.queue.qsize()


def test_deficit_first_ranking_serves_the_neediest_and_effective_deficit_grows_while_waiting():
    p = C3Params(max_batch_size=1, priority_mode="deficit", max_skips=10 ** 6)
    needy, rich = _req(1, z=40.0), _req(2, z=0.0)
    assert select_c3_deficit([rich, needy], 100.0, p)[0] is needy
    r = _req(3, z=5.0, decision=0.0)
    assert effective_deficit(r, 2000.0, 3.0, 50.0) == pytest.approx(5.0 + 3.0 * 2.0)  # + x_req * 2 s of waiting


def test_starvation_guard_bounds_the_number_of_batches_a_request_can_be_skipped():
    p = C3Params(max_batch_size=1, priority_mode="density", max_skips=2)
    victim = _req(0, ctx=900, useful=1.0)                                             # worst density: always loses on rank
    served_at = None
    for rnd in range(20):
        pool = [victim] + [_req(10 + rnd * 3 + j, ctx=100, useful=3.0, rid=rnd) for j in range(3)]
        if select_c3_deficit(pool, 100.0 * rnd, p)[0] is victim:
            served_at = rnd
            break
    assert served_at is not None and served_at <= p.max_skips                         # served after at most max_skips skipped batches


def test_selection_is_deterministic_for_identical_eligible_sets():
    def run():
        reqs = [_req(i, arrival=float(i % 3), ctx=100 + 7 * i, useful=1.0 + (i % 4), z=float(i % 5)) for i in range(12)]
        return [r.client_id for r in select_c3_deficit(reqs, 50.0, C3Params(max_batch_size=5, priority_mode="deficit"))]
    assert run() == run()


# ----------------------------------------------------------------------------------------------------------------------
# closed loop on the real simulator code: causality, conservation, constraints, determinism, liveness
# ----------------------------------------------------------------------------------------------------------------------
ARMS_ALL = ("fix2+legacy", "fix4+PF", "fix4+DS", "fix4+DSz", "depth+PF", "depth+DSz")


@pytest.mark.parametrize("pop", sc.POPULATIONS)
def test_every_arm_has_a_clean_causal_ledger_no_empty_batches_and_respects_constraints(pop):
    for arm in ARMS_ALL:
        sim = sc.run(sc.make_spec(arm, 32, pop, cap=8, **SHORT))
        m = sc.metrics(sim)
        assert m["ledger_violations"] == 0 and not m["invalid"], arm
        assert m["max_batch_seen"] <= 8, arm
        assert all(r["chosen"] for r in sim.batch_log), arm                                # no empty batch ever
        if arm != "fix2+legacy":                                                           # legacy enforces only the logical gamma+1 budget (reference)
            assert all(r["chosen_processed_tokens"] <= 4096 or len(r["chosen"]) == 1 for r in sim.batch_log), arm


def test_queue_conservation_every_submitted_request_is_served_exactly_once():
    sim = sc.run(sc.make_spec("depth+DSz", 24, "hetero_async", cap=8, **SHORT))
    led = sim.coordinator.ledger
    arrivals = [e for e in led.events if e.type == "arrival"]
    served = [tuple(m) for e in led.events if e.type == "batch_formed" for m in e.detail["members"]]
    assert len(served) == len(set(served))                                                  # no request served twice
    assert {(e.detail["client"], e.detail["request"]) for e in arrivals} == set(served)    # and none left unserved at the end of the run
    assert len(sim.records) == len(served)


def test_every_depth_decision_saw_exactly_the_batches_started_and_finished_by_its_virtual_time():
    """Causal-read invariant on the real event trace: the snapshot used at virtual time t contains every batch with start <= t (the coordinator's
    horizon rule publishes them first) and no batch with start > t; goodput uses only batches with finish <= t."""
    import bisect

    sim = sc.run(sc.make_spec("depth+DSz", 24, "hetero_async", cap=8, **SHORT))
    led = sim.coordinator.ledger
    starts = sorted(e.virtual_ts for e in led.events if e.type == "batch_formed")
    finishes = sorted(e.detail["finish_ms"] for e in led.events if e.type == "batch_complete")
    assert sim.controller.c3_log
    for r in sim.controller.c3_log:
        assert r["started"] == bisect.bisect_right(starts, r["t"] + 1e-9), r
        assert r["version"] == bisect.bisect_right(finishes, r["t"] + 1e-9), r
        assert r["start_used"] is None or r["start_used"] <= r["t"] + 1e-9


def test_decisions_and_batch_membership_are_identical_under_permuted_asyncio_scheduling():
    def trace(delay):
        sim = sc.run(sc.make_spec("depth+DSz", 16, "hetero_async", cap=8, delay_s=delay, window_s=25.0, warmup_s=5.0, margin_s=2.0))
        # the LOG ORDER of concurrent clients is real-time order; the content (what each decision/batch was) must not depend on it
        return (sorted((r["client"], round(r["t"], 3), r["gamma"]) for r in sim.controller.c3_log),
                [tuple(r["chosen"]) for r in sim.batch_log],
                sorted((r["client_id"], r["verification_batch_id"], r["useful_tokens"]) for r in sim.records))
    a, b, c = trace(0.0), trace(0.002), trace(0.004)
    assert a == b == c


def test_work_conserving_formation_removes_batch_fragmentation_when_arrivals_are_asynchronous():
    legacy = sc.metrics(sc.run(sc.make_spec("fix2+legacy", 24, "hetero_async", cap=16, **SHORT)))
    wc = sc.metrics(sc.run(sc.make_spec("fix2+PF", 24, "hetero_async", cap=16, **SHORT)))
    assert wc["batch_size_mean"] > 1.5 * legacy["batch_size_mean"]                           # head+W windowing fragments batches under async arrivals
    assert wc["Y"] >= legacy["Y"] and wc["ledger_violations"] == 0


def test_in_the_measured_regime_the_batch_cap_never_binds_so_selection_has_no_leverage():
    """closed system with N <= 2*B_max: the batch that just finished is in transit, so at most N - |prev| <= B_max requests can be eligible."""
    for pop in sc.POPULATIONS:
        sim = sc.run(sc.make_spec("fix4+DSz", 24, pop, cap=16, **SHORT))
        assert sc.metrics(sim)["batches_eligible_gt_cap"] < 0.05
        multi = [r for r in sim.batch_log if r["n_eligible"] > 1]
        assert sum(set(r["c3_fcfs"]) != set(r["c3_deficit"]) for r in multi) <= 0.05 * max(1, len(multi))


def test_selection_has_leverage_once_the_cap_binds_and_deficit_first_flattens_the_rates_in_heterogeneous_overload():
    pf = sc.metrics(sc.run(sc.make_spec("fix4+PF", 24, "hetero_async", cap=8, seed=1)))
    dz = sc.metrics(sc.run(sc.make_spec("fix4+DSz", 24, "hetero_async", cap=8, seed=1)))
    assert pf["batches_eligible_gt_cap"] > 0.9 and dz["batches_eligible_gt_cap"] > 0.9
    assert dz["jain"] > pf["jain"]                                                          # measured: rates are flatter
    assert dz["x_min"] != pytest.approx(pf["x_min"], rel=1e-6)                              # and the trajectory is genuinely different


# ----------------------------------------------------------------------------------------------------------------------
# Mechanism B: what the depth rule does and why
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pop", sc.POPULATIONS)
def test_depth_control_at_assumed_costs_is_identical_to_constant_gamma_max(pop):
    d = sc.metrics(sc.run(sc.make_spec("depth+PF", 24, pop, cap=8, **SHORT)))
    f = sc.metrics(sc.run(sc.make_spec("fix4+PF", 24, pop, cap=8, **SHORT)))
    assert d["x_min"] == pytest.approx(f["x_min"], rel=1e-12) and d["Y"] == pytest.approx(f["Y"], rel=1e-12)


def test_depth_control_does_act_when_the_candidate_token_cost_is_large_and_acceptance_low():
    """The rule is not vacuous by construction: with an (unmeasured) b_gamma of 19 ms/token known to the controller AND true, and low acceptance,
    it picks shallower depths.  Whether such a cost exists on the real verifier is exactly what the C1 traces cannot say."""
    n = 16
    low = dict(alphas=[0.35 + 0.1 * (i % 4) for i in range(n)])
    spec = sc.task_to_spec(dict(arm="depth+PF", n=n, pop="homog_sync", cap=8, cost_bg=19.0, truth_b_gamma=19.0, **low, **SHORT))
    sim = sc.run(spec)
    g = [r["gamma"] for r in sim.controller.c3_log]
    assert sum(x != 4 for x in g) / len(g) > 0.1 and min(g) < 4


def test_absorbing_gamma_zero_without_optimism_when_acceptance_is_learned_online():
    """A depth rule that can output gamma=0 needs exploration/optimism: with learned alpha_hat and no UCB a client that once sees low acceptance stops
    producing acceptance feedback.  (C3 uses an offline-profiled alpha for every arm; learning is a separate question - Theorem 4.)"""
    spec = sc.make_spec("depth+PF", 24, "homog_sync", cap=16, oracle_alpha=False, **SHORT)
    sim = sc.run(spec)
    g = {}
    for r in sim.controller.c3_log:
        g.setdefault(r["client"], []).append(r["gamma"])
    stuck = [c for c, v in g.items() if len(v) > 20 and all(x == 0 for x in v[-20:])]
    assert stuck, "expected at least one client absorbed at gamma=0 (documented failure mode of unoptimistic learned depth control)"


def test_unmodified_capacity_dpp_and_legacy_schedulers_are_not_touched():
    # the canonical policy names/semantics: Capacity-DPP still resolves to the unmodified controller path; C3 policy is a separate subclass
    from edge_specsim.controller import OnlineController
    c = OnlineController(gamma_choices=[0, 1, 2], V=50.0, min_tps=3.0, server_price_scale=0.02, policy="capacity_dpp", fixed_gamma=0)
    assert c.policy == "capacity_dpp" and not hasattr(c, "c3_log")
    with pytest.raises(ValueError):
        OnlineController(gamma_choices=[0, 1, 2], V=50.0, min_tps=3.0, server_price_scale=0.02, policy="c3_depth", fixed_gamma=0)


# ----------------------------------------------------------------------------------------------------------------------
# online service-time estimate: what it can and cannot learn
# ----------------------------------------------------------------------------------------------------------------------
def test_online_estimator_learns_context_cost_and_reports_the_unidentified_gamma_coefficient():
    import random

    from edge_specsim.c3_control import OnlineCostEstimator

    rng = random.Random(0)
    est = OnlineCostEstimator(CostModel(a_ms=30.0, b_ms_per_token=0.05, b_gamma_ms_per_token=0.05, c_req_ms=0.0))
    prior_std = est.coef_std().copy()
    truth = dict(a=52.0, b=0.19, c=6.0, bg=0.19)                            # a real verifier may have a per-request fixed cost
    def wall(members):
        return (truth["a"] + truth["b"] * sum(c for c, _ in members) + truth["c"] * len(members) + truth["bg"] * sum(g for _, g in members)
                + rng.gauss(0, 10))
    for _ in range(800):                                                    # gamma constant (=2) in every batch, as in the C1 traces
        members = [(rng.randint(80, 300), 2) for _ in range(rng.randint(1, 16))]
        est.update(members, wall(members))
    a, b, c, bg = est.theta
    assert abs(b - 0.19) < 0.01                                             # the context slope is identified
    assert abs((c + 2 * bg) - (6.0 + 2 * 0.19)) < 1.5                       # only the COMBINATION c_req + gamma*b_gamma is identified ...
    std_const = est.coef_std().copy()
    assert std_const[3] > 0.3 * prior_std[3] and std_const[2] > 0.3 * prior_std[2]   # ... each coefficient alone keeps a large part of its prior spread
    for _ in range(800):                                                    # once the depth varies across batches the split becomes identifiable
        members = [(rng.randint(80, 300), rng.randint(0, 4)) for _ in range(rng.randint(1, 16))]
        est.update(members, wall(members))
    assert est.coef_std()[3] < 0.2 * std_const[3] and est.coef_std()[2] < 0.5 * std_const[2]   # identification gained once the depth varies
    assert abs(est.theta[3] - 0.19) < 0.1 and abs(est.theta[2] - 6.0) < 2.0


def test_online_estimator_prediction_uses_only_formation_time_features():
    from edge_specsim.c3_control import OnlineCostEstimator

    est = OnlineCostEstimator()
    before = est.predict_ms([(100, 2), (200, 4)])
    est.update([(100, 2), (200, 4)], 500.0)                                 # the measurement arrives after the prediction was made
    assert before == pytest.approx(CostModel().batch_ms([(100, 2), (200, 4)]), rel=1e-6)


def test_simulator_start_offset_and_jitter_options_are_opt_in_and_keep_the_ledger_clean():
    """The real simulator's opt-in asynchrony knobs (clients.start_offset_ms_max / arrival_jitter_ms_max) exist for the GPU pilot; defaults are no-ops."""
    import inspect

    from edge_specsim import simulator as simmod

    src = inspect.getsource(simmod.EdgeSpecSimulator.__init__)
    assert "start_offset_ms_max" in src and 'get("start_offset_ms_max", 0.0)' in src
    run_src = inspect.getsource(simmod.EdgeSpecSimulator._run_round)
    assert 'get("arrival_jitter_ms_max", 0.0)' in run_src
    sim = sc.run(sc.make_spec("fix4+PF", 16, "hetero_async", cap=8, **SHORT))      # harness mirrors both knobs (offsets + jitter) with the same semantics
    assert sc.metrics(sim)["ledger_violations"] == 0
