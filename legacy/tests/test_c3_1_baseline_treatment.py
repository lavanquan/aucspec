"""C3.1 Task C: the three labelled arms are kept apart, B (work-conserving FCFS) and T (deficit-first) differ ONLY in the ranking, and the first-batch /
overloaded-batch behaviour is checked in the actual simulator (real queue + batcher + coordinator)."""
import dataclasses
import types

import pytest

import c3_scenarios as sc
from edge_specsim.c3_control import C3Params, ConstraintInfeasible, select_c3_deficit

SHORT = dict(window_s=40.0, warmup_s=8.0, margin_s=3.0)


def test_arm_labels_are_distinct_and_map_to_the_documented_implementations():
    assert sc.ARMS["fix4+legacy"][2:] == ("fcfs", False)                 # legacy head+W FCFS: historical reference only
    assert sc.ARMS["fix4+PF"][2:] == ("c3_fcfs", True)                   # C3 work-conserving FCFS: the fair baseline
    assert sc.ARMS["fix4+DSz"][2:] == ("c3_deficit", True)               # C3 work-conserving deficit-first: the treatment


@pytest.mark.parametrize("pop", ["measured", "measured_jitter", "wide_async", "hetero_async"])
def test_baseline_and_treatment_have_identical_information_caps_depth_offsets_jitter_and_prompts(pop):
    B = sc.make_spec("fix4+PF", 24, pop, seed=3, cap=8, pop_seed=7, **SHORT)
    T = sc.make_spec("fix4+DSz", 24, pop, seed=3, cap=8, pop_seed=7, **SHORT)
    fb, ft = dataclasses.asdict(B), dataclasses.asdict(T)
    differing = {k for k in fb if fb[k] != ft[k]}
    assert differing <= {"scheduler", "c3"}                              # everything else (alphas, prompts, offsets, jitter, caps, depth, seed, V, x_req, truth costs, oracle alpha) is identical
    assert fb["scheduler"] == "c3_fcfs" and ft["scheduler"] == "c3_deficit"
    cb, ct = fb["c3"], ft["c3"]
    assert {k for k in cb if cb[k] != ct[k]} == {"priority_mode"}        # inside the scheduler parameters ONLY the ranking differs: same caps, token budget, cost model, V, x_req
    assert B.work_conserving and T.work_conserving and B.fixed_gamma == T.fixed_gamma == 4 and B.max_batch == T.max_batch == 8


def test_baseline_and_treatment_share_prompt_assignment_in_the_actual_simulator():
    sb = sc.run(sc.make_spec("fix4+PF", 16, "measured_jitter", cap=8, pop_seed=2, **SHORT))
    st = sc.run(sc.make_spec("fix4+DSz", 16, "measured_jitter", cap=8, pop_seed=2, **SHORT))
    first = lambda sim: {c.client_id: c.prompt_queue[0].sample_id if c.prompt_queue else c.current_sample.sample_id for c in sim.clients}   # noqa: E731
    assert first(sb) == first(st)
    assert [c.rtt_ms for c in sb.clients] == [c.rtt_ms for c in st.clients]
    assert sb.controller.policy == st.controller.policy == "fixed_gamma" and sb.controller.fixed_gamma == st.controller.fixed_gamma == 4


def test_the_arrival_tie_break_ablation_uses_no_acceptance_estimate():
    """deficit-first with arrival tie-break (diagnostic arm) is independent of phi_hat / alpha: permuting expected useful tokens cannot change it."""
    def req(cid, z, useful, arrival):
        return types.SimpleNamespace(client_id=cid, arrival_time=arrival, request_id=0, context_token_ids=[0] * 100, candidate_token_ids=[0] * 4, gamma=4,
                                     weight=50.0 + z, decision_time_ms=0.0, expected_useful_tokens=useful, c3_skips=0)
    p = C3Params(max_batch_size=3, priority_mode="deficit_fcfs", max_skips=10 ** 6)
    zs, arr = [5.0, 5.0, 9.0, 1.0, 5.0, 0.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    a = [r.client_id for r in select_c3_deficit([req(i, zs[i], 1.0 + i, arr[i]) for i in range(6)], 100.0, p)]
    b = [r.client_id for r in select_c3_deficit([req(i, zs[i], 6.0 - i, arr[i]) for i in range(6)], 100.0, p)]
    assert a == b == [2, 0, 1]


# ------------------------------------------------------------------ first and overloaded batches in the actual simulator
def test_first_batch_in_the_actual_simulator_respects_the_cap_the_cutoff_and_arrival_order_for_the_baseline():
    n, cap = 24, 8
    kw = dict(start_offsets_ms=[0.0] * n)                                 # everybody arrives together: the first batch is overloaded
    sb = sc.run(sc.make_spec("fix4+PF", n, "homog_sync", cap=cap, **kw, **SHORT))
    st = sc.run(sc.make_spec("fix4+DSz", n, "homog_sync", cap=cap, **kw, **SHORT))
    for sim in (sb, st):
        first = sim.batch_log[0]
        assert first["n_eligible"] == n and len(first["chosen"]) == cap
        assert first["chosen_processed_tokens"] <= 4096
    fb = sb.batch_log[0]
    assert fb["chosen"] == fb["eligible_ids"][:cap]                       # baseline = arrival order (ties: client id)
    ft = st.batch_log[0]
    assert set(ft["chosen"]) <= set(ft["eligible_ids"])
    assert sb.coordinator.ledger.check_causality() == [] and st.coordinator.ledger.check_causality() == []


@pytest.mark.parametrize("n,cap", [(24, 8), (48, 16)])
def test_every_overloaded_batch_fills_the_cap_in_both_arms_and_the_baseline_is_arrival_ordered(n, cap):
    sb = sc.run(sc.make_spec("fix4+PF", n, "measured_jitter", cap=cap, pop_seed=4, **SHORT))
    st = sc.run(sc.make_spec("fix4+DSz", n, "measured_jitter", cap=cap, pop_seed=4, **SHORT))
    for sim, label in ((sb, "B"), (st, "T")):
        binding = [r for r in sim.batch_log if r["n_eligible"] > r["cap"]]
        assert len(binding) > 0.9 * len(sim.batch_log), label
        assert all(len(r["chosen"]) == cap for r in binding), label       # processed-token budget (4096) never binds at these contexts => fill to the cap
        assert all(r["chosen_processed_tokens"] <= 4096 for r in sim.batch_log), label
    assert all(r["chosen"] == r["eligible_ids"][:cap] for r in sb.batch_log if r["n_eligible"] > r["cap"])


def test_real_simulator_fails_closed_when_a_request_cannot_meet_the_processed_token_budget():
    with pytest.raises(ConstraintInfeasible):
        sc.run(sc.task_to_spec(dict(arm="fix4+PF", n=8, pop="homog_sync", cap=8, token_budget_processed=100, **SHORT)))


def test_membership_difference_is_measured_on_identical_eligible_sets_and_separates_deficit_from_tie_break():
    sb = sc.run(sc.make_spec("fix4+PF", 24, "measured_jitter", cap=8, pop_seed=4, **SHORT))
    binding = [r for r in sb.batch_log if r["n_eligible"] > r["cap"]]
    assert binding and "T_deficit_density" in binding[0] and "T_deficit_arrival" in binding[0] and "T0_density_only" in binding[0]
    # on the same eligible sets: the density tie-break never changes the deficit-first choice here, the deficit does
    assert all(set(r["T_deficit_density"]) == set(r["T_deficit_arrival"]) for r in binding)
    assert sum(set(r["T_deficit_density"]) != set(r["T0_density_only"]) for r in binding) > 0.5 * len(binding)


# ------------------------------------------------------------------ C4: falsification ranks (diagnostic modes only)
def _r(cid, arrival, decision, z, V=50.0):
    return types.SimpleNamespace(client_id=cid, arrival_time=arrival, request_id=0, context_token_ids=[0] * 100, candidate_token_ids=[0] * 2, gamma=2,
                                 weight=V + z, decision_time_ms=decision, expected_useful_tokens=2.0, c3_skips=0)


def test_age_rank_serves_the_oldest_decision_and_z_rank_uses_only_the_stamped_client_deficit():
    reqs = [_r(0, arrival=50.0, decision=10.0, z=0.0), _r(1, arrival=40.0, decision=30.0, z=9.0), _r(2, arrival=60.0, decision=20.0, z=3.0)]
    age = select_c3_deficit(list(reqs), 100.0, C3Params(max_batch_size=1, priority_mode="age", max_skips=10 ** 6))
    zon = select_c3_deficit(list(reqs), 100.0, C3Params(max_batch_size=1, priority_mode="z_only", max_skips=10 ** 6))
    assert [r.client_id for r in age] == [0] and [r.client_id for r in zon] == [1]


def test_in_the_grounded_homogeneous_hardware_family_age_order_equals_arrival_order_so_age_only_cannot_differ_from_fcfs():
    sb = sc.run(sc.make_spec("fix2+PF", 24, "measured", cap=8, pop_seed=2, **SHORT))
    sa = sc.run(sc.make_spec("fix2+AGE", 24, "measured", cap=8, pop_seed=2, **SHORT))
    mb, ma = sc.metrics(sb), sc.metrics(sa)
    assert abs(mb["x_min"] - ma["x_min"]) < 0.1 and abs(mb["Y"] - ma["Y"]) < 0.01 * mb["Y"]      # one short window: tie-order butterflies only
    binding = [r for r in sb.batch_log if r["n_eligible"] > r["cap"]]
    assert sum(set(r["R_age"]) != set(r["c3_fcfs"]) for r in binding) <= 0.01 * len(binding)      # age-only is (almost) identical to FCFS here
    assert sum(set(r["R_zonly"]) != set(r["c3_fcfs"]) for r in binding) > 0.5 * len(binding)        # the stamped deficit is what differs
