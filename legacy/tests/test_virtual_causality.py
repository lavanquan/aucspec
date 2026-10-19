"""CPU tests for virtual-time causality (audit items D1/D2)."""
import asyncio
import itertools
import random

import pytest

from causality_harness import Spec, default_wall, reference_batches, run_scenario
from edge_specsim.controller import OnlineController
from edge_specsim.virtual_causality import (
    CausalityDeadlock,
    CausalityViolation,
    EventLedger,
    VirtualCoordinator,
)


def run(coro):
    return asyncio.run(coro)


def canon(result):
    """Real-order independent view of a run: batches + per-client receive times + ledger content."""
    led = sorted(
        (e.virtual_ts, e.type, e.producer, repr(sorted((k, v) for k, v in e.detail.items() if k != "min_idle_lb")))
        for e in result["ledger"].events
    )
    return result["batches"], {k: [round(x, 9) for x in v] for k, v in result["receives"].items()}, led


def scenario(real_delays, rtts=(20.0, 30.0, 40.0, 50.0), draft=(80.0, 110.0, 95.0)):
    return [Spec(i, rtts[i], list(draft), real_delays[i]) for i in range(len(rtts))]


# ---------------------------------------------------------------- D1 -------------------------------------------
def test_d1_legacy_misses_already_arrived_request_causal_does_not():
    # client 1 has the EARLIER virtual arrival (1 ms draft) but is really slow (0.15 s); client 0 is really fast.
    specs = [Spec(0, 20.0, [10.0], [0.0]), Spec(1, 20.0, [1.0], [0.15])]
    legacy = run(run_scenario(specs, causal=False))
    causal = run(run_scenario(specs, causal=True))
    assert any(v.startswith("D1") for v in legacy["violations"]), legacy["violations"]
    assert causal["violations"] == []
    assert [b["members"] for b in causal["batches"]] == [[(0, 0), (1, 0)]]
    assert [b["members"] for b in legacy["batches"]] != [[(0, 0), (1, 0)]]


def test_arrivals_before_at_and_after_the_cutoff():
    # head arrives at 10+rtt/2=20; wait 50 -> cutoff 70.  B arrives exactly at 70 (included), C at 70.001 (not).
    specs = [Spec(0, 20.0, [10.0], [0.0]),
             Spec(1, 20.0, [50.0], [0.12]),          # arrival 60  (before)
             Spec(2, 20.0, [60.0], [0.2]),           # arrival 70  (at the cutoff: eligible)
             Spec(3, 20.0, [60.001], [0.0])]         # arrival 70.001 (after the cutoff)
    r = run(run_scenario(specs, causal=True))
    assert r["violations"] == []
    assert [b["members"] for b in r["batches"]] == [[(0, 0), (1, 0), (2, 0)], [(3, 0)]]


def test_batches_while_other_drafts_are_in_flight():
    specs = [Spec(0, 20.0, [10.0, 10.0, 10.0], [0.0]),
             Spec(1, 20.0, [400.0, 5.0, 5.0], [0.2, 0.0, 0.0])]   # one slow in virtual AND real time
    r = run(run_scenario(specs, causal=True))
    assert r["violations"] == []
    assert r["batches"] == [dict(b, members=b["members"]) for b in reference_batches(specs)]


def test_tied_timestamps_are_ordered_deterministically():
    specs = [Spec(i, 20.0, [10.0, 10.0], [d]) for i, d in zip(range(4), (0.12, 0.0, 0.06, 0.03))]
    outs = [run(run_scenario(specs, causal=True)) for _ in range(3)]
    ref = reference_batches(specs)
    for o in outs:
        assert o["violations"] == []
        assert o["batches"] == ref
    # exact ties: every client has the same arrival; client-id order breaks the tie, never real order
    first = outs[0]["ledger"].events
    arrivals = [e.detail["client"] for e in first if e.type == "arrival" and e.virtual_ts == 20.0]
    assert sorted(arrivals) == [0, 1, 2, 3]


def test_non_overlapping_verifier_service_and_no_drop_or_duplicate():
    specs = [Spec(i, 20.0 + 5 * i, [30.0 + 7 * i, 25.0, 40.0], [0.0, 0.02]) for i in range(6)]
    r = run(run_scenario(specs, causal=True, max_batch=3))
    assert r["violations"] == []
    served = [m for b in r["batches"] for m in b["members"]]
    assert len(served) == len(set(served)) == sum(len(s.draft_ms) for s in specs)
    for a, b in zip(r["batches"], r["batches"][1:]):
        assert a["finish"] <= b["start"] + 1e-9
    assert max(r["target_calls"]) <= 3


@pytest.mark.parametrize("seed", range(12))
def test_causal_matches_sequential_virtual_reference_for_random_real_delays(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 7)
    specs = [Spec(i, rng.choice([10.0, 20.0, 35.0, 60.0]),
                  [rng.choice([5.0, 20.0, 50.0, 120.0, 300.0]) + rng.random() for _ in range(3)],
                  [rng.random() * 0.05 for _ in range(3)]) for i in range(n)]
    mb = rng.choice([2, 3, 8])
    r = run(run_scenario(specs, causal=True, max_batch=mb))
    assert r["violations"] == []
    ref = reference_batches(specs, max_batch=mb)
    assert [b["members"] for b in r["batches"]] == [b["members"] for b in ref]
    for a, b in zip(r["batches"], ref):
        assert a["start"] == pytest.approx(b["start"]) and a["finish"] == pytest.approx(b["finish"])


def test_permuted_real_completion_order_gives_identical_virtual_outcome():
    base = [0.0, 0.03, 0.06, 0.09]
    outcomes = []
    for perm in itertools.islice(itertools.permutations(base), 8):
        delays = {i: [perm[i]] for i in range(4)}
        outcomes.append(canon(run(run_scenario(scenario(delays), causal=True))))
    assert all(o == outcomes[0] for o in outcomes[1:])


def test_deterministic_replay_same_trace_twice():
    specs = [Spec(i, 20.0 + 3 * i, [30.0 + 11 * i, 70.0, 20.0], [0.01 * (3 - i)]) for i in range(4)]
    a = canon(run(run_scenario(specs, causal=True)))
    b = canon(run(run_scenario(specs, causal=True)))
    assert a == b


# ---------------------------------------------------------------- D2 -------------------------------------------
def test_d2_decision_never_sees_a_future_server_update():
    # client 0 finishes a batch at f~100; client 1 decides at virtual t=5 but only after a real delay, so in
    # real order the batch has already been observed when its decision is evaluated.
    specs = [Spec(0, 20.0, [10.0, 10.0], [0.0]), Spec(1, 20.0, [10.0, 10.0], [0.0])]
    legacy = run(run_scenario(specs, causal=False, pre_decision_real_delay_s={1: 0.1}))
    causal = run(run_scenario(specs, causal=True))
    assert any(v.startswith("D2") for v in legacy["violations"]), legacy["violations"]
    assert causal["violations"] == []
    for d in causal["decisions"]:
        assert d["visible_ts"] <= d["t"] + 1e-9


def test_controller_snapshot_history_is_a_function_of_virtual_time():
    c = OnlineController(gamma_choices=[0, 1], V=50.0, min_tps=3.0, server_price_scale=1.0, policy="fixed_gamma")
    assert c.server_state_at(1e9) == (0.0, 0.0, 0.0, 0.0, 0)
    c.observe_server_batch(0, verify_finish_ms=100.0, theta_f_ms_per_token=40.0, verifier_token_cost=10)   # +400ms
    q100 = c.server_queue
    c.observe_server_batch(1, verify_finish_ms=350.0, theta_f_ms_per_token=40.0, verifier_token_cost=10)
    assert c.server_state_at(99.9)[4] == 0
    assert c.server_state_at(100.0)[0] == pytest.approx(q100) and c.server_state_at(100.0)[3] == 100.0
    assert c.server_state_at(349.9)[3] == 100.0                      # the batch finishing at 350 is invisible
    assert c.server_state_at(350.0)[3] == 350.0 and c.server_state_at(350.0)[4] == 2
    visible_ts, version = c.load_server_state_view(120.0)
    assert (visible_ts, version) == (100.0, 1) and c.server_queue == pytest.approx(q100)


def test_stamped_view_is_not_modified_by_later_observations():
    c = OnlineController(gamma_choices=[0, 1], V=50.0, min_tps=3.0, server_price_scale=1.0, policy="fixed_gamma")
    c.observe_server_batch(0, 100.0, 40.0, 10)
    c.load_server_state_view(150.0)
    seen = c.server_queue
    c.observe_server_batch(1, 900.0, 40.0, 10)       # published later in real time, later in virtual time
    assert c.server_state_at(150.0)[0] == pytest.approx(seen)


# ---------------------------------------------------------------- guards ---------------------------------------
def test_lookahead_violation_is_detected():
    coord = VirtualCoordinator(50.0)
    coord.register_client(0, 20.0)
    coord.begin_round(0, 100.0)             # promises arrival >= 110
    with pytest.raises(CausalityViolation):
        coord.note_submit(0, 0, 105.0)


def test_degenerate_lookahead_is_rejected():
    with pytest.raises(ValueError):
        VirtualCoordinator(0.0).register_client(0, 0.0)


def test_deadlock_is_detected_instead_of_hanging():
    async def go():
        coord = VirtualCoordinator(1.0)
        coord.register_client(0, 2.0)
        coord.register_client(1, 2.0)
        orig = coord.begin_round
        coord.begin_round = lambda cid, t: None      # sabotage: promise nothing -> horizon never advances
        # both clients wait for a horizon that only their own (unpromised) bounds could raise
        coord.clients[0].lb = coord.clients[1].lb = 0.0
        await asyncio.wait_for(asyncio.gather(coord.wait_decision(0, 10.0), coord.wait_decision(1, 12.0)), 3)

    with pytest.raises(CausalityDeadlock):
        run(go())


def test_no_real_time_sleep_is_used_for_correctness():
    import inspect
    import edge_specsim.verification_batcher as vb
    import edge_specsim.virtual_causality as vc
    causal_src = inspect.getsource(vb.VerificationBatcher._loop_causal) + inspect.getsource(vc)
    assert "asyncio.sleep" not in causal_src and "time.sleep" not in causal_src
