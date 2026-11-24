"""SLO-aware baseline (controller.fixed_slo_mode=rate_floor) and the VTC
batch scheduler (verification_batching.scheduler=vtc)."""
from __future__ import annotations

from types import SimpleNamespace

from edge_specsim.verification_queue import VerificationQueue
from tests.test_capacity import _client, _ctrl


# ------------------------------------------------------------ fixed_slo ----
def _slo(mode: str, x: float):
    ctrl = _ctrl(V=50.0)
    ctrl.policy = "fixed_slo"
    ctrl.min_tps = x
    ctrl.fixed_slo_mode = mode
    ctrl.current_verifier_theta_f_ms_per_token = 2.15
    return ctrl


def _rates(ctrl, cl):
    return {g: 1000.0 * ctrl._turbospec_goodput(cl, g) for g in ctrl.gamma_choices}


def test_default_mode_is_round_time():
    ctrl = _ctrl()
    assert ctrl.fixed_slo_mode == "round_time"


def test_rate_floor_picks_largest_gamma_meeting_x():
    cl = _client(alpha=0.9)
    probe = _slo("rate_floor", 0.0)
    r = _rates(probe, cl)
    x = sorted(r.values())[1]  # met by all but the slowest gamma
    ctrl = _slo("rate_floor", x)
    g = ctrl._fixed_slo_gamma(cl)
    assert r[g] >= x
    assert g == max(k for k, v in r.items() if v >= x)


def test_rate_floor_falls_back_to_max_rate_gamma():
    cl = _client(alpha=0.9)
    ctrl = _slo("rate_floor", 1e9)  # nothing meets x
    r = _rates(ctrl, cl)
    assert ctrl._fixed_slo_gamma(cl) == max(r, key=r.get)


def test_round_time_mode_unchanged():
    cl = _client(alpha=0.9)
    ctrl = _slo("round_time", 5.0)
    ctrl.fixed_slo_ms = 1e9
    assert ctrl._fixed_slo_gamma(cl) == max(ctrl.gamma_choices)
    ctrl.fixed_slo_ms = 0.0
    assert ctrl._fixed_slo_gamma(cl) == 0


def test_exploration_covers_fixed_slo_only_with_flag():
    for flag, expect in ((False, 0), (True, 1)):
        ctrl = _slo("rate_floor", 5.0)
        ctrl.min_exploration_rounds = 2
        ctrl.exploration_gamma = 1
        ctrl.turbospec_exploration = flag
        cl = _client(alpha=0.9)
        cl.speculative_rounds = 0
        assert ctrl._apply_exploration(cl, 0) == expect


# ------------------------------------------------------------------ vtc ----
def _req(cid: int, gamma: int, t: float):
    return SimpleNamespace(client_id=cid, gamma=gamma, verifier_token_cost=max(1, gamma + 1), arrival_time=t)


def _pop(q, B=32, G=100):
    return q.pop_batch("vtc", max_batch_size=B, batch_wait_ms=1e9, verify_token_budget=G, utility_lambda=0.0)


def test_vtc_lowest_counter_first_and_charges_gamma_plus_one():
    q = VerificationQueue()
    q.vtc_counter = {0: 10.0, 1: 0.0, 2: 5.0}
    q.extend([_req(0, 4, 0.0), _req(1, 4, 1.0), _req(2, 4, 2.0)])
    got = _pop(q, B=2)
    assert [r.client_id for r in got] == [1, 2]
    assert q.vtc_counter[1] == 5.0 and q.vtc_counter[2] == 10.0
    assert [r.client_id for r in q.waiting] == [0]


def test_vtc_respects_token_budget_and_batch_size():
    q = VerificationQueue()
    q.extend([_req(i, 4, float(i)) for i in range(30)])
    got = _pop(q, B=32, G=100)
    assert len(got) == 20 and sum(r.verifier_token_cost for r in got) == 100
    q2 = VerificationQueue()
    q2.extend([_req(i, 0, float(i)) for i in range(40)])
    assert len(_pop(q2, B=32, G=100)) == 32


def test_vtc_new_client_starts_at_backlog_minimum_no_relift():
    q = VerificationQueue()
    q.vtc_counter = {0: 50.0, 1: 2.0}
    q.push(_req(0, 1, 0.0))
    q.push(_req(1, 1, 1.0))  # known client: keeps its deficit (no lift)
    assert q.vtc_counter[1] == 2.0
    q.push(_req(7, 1, 2.0))  # new client: starts at min backlogged counter
    assert q.vtc_counter[7] == 2.0


def test_vtc_long_run_share_is_fair():
    # two backlogged clients with different request sizes: served verifier
    # tokens stay within one request of each other (VTC's bounded gap)
    q = VerificationQueue()
    served = {0: 0, 1: 0}
    t = 0.0
    q.extend([_req(0, 4, t), _req(1, 1, t)])
    for _ in range(200):
        got = _pop(q, B=1, G=100)
        for r in got:
            served[r.client_id] += r.verifier_token_cost
            t += 1.0
            q.push(_req(r.client_id, r.gamma, t))
    assert abs(served[0] - served[1]) <= 5


def test_other_schedulers_unaffected():
    q = VerificationQueue()
    q.extend([_req(i, 4, float(i)) for i in range(5)])
    got = q.pop_batch("fcfs", max_batch_size=32, batch_wait_ms=1e9, verify_token_budget=100, utility_lambda=0.0)
    assert [r.client_id for r in got] == [0, 1, 2, 3, 4]
