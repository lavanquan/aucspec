"""CPU-only characterization tests for the verifier-price signal path.

These tests document CURRENT behaviour (no new pricing algorithm is enabled):

* fixed-budget mode hands the controller theta_f = 0.0, so the verifier price
  terms in depth selection and knapsack values vanish even when the server
  virtual queue Q_s is positive;
* a synthetic positive theta_f (what a future estimator would supply) does
  change depth scores and knapsack item values the way the formulas say;
* the theta_f recorded in batch metadata (batch service / logical tokens) is a
  different quantity from the price coefficient the controller used.
"""

from __future__ import annotations

import asyncio

import pytest

from edge_specsim.verification_batcher import VerificationBatcher
from edge_specsim.verification_queue import VerificationQueue
from test_capacity import _client, _ctrl
from test_capacity_knapsack import _req
from test_verifier_service_time import MockTarget, _draft


def _batcher(**kw):
    return VerificationBatcher(MockTarget({1: 10.0}), verify_token_budget=100, **kw)


# ------------------------------------------------------- zero price in fixed mode
def test_fixed_budget_signal_is_zero_price():
    budget, regime, theta_f = _batcher().estimate_verify_budget_signal(context_length=300)
    assert (budget, regime, theta_f) == (100, "memory_bound", 0.0)


def test_simulator_wiring_sets_controller_theta_f_to_zero():
    ctrl = _ctrl()
    assert ctrl.current_verifier_theta_f_ms_per_token == 1.0  # constructor default, never used in fixed mode
    _, _, theta_f = _batcher().estimate_verify_budget_signal(context_length=300)
    ctrl.set_verifier_theta_f_signal(theta_f)  # exactly what simulator._run_round does
    assert ctrl.current_verifier_theta_f_ms_per_token == 0.0


def test_server_queue_positive_but_verifier_penalty_zero():
    ctrl = _ctrl()
    ctrl.set_verifier_theta_f_signal(0.0)
    # a 250 ms batch observed in a 100 ms slot drives Q_s positive (batch metadata theta_f is nonzero)
    ctrl.observe_server_batch(batch_id=0, verify_finish_ms=250.0, theta_f_ms_per_token=25.0, verifier_token_cost=10)
    assert ctrl.server_queue == pytest.approx(150.0)
    client = _client(z_queue=2.0, device_queue=0.0, alpha=0.85)
    baseline = _ctrl()
    baseline.set_verifier_theta_f_signal(0.0)
    for gamma in (0, 1, 2, 4):
        prof = client.positional_acceptance_profile(gamma, use_ucb=True) if gamma else []
        assert ctrl.capacity_dpp_score(client, gamma, prof) == pytest.approx(
            baseline.capacity_dpp_score(client, gamma, prof)
        )  # queue state has no effect on depth scores
    assert ctrl._capacity_dpp_gamma(client, use_ucb=True) == baseline._capacity_dpp_gamma(client, use_ucb=True)


def test_zero_price_request_value_ignores_queue():
    r = _req(0, gamma=4, weight=10.0, acceptance=0.8, server_queue_price=150.0, theta_f_ms_per_token=0.0)
    r0 = _req(0, gamma=4, weight=10.0, acceptance=0.8, server_queue_price=0.0, theta_f_ms_per_token=0.0)
    assert r.capacity_dpp_value == pytest.approx(r0.capacity_dpp_value)


# ------------------------------------------------ synthetic positive signal works
def test_positive_signal_lowers_depth_score_by_formula():
    client = _client(z_queue=2.0, device_queue=0.0, alpha=0.85)
    ctrl = _ctrl()
    ctrl.server_queue = 3.0
    ctrl.set_verifier_theta_f_signal(0.0)
    prof = client.positional_acceptance_profile(3, use_ucb=True)
    s0 = ctrl.capacity_dpp_score(client, 3, prof)
    ctrl.set_verifier_theta_f_signal(0.5)
    s1 = ctrl.capacity_dpp_score(client, 3, prof)
    assert s0 - s1 == pytest.approx(3.0 * 0.5 * (3 + 1))  # Q_s * theta_f * (gamma + 1)


def test_positive_signal_reduces_selected_depth():
    client = _client(z_queue=0.0, device_queue=0.0, alpha=0.85)
    ctrl = _ctrl(V=5.0)
    ctrl.server_queue = 4.0
    ctrl.set_verifier_theta_f_signal(0.0)
    deep = ctrl._capacity_dpp_gamma(client, use_ucb=True)
    ctrl.set_verifier_theta_f_signal(5.0)
    shallow = ctrl._capacity_dpp_gamma(client, use_ucb=True)
    assert shallow < deep


def test_positive_signal_changes_knapsack_item_values_and_selection():
    free = [_req(i, gamma=g, weight=10.0, acceptance=0.8) for i, g in enumerate((4, 1))]
    priced = [
        _req(i, gamma=g, weight=10.0, acceptance=0.8, server_queue_price=8.0, theta_f_ms_per_token=1.0)
        for i, g in enumerate((4, 1))
    ]
    for a, b in zip(free, priced):
        assert b.capacity_dpp_value == pytest.approx(a.capacity_dpp_value - 8.0 * 1.0 * a.verifier_token_cost)

    def pick(reqs):
        q = VerificationQueue(random_seed=0)
        for r in reqs:
            q.push(r)
        return sorted(r.client_id for r in q.pop_batch(
            scheduler_name="capacity_knapsack", max_batch_size=1, batch_wait_ms=10.0,
            verify_token_budget=100, utility_lambda=0.0))

    assert pick(free) == [0]      # deep draft has the larger value without a price
    assert pick(priced) == [1]    # price on verifier tokens favours the shallow draft


# ------------------------------------------------------------------- units
def test_metadata_theta_f_is_batch_average_not_the_controller_price():
    target = MockTarget({1: 120.0})

    async def go():
        b = VerificationBatcher(target, service_time_model="batch_wall", verify_token_budget=100)
        await b.start()
        # the simulator would stamp the controller's current (zero) price on the request
        r = await b.submit(0, list(range(10)), _draft(0, 3), 50, 0.0, server_queue_price=7.0, theta_f_ms_per_token=0.0)
        await b.close()
        return r

    meta, _ = asyncio.run(go())
    assert meta.batch_token_cost == 4  # logical positions gamma + 1
    assert meta.modeled_batch_service_ms == pytest.approx(120.0)
    # unit identity used for the Q_s update: theta_f[ms/token] * logical tokens = batch service [ms]
    assert meta.theta_f_ms_per_token * meta.batch_token_cost == pytest.approx(meta.compute_bound_batch_service_ms)
    assert meta.theta_f_ms_per_token == pytest.approx(30.0)
    # ... which is NOT the zero coefficient the controller priced with, and not a marginal cost
    assert meta.batch_processed_tokens == 10 + 3  # what the GPU actually processed
    assert meta.batch_processed_tokens != meta.batch_token_cost


def test_queue_update_uses_service_ms_and_slot_ms_units():
    ctrl = _ctrl()
    ctrl.observe_server_batch(0, verify_finish_ms=50.0, theta_f_ms_per_token=10.0, verifier_token_cost=4)
    assert ctrl.server_queue == 0.0  # 40 ms of work in a 100 ms slot
    ctrl.observe_server_batch(1, verify_finish_ms=60.0, theta_f_ms_per_token=10.0, verifier_token_cost=14)
    assert ctrl.server_queue == pytest.approx(80.0)  # (40 + 140) ms - 100 ms slot budget


def test_max_finish_diagnostic_does_not_change_queue_state():
    ctrl = _ctrl()
    assert ctrl.server_state_max_finish_ms == 0.0
    ctrl.observe_server_batch(0, verify_finish_ms=250.0, theta_f_ms_per_token=25.0, verifier_token_cost=10)
    ctrl.observe_server_batch(1, verify_finish_ms=180.0, theta_f_ms_per_token=1.0, verifier_token_cost=1)  # out of order
    assert ctrl.server_state_max_finish_ms == 250.0
    ref = _ctrl()
    ref.observe_server_batch(0, verify_finish_ms=250.0, theta_f_ms_per_token=25.0, verifier_token_cost=10)
    ref.observe_server_batch(1, verify_finish_ms=180.0, theta_f_ms_per_token=1.0, verifier_token_cost=1)
    assert ctrl.server_queue == ref.server_queue
    off = _ctrl(use_virtual_queues=False)
    off.observe_server_batch(0, verify_finish_ms=90.0, theta_f_ms_per_token=1.0, verifier_token_cost=1)
    assert off.server_state_max_finish_ms == 90.0 and off.server_queue == 0.0
