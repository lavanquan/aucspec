import math

import pytest

from sim.controllers.auc_controller import (
    AUCController,
    PendingVerification,
    StreamLinkState,
    UCBBernsteinEstimator,
    price_c,
    radio_allocation,
    select_batch,
    speculation_index,
)


def test_price_c_matches_formula():
    c = price_c(
        lambda_price=2.0,
        mu_price=1.0,
        theta_f_seconds_per_token=0.1,
        tau_d_seconds=0.01,
        kappa_bits_per_token=1000.0,
        uplink_rate_bps=1.0e6,
        V=10.0,
        z_i=5.0,
    )
    expected = (2.0 * 0.1 + 1.0 * (0.01 + 1000.0 / 1.0e6)) / (10.0 + 5.0)
    assert c == pytest.approx(expected)


def test_price_c_requires_positive_denominator():
    with pytest.raises(ValueError):
        price_c(0.0, 0.0, 0.1, 0.01, 1000.0, 1.0e6, V=1.0, z_i=-1.0)


def test_speculation_index_zero_cost_returns_gamma_max():
    assert speculation_index(c_i=0.0, alpha_hat_i=0.8, gamma_max=8) == 8


def test_speculation_index_high_cost_returns_zero():
    assert speculation_index(c_i=1.5, alpha_hat_i=0.8, gamma_max=8) == 0


def test_speculation_index_matches_closed_form():
    c = 0.05
    alpha = 0.85
    expected = max(0, min(8, math.ceil(math.log(c) / math.log(alpha)) - 1))
    assert speculation_index(c, alpha, gamma_max=8) == expected


def test_speculation_index_monotone_in_alpha():
    # Higher acceptance rate should never yield a smaller speculation index.
    low = speculation_index(c_i=0.1, alpha_hat_i=0.5, gamma_max=10)
    high = speculation_index(c_i=0.1, alpha_hat_i=0.95, gamma_max=10)
    assert high >= low


def test_select_batch_greedy_by_weight_per_token():
    pending = [
        PendingVerification(stream_index=0, gamma=3, weight=8.0),  # 8/4 = 2.0
        PendingVerification(stream_index=1, gamma=1, weight=3.0),  # 3/2 = 1.5
        PendingVerification(stream_index=2, gamma=0, weight=0.9),  # 0.9/1 = 0.9
    ]
    # Budget only fits the top-ranked item (cost 4) plus the next (cost 2) = 6.
    selected = select_batch(pending, gamma_budget=6.0)
    assert selected == [0, 1]


def test_select_batch_respects_budget_exactly():
    pending = [PendingVerification(i, gamma=1, weight=1.0) for i in range(5)]  # cost 2 each
    selected = select_batch(pending, gamma_budget=5.0)
    assert len(selected) == 2  # floor(5/2)


def test_radio_allocation_matches_waterfilling():
    mu = [1.0, 4.0]
    gammas = [2, 2]
    g = [1.0, 1.0]
    kappa = 10.0
    allocation = radio_allocation(mu, kappa, gammas, g, total_bandwidth_hz=100.0)
    assert sum(allocation) == pytest.approx(100.0)
    assert allocation[1] > allocation[0]


def test_ucb_estimator_defaults_to_alpha_max_with_no_data():
    est = UCBBernsteinEstimator(n_streams=1, alpha_min=0.5, alpha_max=0.95)
    assert est.estimate(0, t=0) == 0.95


def test_ucb_estimator_converges_toward_true_alpha():
    import random

    rng = random.Random(0)
    true_alpha = 0.8
    est = UCBBernsteinEstimator(n_streams=1, alpha_min=0.01, alpha_max=0.99)
    for t in range(2000):
        # Simulate a round with gamma=4 drafts under Bernoulli(true_alpha).
        accepted = 0
        for _ in range(4):
            if rng.random() < true_alpha:
                accepted += 1
            else:
                break
        est.observe(0, accepted, accepted == 4)
    # UCB estimate is optimistic (an upper bound), so it should sit at or
    # above the true rate but converge close to it with enough data.
    estimate = est.estimate(0, t=2000)
    assert true_alpha <= estimate <= true_alpha + 0.1


def test_ucb_estimator_clamps_to_alpha_bounds():
    est = UCBBernsteinEstimator(n_streams=1, alpha_min=0.1, alpha_max=0.6)
    for _ in range(50):
        est.observe(0, accepted=1, all_accepted=True)
    assert est.estimate(0, t=50) <= 0.6


def make_controller(V: float, alpha_estimator) -> AUCController:
    return AUCController(
        n_streams=2,
        gamma_max=8,
        V=V,
        min_interactivity_x=0.0,
        kappa_bits_per_token=1000.0,
        theta_f_seconds_per_token=0.0001,
        total_bandwidth_hz=1.0e6,
        gamma_budget=1000.0,  # generous: never the bottleneck in this test
        alpha_estimator=alpha_estimator,
    )


class _ConstantAlpha:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def estimate(self, stream_index: int, t: int) -> float:
        return self.alpha


def test_decide_returns_one_gamma_and_bandwidth_share_per_stream():
    controller = make_controller(V=10.0, alpha_estimator=_ConstantAlpha(0.8))
    links = [
        StreamLinkState(tau_d_seconds=0.01, uplink_rate_bps=1.0e6, spectral_efficiency=3.0),
        StreamLinkState(tau_d_seconds=0.01, uplink_rate_bps=1.0e6, spectral_efficiency=3.0),
    ]
    decision = controller.decide(t=0, links=links)
    assert len(decision.gammas) == 2
    assert len(decision.bandwidth_allocation_hz) == 2
    assert sum(decision.bandwidth_allocation_hz) == pytest.approx(1.0e6)
    assert set(decision.selected_batch) <= {0, 1}


def test_decide_rejects_wrong_number_of_links():
    controller = make_controller(V=10.0, alpha_estimator=_ConstantAlpha(0.8))
    with pytest.raises(ValueError):
        controller.decide(t=0, links=[StreamLinkState(0.01, 1.0e6, 3.0)])


def test_update_queues_reduces_z_when_target_is_met():
    controller = make_controller(V=10.0, alpha_estimator=_ConstantAlpha(0.8))
    links = [StreamLinkState(0.01, 1.0e6, 3.0), StreamLinkState(0.01, 1.0e6, 3.0)]
    decision = controller.decide(t=0, links=links)
    before = list(controller.queues.Z)
    controller.update_queues(
        decision,
        tokens_delivered=[10.0, 10.0],  # plenty of tokens delivered
        links=links,
        verification_batch_tokens=10.0,
        slot_seconds=1.0,
    )
    assert all(after <= b for after, b in zip(controller.queues.Z, before))
