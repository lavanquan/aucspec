import pytest

from sim.controllers.auc_controller import StreamLinkState
from sim.controllers.turbospec import MovingAverageAcceptance, TurboSpecController
from sim.core.server import RooflineParams


def make_roofline() -> RooflineParams:
    return RooflineParams(
        theta_0_seconds=0.002,
        theta_f_seconds_per_token=0.00005,
        beta_mem_bytes_per_second=2.0e12,
        b_w_bytes=1.0e10,
        b_kv_bytes_per_token=0.0,
    )


def test_moving_average_tracks_observed_rate():
    ma = MovingAverageAcceptance(decay=0.9, alpha_hat=0.5)
    for _ in range(500):
        ma.observe(accepted=4, gamma=4)  # empirical rate 1.0 every round
    assert ma.alpha_hat == pytest.approx(1.0, abs=1e-3)


def test_moving_average_ignores_gamma_zero_rounds():
    ma = MovingAverageAcceptance(decay=0.9, alpha_hat=0.5)
    ma.observe(accepted=0, gamma=0)
    assert ma.alpha_hat == 0.5  # unchanged


def test_decide_picks_same_k_for_every_stream():
    ctrl = TurboSpecController(
        n_streams=3,
        gamma_max=6,
        tau_d_seconds=0.005,
        kappa_bits_per_token=500.0,
        uplink_rate_bps=2.0e6,
        delta_seconds=0.01,
        roofline=make_roofline(),
        total_bandwidth_hz=1.0e6,
    )
    links = [StreamLinkState(0.005, 2.0e6, 3.0)] * 3
    decision = ctrl.decide(t=0, links=links)
    assert len(set(decision.gammas)) == 1  # one global k for the whole batch
    assert decision.selected_batch == [0, 1, 2]


def test_higher_alpha_never_yields_a_lower_k():
    low_alpha = TurboSpecController(
        n_streams=2, gamma_max=8, tau_d_seconds=0.005, kappa_bits_per_token=500.0,
        uplink_rate_bps=2.0e6, delta_seconds=0.01, roofline=make_roofline(), total_bandwidth_hz=1.0e6,
        acceptance=MovingAverageAcceptance(alpha_hat=0.3),
    )
    high_alpha = TurboSpecController(
        n_streams=2, gamma_max=8, tau_d_seconds=0.005, kappa_bits_per_token=500.0,
        uplink_rate_bps=2.0e6, delta_seconds=0.01, roofline=make_roofline(), total_bandwidth_hz=1.0e6,
        acceptance=MovingAverageAcceptance(alpha_hat=0.95),
    )
    links = [StreamLinkState(0.005, 2.0e6, 3.0)] * 2
    k_low = low_alpha.decide(t=0, links=links).gammas[0]
    k_high = high_alpha.decide(t=0, links=links).gammas[0]
    assert k_high >= k_low
