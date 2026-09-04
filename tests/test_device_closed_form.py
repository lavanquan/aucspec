"""T1.5: verify eq. (2), (4), (10) closed-form matches Monte Carlo simulation
for the simple case (1 stream, gamma fixed), error < 1%.
"""

import pytest

from sim.core.acceptance import expected_tokens_per_round
from sim.core.device import (
    SingleStreamSimulator,
    round_cycle_seconds,
    symmetric_cycle_seconds,
    symmetric_operating_point,
)
from sim.core.server import RooflineParams, verification_latency


def make_roofline() -> RooflineParams:
    return RooflineParams(
        theta_0_seconds=0.003,
        theta_f_seconds_per_token=0.00008,
        beta_mem_bytes_per_second=2.0e12,
        b_w_bytes=1.5e10,
        b_kv_bytes_per_token=1.0e5,
    )


@pytest.mark.parametrize("gamma,alpha", [(0, 0.8), (3, 0.85), (6, 0.6)])
def test_single_stream_interactivity_matches_eq4_within_1pct(gamma, alpha):
    roofline = make_roofline()
    tau_d = 0.010
    kappa = 1500.0
    r = 5.0e6
    delta = 0.015
    context_length = 512.0

    sim = SingleStreamSimulator(
        gamma=gamma,
        alpha=alpha,
        tau_d_seconds=tau_d,
        kappa_bits_per_token=kappa,
        uplink_rate_bps=r,
        delta_seconds=delta,
        roofline=roofline,
        context_length=context_length,
        seed=123,
    )
    result = sim.run(num_rounds=200_000, warmup_rounds=1_000)

    t_v = verification_latency(roofline, [gamma], [context_length])
    cycle_seconds = round_cycle_seconds(
        gamma, tau_d, kappa, r, queueing_delay_seconds=0.0, verification_latency_seconds=t_v, delta_seconds=delta
    )
    closed_form_x = expected_tokens_per_round(gamma, alpha) / cycle_seconds  # eq. (2) over eq. (4)'s denominator

    assert result["cycle_seconds"] == pytest.approx(cycle_seconds)
    relative_error = abs(result["interactivity_tokens_per_second"] - closed_form_x) / closed_form_x
    assert relative_error < 0.01, f"relative_error={relative_error:.4f} exceeds 1% (gamma={gamma}, alpha={alpha})"


def test_symmetric_operating_point_matches_manual_eq10():
    roofline = make_roofline()
    batch_size = 8
    gamma = 4
    alpha = 0.75
    tau_d = 0.008
    kappa = 1200.0
    r = 4.0e6
    delta = 0.012

    cycle_seconds = symmetric_cycle_seconds(batch_size, gamma, tau_d, kappa, r, delta, roofline)
    expected_cycle = (
        gamma * (tau_d + kappa / r)
        + delta
        + roofline.theta_0_seconds
        + max(
            roofline.b_w_bytes / roofline.beta_mem_bytes_per_second,
            roofline.theta_f_seconds_per_token * batch_size * (gamma + 1),
        )
    )
    assert cycle_seconds == pytest.approx(expected_cycle)

    x, y = symmetric_operating_point(batch_size, gamma, alpha, tau_d, kappa, r, delta, roofline)
    expected_x = expected_tokens_per_round(gamma, alpha) / expected_cycle
    assert x == pytest.approx(expected_x)
    assert y == pytest.approx(batch_size * expected_x)


def test_symmetric_frontier_gamma_zero_is_compute_bound_baseline():
    # With gamma=0 nothing is drafted: phi(0,alpha)=1, so x = 1/T(B,0) regardless of alpha.
    roofline = make_roofline()
    x_low_alpha, _ = symmetric_operating_point(4, 0, 0.1, 0.01, 1000.0, 1.0e6, 0.01, roofline)
    x_high_alpha, _ = symmetric_operating_point(4, 0, 0.99, 0.01, 1000.0, 1.0e6, 0.01, roofline)
    assert x_low_alpha == pytest.approx(x_high_alpha)
