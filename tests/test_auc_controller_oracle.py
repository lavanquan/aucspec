import pytest

from sim.controllers.auc_controller import StreamLinkState
from sim.controllers.auc_controller_oracle import (
    StaticAlphaEstimator,
    build_oracle_controller,
    profile_alpha_from_trace,
)


def test_static_alpha_estimator_is_constant_over_t():
    est = StaticAlphaEstimator(alpha_hat=[0.7, 0.9])
    assert est.estimate(0, t=0) == 0.7
    assert est.estimate(0, t=99999) == 0.7
    assert est.estimate(1, t=5) == 0.9


def test_profile_alpha_uses_only_first_half_of_trace():
    # Stream 0: first half all True (alpha_hat=1.0), second half all False.
    stream0 = [True] * 100 + [False] * 100
    estimates = profile_alpha_from_trace([stream0], split_fraction=0.5)
    assert estimates[0] == pytest.approx(1.0)  # must ignore the second half


def test_profile_alpha_rejects_bad_split_fraction():
    with pytest.raises(ValueError):
        profile_alpha_from_trace([[True, False]], split_fraction=1.5)


def test_profile_alpha_rejects_empty_window():
    with pytest.raises(ValueError):
        profile_alpha_from_trace([[]], split_fraction=0.5)


def test_build_oracle_controller_uses_static_estimator_not_ucb():
    controller = build_oracle_controller(
        alpha_hat=[0.6, 0.8],
        n_streams=2,
        gamma_max=8,
        V=10.0,
        min_interactivity_x=0.0,
        kappa_bits_per_token=1000.0,
        theta_f_seconds_per_token=0.0001,
        total_bandwidth_hz=1.0e6,
        gamma_budget=1000.0,
    )
    assert isinstance(controller.alpha_estimator, StaticAlphaEstimator)
    links = [StreamLinkState(0.01, 1.0e6, 3.0), StreamLinkState(0.01, 1.0e6, 3.0)]
    # Same decision at t=0 and t=999: no learning, no time-dependence.
    d0 = controller.decide(t=0, links=links)
    d999 = controller.decide(t=999, links=links)
    assert d0.gammas == d999.gammas
