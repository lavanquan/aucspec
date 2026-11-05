"""capacity_dpp pricing variants: point acceptance estimate and the
quadratic drift term (controller.capacity_dpp.*)."""
from __future__ import annotations

import pytest

from tests.test_capacity import _client, _ctrl


def _gamma(ctrl, cl):
    return ctrl._capacity_dpp_gamma(
        cl, use_ucb=ctrl.capacity_dpp_acceptance_estimate == "ucb"
    )


def test_defaults_keep_original_score():
    ctrl = _ctrl(V=3000.0)
    assert ctrl.capacity_dpp_quadratic_drift_weight == 0.0
    assert ctrl.capacity_dpp_acceptance_estimate == "ucb"
    ctrl.server_queue = 500.0
    ctrl.current_verifier_theta_f_ms_per_token = 3.83
    ctrl.current_active_clients = 48
    cl = _client(alpha=0.9)
    prof = cl.positional_acceptance_profile(4, use_ucb=True)
    phi = 1 + 0.9 + 0.9**2 + 0.9**3 + 0.9**4
    expected = 3000.0 * phi - 500.0 * 3.83 * 5
    assert ctrl.capacity_dpp_score(cl, 4, prof) == pytest.approx(expected)


def test_quadratic_drift_term_value():
    ctrl = _ctrl(V=3000.0)
    ctrl.capacity_dpp_quadratic_drift_weight = 1.0
    ctrl.server_queue = 0.0
    ctrl.current_verifier_theta_f_ms_per_token = 2.0
    ctrl.current_active_clients = 10
    cl = _client(alpha=1.0 - 1e-9)
    prof = cl.positional_acceptance_profile(2, use_ucb=True)
    # 0.5 * w * N * (theta_f * (gamma+1))^2 = 0.5 * 10 * 36
    assert ctrl.capacity_dpp_score(cl, 2, prof) == pytest.approx(3000.0 * 3 - 180.0, rel=1e-6)


def test_linear_price_with_unit_acceptance_is_bang_bang():
    # alpha ~ 1 => phi linear in gamma; linear price => only 0 or gamma_max
    ctrl = _ctrl(V=3000.0)
    ctrl.current_verifier_theta_f_ms_per_token = 3.83
    cl = _client(alpha=0.999)
    picks = set()
    for q in range(0, 2000, 25):
        ctrl.server_queue = float(q)
        picks.add(_gamma(ctrl, cl))
    assert picks <= {0, 4}


def test_quadratic_drift_gives_interior_gamma():
    ctrl = _ctrl(V=3000.0)
    ctrl.capacity_dpp_quadratic_drift_weight = 1.0
    ctrl.current_verifier_theta_f_ms_per_token = 3.83
    ctrl.current_active_clients = 48
    cl = _client(alpha=0.999)
    picks = set()
    for q in range(0, 2000, 25):
        ctrl.server_queue = float(q)
        picks.add(_gamma(ctrl, cl))
    assert picks & {1, 2, 3}


def test_quadratic_drift_gamma_nonincreasing_in_population():
    cl = _client(alpha=0.9)
    prev = None
    for n in (1, 8, 16, 32, 48, 96):
        ctrl = _ctrl(V=3000.0)
        ctrl.capacity_dpp_quadratic_drift_weight = 1.0
        ctrl.current_verifier_theta_f_ms_per_token = 3.83
        ctrl.server_queue = 200.0
        ctrl.current_active_clients = n
        g = _gamma(ctrl, cl)
        if prev is not None:
            assert g <= prev
        prev = g
    assert prev < 4


def test_point_estimate_uses_alpha_hat():
    ctrl = _ctrl(V=3000.0)
    ctrl.capacity_dpp_acceptance_estimate = "point"
    ctrl.current_verifier_theta_f_ms_per_token = 3.83
    ctrl.server_queue = 600.0
    cl = _client(alpha=0.8)
    cl.alpha_ucb = 0.999
    # alpha_ucb ~1 -> gamma_max under ucb; alpha_hat 0.8 -> shallower
    ctrl_ucb = _ctrl(V=3000.0)
    ctrl_ucb.current_verifier_theta_f_ms_per_token = 3.83
    ctrl_ucb.server_queue = 600.0
    assert _gamma(ctrl_ucb, cl) == 4
    assert _gamma(ctrl, cl) < 4


def _n48_state(ctrl):
    # decision state seen at N48 with causal_release_bound (job 52141649)
    ctrl.capacity_dpp_quadratic_drift_weight = 1.0
    ctrl.current_verifier_theta_f_ms_per_token = 3.83
    ctrl.current_active_clients = 48
    ctrl.server_queue = 630.0


def test_z_weight_default_is_original():
    ctrl = _ctrl(V=3000.0)
    assert ctrl.capacity_dpp_z_weight == 1.0
    cl = _client(alpha=0.8)
    cl.z_queue = 3.0
    assert ctrl.capacity_dpp_weighted_z(cl) == pytest.approx(3.0)


def test_small_deficit_cannot_move_gamma_at_unit_weight():
    ctrl = _ctrl(V=3000.0)
    _n48_state(ctrl)
    cl = _client(alpha=0.8)
    cl.z_queue = 0.0
    g0 = _gamma(ctrl, cl)
    cl.z_queue = 5.0  # ~15 s at 0.33 tok/s short of x
    assert _gamma(ctrl, cl) == g0 == 0


def test_z_weight_lets_lagging_client_speculate():
    ctrl = _ctrl(V=3000.0)
    _n48_state(ctrl)
    ctrl.capacity_dpp_z_weight = 500.0
    cl = _client(alpha=0.8)
    cl.z_queue = 0.0
    assert _gamma(ctrl, cl) == 0
    cl.z_queue = 5.0
    assert _gamma(ctrl, cl) >= 1
