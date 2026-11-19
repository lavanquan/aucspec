"""capacity_dpp.benefit_floor, load_price and z_load_damping (all off by default)."""
from __future__ import annotations

from tests.test_capacity import _client, _ctrl


def _gamma(ctrl, cl):
    return ctrl._capacity_dpp_gamma(cl, use_ucb=True)


def _over_capacity(floor: float):
    """Every client lags; the fast one has Z far below the mean."""
    ctrl = _ctrl(V=1100.0)
    ctrl.capacity_dpp_quadratic_drift_weight = 1.0
    ctrl.capacity_dpp_z_weight = 500.0
    ctrl.capacity_dpp_z_relative = True
    ctrl.capacity_dpp_benefit_floor = floor
    ctrl.current_verifier_theta_f_ms_per_token = 2.15
    ctrl.current_active_clients = 4
    ctrl.server_queue = 0.0
    clients = [_client(alpha=0.999) for _ in range(4)]
    for cl, z in zip(clients, [0.0, 10.0, 10.0, 10.0]):
        cl.z_queue = z
    ctrl.capacity_dpp_clients = clients
    return ctrl, clients


def test_defaults_are_original():
    ctrl = _ctrl(V=1100.0)
    assert ctrl.capacity_dpp_benefit_floor == 0.0
    assert ctrl.capacity_dpp_load_price is False
    assert ctrl.capacity_dpp_z_load_damping == 0.0
    assert ctrl.capacity_dpp_load_scale() == 1.0
    assert ctrl.capacity_dpp_benefit_weight(-5000.0) == 0.0


def test_hard_clip_gives_fast_client_gamma_zero():
    ctrl, clients = _over_capacity(floor=0.0)
    assert _gamma(ctrl, clients[0]) == 0


def test_benefit_floor_keeps_speculation_for_fast_client():
    ctrl, clients = _over_capacity(floor=0.5)
    assert ctrl.capacity_dpp_benefit_weight(-1e9) == 550.0
    assert _gamma(ctrl, clients[0]) > 0
    # laggards are unaffected by the floor
    ctrl0, clients0 = _over_capacity(floor=0.0)
    assert _gamma(ctrl, clients[1]) == _gamma(ctrl0, clients0[1])


def test_fill_ema_and_load_scale():
    ctrl = _ctrl(V=1100.0)
    ctrl.capacity_dpp_load_price = True
    ctrl.capacity_dpp_load_ema = 0.5
    ctrl.observe_batch_fill(1, 40, 100)
    assert abs(ctrl.capacity_dpp_fill_ema - 0.4) < 1e-12
    ctrl.observe_batch_fill(1, 100, 100)  # duplicate batch id ignored
    assert abs(ctrl.capacity_dpp_fill_ema - 0.4) < 1e-12
    ctrl.observe_batch_fill(2, 100, 100)
    assert abs(ctrl.capacity_dpp_fill_ema - 0.7) < 1e-12
    assert abs(ctrl.capacity_dpp_load_scale() - 0.7) < 1e-12
    ctrl.capacity_dpp_fill_ema = 0.01
    assert ctrl.capacity_dpp_load_scale() == ctrl.capacity_dpp_load_price_min


def test_unsaturated_verifier_allows_deeper_gamma():
    def setup(fill):
        ctrl = _ctrl(V=1100.0)
        ctrl.capacity_dpp_quadratic_drift_weight = 1.0
        ctrl.current_verifier_theta_f_ms_per_token = 2.15
        ctrl.current_active_clients = 64
        ctrl.server_queue = 0.0
        ctrl.capacity_dpp_load_price = True
        ctrl.capacity_dpp_fill_ema = fill
        return ctrl
    cl = _client(alpha=0.999)
    g_low = _gamma(setup(0.3), cl)
    g_full = _gamma(setup(1.0), cl)
    assert g_low > g_full


def test_z_load_damping_shrinks_redistribution_when_saturated():
    ctrl = _ctrl(V=1100.0)
    ctrl.capacity_dpp_z_weight = 500.0
    ctrl.capacity_dpp_z_relative = True
    clients = [_client(alpha=0.9) for _ in range(2)]
    clients[0].z_queue, clients[1].z_queue = 0.0, 2.0
    ctrl.capacity_dpp_clients = clients
    assert ctrl.capacity_dpp_gamma_z(clients[1]) == 500.0
    ctrl.capacity_dpp_z_load_damping = 0.8
    ctrl.capacity_dpp_fill_ema = 1.0
    assert abs(ctrl.capacity_dpp_gamma_z(clients[1]) - 100.0) < 1e-9
