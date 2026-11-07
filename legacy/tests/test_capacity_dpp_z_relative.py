"""capacity_dpp.z_relative: Z prices gamma relative to the mean Z."""
from __future__ import annotations

from tests.test_capacity import _client, _ctrl


def _gamma(ctrl, cl):
    return ctrl._capacity_dpp_gamma(cl, use_ucb=True)


def _setup(z_relative: bool):
    ctrl = _ctrl(V=1500.0)
    ctrl.capacity_dpp_quadratic_drift_weight = 1.0
    ctrl.capacity_dpp_z_weight = 500.0
    ctrl.current_verifier_theta_f_ms_per_token = 2.15
    ctrl.current_active_clients = 64
    ctrl.server_queue = 0.0
    ctrl.capacity_dpp_z_relative = z_relative
    clients = [_client(alpha=0.999) for _ in range(4)]
    for cl, z in zip(clients, [20.0, 20.0, 20.0, 26.0]):
        cl.z_queue = z
    ctrl.capacity_dpp_clients = clients
    return ctrl, clients


def test_default_is_absolute():
    ctrl = _ctrl(V=1500.0)
    assert ctrl.capacity_dpp_z_relative is False
    cl = _client(alpha=0.9)
    cl.z_queue = 3.0
    ctrl.capacity_dpp_z_weight = 50.0
    assert ctrl.capacity_dpp_gamma_z(cl) == 150.0


def test_absolute_z_drives_everyone_to_gamma_max_when_all_lag():
    ctrl, clients = _setup(z_relative=False)
    assert all(_gamma(ctrl, c) == max(ctrl.gamma_choices) for c in clients)


def test_relative_z_only_lifts_the_laggard():
    ctrl, clients = _setup(z_relative=True)
    g = [_gamma(ctrl, c) for c in clients]
    assert g[0] == g[1] == g[2]
    assert g[3] > g[0]
    assert g[0] < max(ctrl.gamma_choices)
