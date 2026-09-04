import pytest

from sim.controllers.gelato import GelatoController, GelatoDevice, GelatoStreamParams
from sim.core.server import RooflineParams


def make_roofline() -> RooflineParams:
    return RooflineParams(
        theta_0_seconds=0.002,
        theta_f_seconds_per_token=0.00005,
        beta_mem_bytes_per_second=2.0e12,
        b_w_bytes=1.0e10,
        b_kv_bytes_per_token=0.0,
    )


def make_device(V=10.0, energy_budget=1.0) -> GelatoDevice:
    return GelatoDevice(
        gamma_max=6, V=V, energy_budget_watts=energy_budget, alpha_hat=0.8, roofline=make_roofline()
    )


def make_params() -> GelatoStreamParams:
    return GelatoStreamParams(
        tau_d_seconds=0.005,
        uplink_rate_bps=2.0e6,
        kappa_bits_per_token=500.0,
        delta_seconds=0.01,
        draft_power_watts=0.5,
        uplink_power_watts=0.2,
    )


def test_choose_gamma_returns_a_value_within_range():
    device = make_device()
    gamma = device.choose_gamma(make_params())
    assert 0 <= gamma <= device.gamma_max


def test_tight_energy_budget_pushes_gamma_down():
    generous = make_device(V=10.0, energy_budget=1000.0)
    tight = make_device(V=10.0, energy_budget=1000.0)
    params = make_params()
    # Simulate a large accumulated energy backlog on the tight device (as if
    # it had been over budget for a while), which should discourage further
    # speculation relative to a device with no backlog.
    tight.Q = 1000.0
    g_generous = generous.choose_gamma(params)
    g_tight = tight.choose_gamma(params)
    assert g_tight <= g_generous


def test_update_queue_matches_eq7():
    device = make_device(energy_budget=2.0)
    device.update_queue(energy_used_watts=5.0)
    assert device.Q == pytest.approx(3.0)  # [0 + 5 - 2]^+
    device.update_queue(energy_used_watts=0.0)
    assert device.Q == pytest.approx(1.0)  # [3 + 0 - 2]^+


def test_update_queue_clips_at_zero():
    device = make_device(energy_budget=10.0)
    device.update_queue(energy_used_watts=1.0)
    assert device.Q == 0.0


def test_entropy_halting_stops_early_when_backlog_exceeds_threshold():
    device = make_device()
    device.entropy_leak_rate = 0.1
    device.entropy_bucket_capacity = 1.0
    # Entropy proxy 0.6 each step, leak 0.1: backlog is 0.5, 1.0, 1.5, ... --
    # exceeds capacity 1.0 at the 3rd token (index 2).
    halted_at = device.apply_entropy_halting(budget_gamma=5, per_token_entropy_proxy=[0.6, 0.6, 0.6, 0.6, 0.6])
    assert halted_at < 5


def test_entropy_halting_stays_within_budget_when_backlog_never_exceeds():
    device = make_device()
    device.entropy_leak_rate = 1.0
    device.entropy_bucket_capacity = 1.0
    # Entropy proxy always below the leak rate: backlog stays at 0 forever.
    halted_at = device.apply_entropy_halting(budget_gamma=5, per_token_entropy_proxy=[0.1] * 5)
    assert halted_at == 5


def test_controller_wraps_n_independent_devices():
    devices = [make_device(), make_device()]
    controller = GelatoController(devices=devices)
    params = [make_params(), make_params()]
    gammas = controller.decide_gammas(params)
    assert len(gammas) == 2
    controller.update_queues(energies_watts=[0.1, 0.1])
    assert devices[0].Q >= 0.0 and devices[1].Q >= 0.0


def test_controller_rejects_mismatched_lengths():
    controller = GelatoController(devices=[make_device()])
    with pytest.raises(ValueError):
        controller.decide_gammas([make_params(), make_params()])
