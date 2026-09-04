import pytest

from sim.controllers.auc_controller import StreamLinkState
from sim.controllers.fixed_slo import FixedSLOController
from sim.core.server import RooflineParams


def make_roofline() -> RooflineParams:
    return RooflineParams(
        theta_0_seconds=0.002,
        theta_f_seconds_per_token=0.00005,
        beta_mem_bytes_per_second=2.0e12,
        b_w_bytes=1.0e10,
        b_kv_bytes_per_token=0.0,
    )


def test_tight_slo_forces_gamma_zero():
    ctrl = FixedSLOController(
        n_streams=1,
        gamma_max=8,
        slo_seconds=[0.001],  # far too tight to speculate at all
        kappa_bits_per_token=500.0,
        delta_seconds=0.01,
        roofline=make_roofline(),
        gamma_budget=100.0,
        total_bandwidth_hz=1.0e6,
    )
    links = [StreamLinkState(0.01, 1.0e6, 3.0)]
    decision = ctrl.decide(t=0, links=links)
    assert decision.gammas == [0]


def test_loose_slo_allows_gamma_max():
    ctrl = FixedSLOController(
        n_streams=1,
        gamma_max=4,
        slo_seconds=[10.0],  # generous
        kappa_bits_per_token=500.0,
        delta_seconds=0.01,
        roofline=make_roofline(),
        gamma_budget=100.0,
        total_bandwidth_hz=1.0e6,
    )
    links = [StreamLinkState(0.01, 1.0e6, 3.0)]
    decision = ctrl.decide(t=0, links=links)
    assert decision.gammas == [4]


def test_each_stream_uses_its_own_slo_independently():
    ctrl = FixedSLOController(
        n_streams=2,
        gamma_max=6,
        slo_seconds=[10.0, 0.001],
        kappa_bits_per_token=500.0,
        delta_seconds=0.01,
        roofline=make_roofline(),
        gamma_budget=100.0,
        total_bandwidth_hz=1.0e6,
    )
    links = [StreamLinkState(0.01, 1.0e6, 3.0), StreamLinkState(0.01, 1.0e6, 3.0)]
    decision = ctrl.decide(t=0, links=links)
    assert decision.gammas[0] > decision.gammas[1]


def test_rejects_mismatched_slo_length():
    with pytest.raises(ValueError):
        FixedSLOController(
            n_streams=2,
            gamma_max=6,
            slo_seconds=[1.0],  # wrong length
            kappa_bits_per_token=500.0,
            delta_seconds=0.01,
            roofline=make_roofline(),
            gamma_budget=100.0,
            total_bandwidth_hz=1.0e6,
        )
