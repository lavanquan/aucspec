import pytest

from sim.core.server import RooflineParams
from sim.metrics.frontier import OperatingPoint, build_frontier, sweep_symmetric_gamma


def test_build_frontier_sorts_and_keeps_best_y_per_x():
    points = [
        OperatingPoint(min_interactivity=2.0, goodput=10.0),
        OperatingPoint(min_interactivity=1.0, goodput=5.0),
        OperatingPoint(min_interactivity=1.0, goodput=7.0),  # dominates the x=1.0 point above
    ]
    frontier = build_frontier(points)
    assert [p.min_interactivity for p in frontier] == [1.0, 2.0]
    assert frontier[0].goodput == 7.0


def test_build_frontier_rejects_negative_values():
    with pytest.raises(ValueError):
        build_frontier([OperatingPoint(min_interactivity=-1.0, goodput=1.0)])


def test_sweep_symmetric_gamma_returns_one_point_per_choice():
    roofline = RooflineParams(
        theta_0_seconds=0.003,
        theta_f_seconds_per_token=0.0001,
        beta_mem_bytes_per_second=1.0e9,
        b_w_bytes=1.0e6,
        b_kv_bytes_per_token=0.0,
    )
    gamma_choices = [0, 1, 2, 4, 8]
    points = sweep_symmetric_gamma(
        batch_size=8,
        alpha=0.8,
        gamma_choices=gamma_choices,
        tau_d_seconds=0.01,
        kappa_bits_per_token=1000.0,
        uplink_rate_bps=5.0e6,
        delta_seconds=0.01,
        roofline=roofline,
    )
    assert [p.label for p in points] == [f"gamma={g}" for g in gamma_choices]
    assert all(p.goodput > 0.0 for p in points)
