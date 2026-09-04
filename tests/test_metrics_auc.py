import pytest

from sim.metrics.auc import normalized_auc, trapezoidal_auc
from sim.metrics.frontier import OperatingPoint


def test_trapezoidal_auc_matches_hand_computed_area():
    # Two points: a straight line from (0, 10) to (2, 4).
    points = [
        OperatingPoint(min_interactivity=0.0, goodput=10.0),
        OperatingPoint(min_interactivity=2.0, goodput=4.0),
    ]
    # Trapezoid area = 0.5*(10+4)*2 = 14
    assert trapezoidal_auc(points) == pytest.approx(14.0)


def test_trapezoidal_auc_multi_segment():
    points = [
        OperatingPoint(min_interactivity=0.0, goodput=10.0),
        OperatingPoint(min_interactivity=1.0, goodput=8.0),
        OperatingPoint(min_interactivity=3.0, goodput=2.0),
    ]
    expected = 0.5 * (10 + 8) * 1.0 + 0.5 * (8 + 2) * 2.0
    assert trapezoidal_auc(points) == pytest.approx(expected)


def test_trapezoidal_auc_needs_at_least_two_points():
    assert trapezoidal_auc([]) == 0.0
    assert trapezoidal_auc([OperatingPoint(0.0, 5.0)]) == 0.0


def test_trapezoidal_auc_rejects_unsorted_points():
    points = [
        OperatingPoint(min_interactivity=2.0, goodput=4.0),
        OperatingPoint(min_interactivity=0.0, goodput=10.0),
    ]
    with pytest.raises(ValueError):
        trapezoidal_auc(points)


def test_normalized_auc_is_one_for_a_flat_frontier():
    # A constant-Y frontier from x=0 to x=xbar has AUC = xbar*Y, so hat-AUC = 1.
    points = [
        OperatingPoint(min_interactivity=0.0, goodput=5.0),
        OperatingPoint(min_interactivity=4.0, goodput=5.0),
    ]
    assert normalized_auc(points) == pytest.approx(1.0)


def test_normalized_auc_requires_x0_point():
    points = [
        OperatingPoint(min_interactivity=1.0, goodput=5.0),
        OperatingPoint(min_interactivity=4.0, goodput=5.0),
    ]
    with pytest.raises(ValueError):
        normalized_auc(points)
