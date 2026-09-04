import math

import pytest

from sim.core.channel import shannon_rate, snr_linear_from_db, spectral_efficiency, sqrt_waterfilling


def test_shannon_rate_matches_formula():
    # r = w * log2(1 + SNR), eq. (1)
    assert shannon_rate(1.0e6, 3.0) == pytest.approx(1.0e6 * math.log2(4.0))


def test_shannon_rate_zero_bandwidth():
    assert shannon_rate(0.0, 100.0) == 0.0


def test_snr_linear_from_db_roundtrip():
    assert snr_linear_from_db(0.0) == pytest.approx(1.0)
    assert snr_linear_from_db(10.0) == pytest.approx(10.0)


def test_spectral_efficiency_matches_shannon_rate():
    snr = snr_linear_from_db(6.0)
    assert shannon_rate(2.0, snr) == pytest.approx(2.0 * spectral_efficiency(snr))


def test_sqrt_waterfilling_sums_to_budget():
    costs = [1.0, 4.0, 9.0]
    allocation = sqrt_waterfilling(costs, total_bandwidth=100.0)
    assert sum(allocation) == pytest.approx(100.0)
    # Larger cost -> larger share, proportional to sqrt(cost).
    assert allocation[2] > allocation[1] > allocation[0]
    assert allocation[2] / allocation[0] == pytest.approx(3.0)  # sqrt(9)/sqrt(1)


def test_sqrt_waterfilling_zero_cost_gets_nothing():
    allocation = sqrt_waterfilling([0.0, 1.0], total_bandwidth=10.0)
    assert allocation[0] == 0.0
    assert allocation[1] == pytest.approx(10.0)


def test_sqrt_waterfilling_all_zero_splits_evenly():
    allocation = sqrt_waterfilling([0.0, 0.0], total_bandwidth=10.0)
    assert allocation == [5.0, 5.0]


def test_negative_bandwidth_rejected():
    with pytest.raises(ValueError):
        shannon_rate(-1.0, 1.0)
