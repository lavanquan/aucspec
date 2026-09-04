import random

import pytest

from sim.core.acceptance import (
    StationaryAcceptance,
    TraceAcceptance,
    expected_tokens_per_round,
    sample_round,
    sliding_window_alpha,
)


def test_phi_matches_closed_form():
    # phi(gamma,alpha) = (1 - alpha^(gamma+1)) / (1 - alpha), eq. (2)
    assert expected_tokens_per_round(0, 0.8) == pytest.approx(1.0)
    assert expected_tokens_per_round(3, 0.5) == pytest.approx((1 - 0.5**4) / 0.5)


def test_phi_alpha_one_limit():
    assert expected_tokens_per_round(5, 1.0) == 6.0


def test_phi_rejects_bad_gamma_or_alpha():
    with pytest.raises(ValueError):
        expected_tokens_per_round(-1, 0.5)
    with pytest.raises(ValueError):
        expected_tokens_per_round(2, 1.5)


def test_sample_round_gamma_zero_always_delivers_bonus_only():
    rng = random.Random(0)
    tokens, all_accepted = sample_round(0, 0.9, rng)
    assert tokens == 1
    assert all_accepted is True


def test_sample_round_alpha_zero_never_accepts():
    rng = random.Random(0)
    for _ in range(20):
        tokens, all_accepted = sample_round(4, 0.0, rng)
        assert tokens == 1
        assert all_accepted is False


def test_sample_round_alpha_one_always_accepts():
    rng = random.Random(0)
    for _ in range(20):
        tokens, all_accepted = sample_round(4, 1.0, rng)
        assert tokens == 5
        assert all_accepted is True


def test_stationary_acceptance_empirical_mean_matches_phi():
    acc = StationaryAcceptance(alpha=0.7, seed=42)
    gamma = 4
    n = 20000
    total = sum(acc.sample(gamma)[0] for _ in range(n))
    empirical_mean = total / n
    assert empirical_mean == pytest.approx(acc.expected(gamma), rel=0.02)


def test_trace_acceptance_alpha_at_clamps():
    trace = TraceAcceptance(alpha_by_round=[0.5, 0.6, 0.7], seed=0)
    assert trace.alpha_at(-1) == 0.5
    assert trace.alpha_at(1) == 0.6
    assert trace.alpha_at(100) == 0.7


def test_sliding_window_alpha_matches_manual_average():
    flags = [True, True, False, True, False, False, True]
    estimates = sliding_window_alpha(flags, window=3)
    assert len(estimates) == len(flags)
    # last window is flags[4:7] = [False, False, True] -> 1/3
    assert estimates[-1] == pytest.approx(1.0 / 3.0)
    # first estimate uses a window of size 1 (only itself)
    assert estimates[0] == pytest.approx(1.0)
