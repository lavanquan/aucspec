import pytest

from sim.core.server import (
    RooflineParams,
    batch_token_count,
    compute_bound_latency,
    memory_bound_latency,
    roofline_knee,
    verification_latency,
)


def make_params(**overrides) -> RooflineParams:
    defaults = dict(
        theta_0_seconds=0.003,
        theta_f_seconds_per_token=0.0001,
        beta_mem_bytes_per_second=1.0e9,
        b_w_bytes=1.0e6,
        b_kv_bytes_per_token=100.0,
    )
    defaults.update(overrides)
    return RooflineParams(**defaults)


def test_batch_token_count():
    assert batch_token_count([0, 3, 7]) == (0 + 1) + (3 + 1) + (7 + 1)


def test_memory_bound_latency_matches_formula():
    params = make_params()
    context_lengths = [10.0, 20.0]
    expected = (params.b_w_bytes + params.b_kv_bytes_per_token * 30.0) / params.beta_mem_bytes_per_second
    assert memory_bound_latency(params, context_lengths) == pytest.approx(expected)


def test_compute_bound_latency_matches_formula():
    params = make_params()
    gammas = [1, 2]
    expected = params.theta_f_seconds_per_token * ((1 + 1) + (2 + 1))
    assert compute_bound_latency(params, gammas) == pytest.approx(expected)


def test_verification_latency_is_max_of_the_two_regimes():
    # Memory-bound dominates: tiny batch, huge context.
    params = make_params()
    mem_dominant = verification_latency(params, gammas=[0], context_lengths=[1.0e9])
    assert mem_dominant == pytest.approx(
        params.theta_0_seconds + memory_bound_latency(params, [1.0e9])
    )

    # Compute-bound dominates: huge batch, zero context (b_kv term vanishes).
    compute_dominant = verification_latency(params, gammas=[1000] * 50, context_lengths=[0.0] * 50)
    assert compute_dominant == pytest.approx(
        params.theta_0_seconds + compute_bound_latency(params, [1000] * 50)
    )


def test_roofline_knee_matches_formula():
    params = make_params()
    context_lengths = [5.0, 5.0]
    expected = memory_bound_latency(params, context_lengths) / params.theta_f_seconds_per_token
    assert roofline_knee(params, context_lengths) == pytest.approx(expected)


def test_negative_params_rejected():
    with pytest.raises(ValueError):
        make_params(theta_0_seconds=-1.0)


def test_zero_memory_bandwidth_rejected():
    with pytest.raises(ValueError):
        make_params(beta_mem_bytes_per_second=0.0)
