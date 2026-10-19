from __future__ import annotations

from edge_specsim.draft_timing import (
    draft_queue_wait_ms,
    normalize_draft_execution_mode,
    resolve_effective_draft_latency_ms,
)


def test_simulation_mode_uses_profiled_latency() -> None:
    latency_ms = resolve_effective_draft_latency_ms(
        mode="simulation",
        measured_gpu_latency_ms=12.5,
        measured_wall_latency_ms=44.0,
        fixed_latency_ms=7.0,
        speed_multiplier=1.6,
    )

    assert latency_ms == 27.0


def test_shared_gpu_emulation_uses_measured_wall_clock_latency() -> None:
    latency_ms = resolve_effective_draft_latency_ms(
        mode="shared_gpu_emulation",
        measured_gpu_latency_ms=12.5,
        measured_wall_latency_ms=44.0,
        fixed_latency_ms=7.0,
        speed_multiplier=1.6,
    )

    assert latency_ms == 44.0


def test_draft_queue_wait_is_non_negative() -> None:
    assert draft_queue_wait_ms(12.5, 44.0) == 31.5
    assert draft_queue_wait_ms(12.5, 10.0) == 0.0


def test_invalid_draft_execution_mode_raises() -> None:
    try:
        normalize_draft_execution_mode("invalid")
    except ValueError as exc:
        assert "draft_execution_mode must be one of" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid draft execution mode")


def test_isolated_profile_is_independent_of_measured_contention() -> None:
    kw = dict(mode="isolated_profile", fixed_latency_ms=5.0, speed_multiplier=1.0, gamma=3,
              profile={"overhead_ms": 2.0, "per_token_ms": 22.0})
    quiet = resolve_effective_draft_latency_ms(measured_gpu_latency_ms=66.0, measured_wall_latency_ms=67.0, **kw)
    noisy = resolve_effective_draft_latency_ms(measured_gpu_latency_ms=450.0, measured_wall_latency_ms=900.0, **kw)
    assert quiet == noisy == 5.0 + 2.0 + 66.0
    assert resolve_effective_draft_latency_ms(measured_gpu_latency_ms=1.0, measured_wall_latency_ms=1.0,
                                              **{**kw, "gamma": 0}) == 5.0


def test_isolated_profile_requires_a_profile() -> None:
    import pytest
    with pytest.raises(ValueError):
        resolve_effective_draft_latency_ms("isolated_profile", 1.0, 1.0, 0.0, 1.0, gamma=2, profile=None)
