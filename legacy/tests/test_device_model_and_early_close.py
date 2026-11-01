from edge_specsim.draft_timing import resolve_effective_draft_latency_ms


def test_device_model_ignores_measured_latency():
    ms = resolve_effective_draft_latency_ms(
        mode="device_model", measured_gpu_latency_ms=999.0, measured_wall_latency_ms=5000.0,
        fixed_latency_ms=10.0, speed_multiplier=3.0, draft_tokens=4, draft_tokens_per_second=40.0,
    )
    assert ms == 10.0 + 100.0


def test_device_model_zero_gamma():
    assert resolve_effective_draft_latency_ms(
        "device_model", 50.0, 50.0, 10.0, 1.0, draft_tokens=0, draft_tokens_per_second=40.0
    ) == 0.0


def test_other_modes_unchanged():
    assert resolve_effective_draft_latency_ms("simulation", 50.0, 80.0, 10.0, 2.0) == 110.0
    assert resolve_effective_draft_latency_ms("shared_gpu_emulation", 50.0, 80.0, 10.0, 2.0) == 80.0
