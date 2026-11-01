from __future__ import annotations


VALID_DRAFT_EXECUTION_MODES = {
    "simulation",
    "shared_gpu_emulation",
    "device_model",
}


def normalize_draft_execution_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_DRAFT_EXECUTION_MODES:
        raise ValueError(
            "draft_execution_mode must be one of: simulation, shared_gpu_emulation, device_model"
        )
    return normalized


def resolve_effective_draft_latency_ms(
    mode: str,
    measured_gpu_latency_ms: float,
    measured_wall_latency_ms: float,
    fixed_latency_ms: float,
    speed_multiplier: float,
    draft_tokens: int = 0,
    draft_tokens_per_second: float = 0.0,
) -> float:
    normalized = normalize_draft_execution_mode(mode)
    if normalized == "device_model":
        # Edge model: each client drafts on its own device, so latency is
        # fixed + gamma / tps_i and does not depend on how many clients run.
        if draft_tokens <= 0:
            return 0.0
        return max(0.0, fixed_latency_ms) + 1000.0 * draft_tokens / max(
            1e-6, draft_tokens_per_second
        )
    if normalized == "simulation":
        return max(0.0, fixed_latency_ms) + max(0.0, speed_multiplier) * max(
            0.0,
            measured_gpu_latency_ms,
        )
    return max(0.0, measured_wall_latency_ms)


def draft_queue_wait_ms(
    measured_gpu_latency_ms: float,
    measured_wall_latency_ms: float,
) -> float:
    return max(0.0, float(measured_wall_latency_ms) - float(measured_gpu_latency_ms))
