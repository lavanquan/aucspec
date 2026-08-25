from __future__ import annotations


VALID_DRAFT_EXECUTION_MODES = {
    "simulation",
    "shared_gpu_emulation",
}


def normalize_draft_execution_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_DRAFT_EXECUTION_MODES:
        raise ValueError(
            "draft_execution_mode must be one of: simulation, shared_gpu_emulation"
        )
    return normalized


def resolve_effective_draft_latency_ms(
    mode: str,
    measured_gpu_latency_ms: float,
    measured_wall_latency_ms: float,
    fixed_latency_ms: float,
    speed_multiplier: float,
) -> float:
    normalized = normalize_draft_execution_mode(mode)
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
