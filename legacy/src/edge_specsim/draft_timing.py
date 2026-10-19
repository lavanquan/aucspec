from __future__ import annotations


VALID_DRAFT_EXECUTION_MODES = {
    "simulation",
    "shared_gpu_emulation",
    # C1: virtual draft latency from an ISOLATED single-device profile, independent of how many draft clients
    # share this emulator process (GIL / GPU contention inflates the measured draft latency ~6.6x at N=8).
    "isolated_profile",
}


def normalize_draft_execution_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_DRAFT_EXECUTION_MODES:
        raise ValueError(
            "draft_execution_mode must be one of: simulation, shared_gpu_emulation"
        )
    return normalized


def isolated_profile_latency_ms(gamma: int, profile: dict | None) -> float:
    """Latency of drafting `gamma` tokens on an isolated device: overhead_ms + per_token_ms * gamma (0 if gamma<=0)."""
    if profile is None or "per_token_ms" not in profile:
        raise ValueError("draft_execution_mode=isolated_profile requires simulation.draft_profile.per_token_ms")
    if gamma <= 0:
        return 0.0
    return max(0.0, float(profile.get("overhead_ms", 0.0))) + max(0.0, float(profile["per_token_ms"])) * int(gamma)


def resolve_effective_draft_latency_ms(
    mode: str,
    measured_gpu_latency_ms: float,
    measured_wall_latency_ms: float,
    fixed_latency_ms: float,
    speed_multiplier: float,
    gamma: int = 0,
    profile: dict | None = None,
) -> float:
    normalized = normalize_draft_execution_mode(mode)
    if normalized == "isolated_profile":
        return max(0.0, fixed_latency_ms) + max(0.0, speed_multiplier) * isolated_profile_latency_ms(gamma, profile)
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
