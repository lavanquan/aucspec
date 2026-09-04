"""Shared verification server: roofline latency model and batch verification.

See docs/notation.md eq. (3), Assumption 2. TASKS.md T1.3. Both the
memory-bound and compute-bound regimes are handled (Theorem 1's two-regime
frontier hinges on the roofline knee Gamma_dagger between them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RooflineParams:
    """Server-side constants in eq. (3): theta_0, theta_f, beta_mem, b_W, b_kv."""

    theta_0_seconds: float
    theta_f_seconds_per_token: float
    beta_mem_bytes_per_second: float
    b_w_bytes: float
    b_kv_bytes_per_token: float

    def __post_init__(self) -> None:
        for name in (
            "theta_0_seconds",
            "theta_f_seconds_per_token",
            "beta_mem_bytes_per_second",
            "b_w_bytes",
            "b_kv_bytes_per_token",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.beta_mem_bytes_per_second == 0.0:
            raise ValueError("beta_mem_bytes_per_second must be positive")


def batch_token_count(gammas: Sequence[int]) -> int:
    """Gamma(B) = sum_{i in B} (gamma_i + 1), the total draft+bonus tokens
    verified in one forward pass over batch B."""
    return sum(int(g) + 1 for g in gammas)


def memory_bound_latency(params: RooflineParams, context_lengths: Sequence[float]) -> float:
    """theta_m(B) = (b_W + b_kv * sum_i L_i) / beta_mem, the memory-bound term
    of eq. (3): weight bytes plus per-token KV-cache bytes summed over the
    batch's context lengths, divided by memory bandwidth."""
    total_kv_bytes = params.b_kv_bytes_per_token * sum(float(length) for length in context_lengths)
    return (params.b_w_bytes + total_kv_bytes) / params.beta_mem_bytes_per_second


def compute_bound_latency(params: RooflineParams, gammas: Sequence[int]) -> float:
    """theta_f * Gamma(B), the compute-bound term of eq. (3)."""
    return params.theta_f_seconds_per_token * batch_token_count(gammas)


def verification_latency(
    params: RooflineParams,
    gammas: Sequence[int],
    context_lengths: Sequence[float],
) -> float:
    """T^v(B) = theta_0 + max(theta_m(B), theta_f * Gamma(B)), eq. (3)."""
    theta_m = memory_bound_latency(params, context_lengths)
    theta_f_term = compute_bound_latency(params, gammas)
    return params.theta_0_seconds + max(theta_m, theta_f_term)


def roofline_knee(params: RooflineParams, context_lengths: Sequence[float]) -> float:
    """Gamma_dagger(B) = theta_m(B) / theta_f, the batch token count at which
    the memory-bound and compute-bound terms of eq. (3) cross."""
    theta_m = memory_bound_latency(params, context_lengths)
    return theta_m / params.theta_f_seconds_per_token
