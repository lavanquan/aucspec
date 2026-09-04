"""Per-device round loop: draft -> uplink -> queue -> verify -> downlink.

Implements the closed-loop renewal cycle under Assumption 4: round k+1 is
issued immediately upon receipt of round k's response, so per-stream round
epochs form renewal cycles under any stationary policy (Definition 1, eq. 4).
TASKS.md T1.4.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .acceptance import expected_tokens_per_round, sample_round
from .server import RooflineParams, verification_latency


def round_cycle_seconds(
    gamma: int,
    tau_d_seconds: float,
    kappa_bits_per_token: float,
    uplink_rate_bps: float,
    queueing_delay_seconds: float,
    verification_latency_seconds: float,
    delta_seconds: float,
) -> float:
    """Denominator of eq. (4): E[gamma*tau_d + kappa*gamma/r + W^q + T^v + delta],
    the wall-clock length of one renewal cycle for a single round.
    """
    if uplink_rate_bps <= 0.0:
        raise ValueError("uplink_rate_bps must be positive")
    draft_seconds = gamma * tau_d_seconds
    uplink_seconds = kappa_bits_per_token * gamma / uplink_rate_bps
    return (
        draft_seconds
        + uplink_seconds
        + queueing_delay_seconds
        + verification_latency_seconds
        + delta_seconds
    )


@dataclass
class SingleStreamSimulator:
    """Monte Carlo simulator for one isolated stream (no batching contention),
    used to validate the closed-form eq. (2), (4) against simulation (T1.5).

    The stream drafts a fixed gamma every round, sends over a fixed-rate
    uplink, and is verified alone in its own batch of size 1 (queueing delay
    fixed at 0 -- no contention to model here; that is Phase 2's job).
    """

    gamma: int
    alpha: float
    tau_d_seconds: float
    kappa_bits_per_token: float
    uplink_rate_bps: float
    delta_seconds: float
    roofline: RooflineParams
    context_length: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def run(self, num_rounds: int, warmup_rounds: int = 0) -> dict:
        """Run num_rounds closed-loop rounds; return empirical interactivity
        (tokens delivered / elapsed seconds) over the post-warmup rounds,
        to compare against eq. (4)/(10)'s closed form.
        """
        if num_rounds <= warmup_rounds:
            raise ValueError("num_rounds must exceed warmup_rounds")
        t_v = verification_latency(self.roofline, [self.gamma], [self.context_length])
        cycle_seconds = round_cycle_seconds(
            self.gamma,
            self.tau_d_seconds,
            self.kappa_bits_per_token,
            self.uplink_rate_bps,
            queueing_delay_seconds=0.0,
            verification_latency_seconds=t_v,
            delta_seconds=self.delta_seconds,
        )
        total_tokens = 0
        total_seconds = 0.0
        for round_index in range(num_rounds):
            tokens, _ = sample_round(self.gamma, self.alpha, self._rng)
            if round_index >= warmup_rounds:
                total_tokens += tokens
                total_seconds += cycle_seconds
        interactivity = total_tokens / total_seconds if total_seconds > 0.0 else 0.0
        return {
            "interactivity_tokens_per_second": interactivity,
            "cycle_seconds": cycle_seconds,
            "rounds_measured": num_rounds - warmup_rounds,
            "total_tokens": total_tokens,
        }


def symmetric_cycle_seconds(
    batch_size: int,
    gamma: int,
    tau_d_seconds: float,
    kappa_bits_per_token: float,
    uplink_rate_bps: float,
    delta_seconds: float,
    roofline: RooflineParams,
) -> float:
    """T(B,gamma), eq. (10): the symmetric saturated instance's per-round
    cycle time when B identical streams are gated-batch verified together at
    a common speculation length gamma (so W_i^q = 0 and b_kv = 0, i.e.
    theta_m is constant -- see Theorem 1's preconditions).
    """
    if uplink_rate_bps <= 0.0:
        raise ValueError("uplink_rate_bps must be positive")
    per_token_cost = tau_d_seconds + kappa_bits_per_token / uplink_rate_bps
    theta_m = roofline.b_w_bytes / roofline.beta_mem_bytes_per_second
    theta_f_term = roofline.theta_f_seconds_per_token * batch_size * (gamma + 1)
    return gamma * per_token_cost + delta_seconds + roofline.theta_0_seconds + max(theta_m, theta_f_term)


def symmetric_operating_point(
    batch_size: int,
    gamma: int,
    alpha: float,
    tau_d_seconds: float,
    kappa_bits_per_token: float,
    uplink_rate_bps: float,
    delta_seconds: float,
    roofline: RooflineParams,
) -> tuple[float, float]:
    """(x(B,gamma), Y), eq. (10): per-stream interactivity and system goodput
    (B * x, since every one of the B streams is identical) in the symmetric
    saturated instance.
    """
    cycle_seconds = symmetric_cycle_seconds(
        batch_size, gamma, tau_d_seconds, kappa_bits_per_token, uplink_rate_bps, delta_seconds, roofline
    )
    x = expected_tokens_per_round(gamma, alpha) / cycle_seconds
    return x, batch_size * x
