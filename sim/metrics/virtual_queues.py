"""Virtual queues Z_i(t), Q(t), Q_i(t) (Theorem 3 backlog bounds), eq. (12).

Feeds the V-tradeoff and backlog plots in EXPERIMENTS.md Exp 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class VirtualQueues:
    """State of the interactivity, server-compute, and device-time virtual
    queues used by the drift-plus-penalty controller (eq. 12)."""

    Z: list[float]
    Q: float
    Qi: list[float]

    @classmethod
    def zeros(cls, n_streams: int) -> "VirtualQueues":
        return cls(Z=[0.0] * n_streams, Q=0.0, Qi=[0.0] * n_streams)


def update_virtual_queues(
    queues: VirtualQueues,
    x_target: float,
    a_i: Sequence[float],
    theta_f_seconds_per_token: float,
    verification_batch_tokens: float,
    gammas: Sequence[int],
    device_time_per_gamma: Sequence[float],
    slot_seconds: float,
) -> VirtualQueues:
    """eq. (12):
      Z_i(t+1) = [Z_i(t) + x*Delta_s - a_i(t)]^+
      Q(t+1)   = [Q(t) + theta_f*Gamma(B(t)) - Delta_s]^+
      Q_i(t+1) = [Q_i(t) + gamma_i(t)*(tau_i^d + kappa/r_i(t)) - Delta_s]^+

    `device_time_per_gamma[i]` is the per-token device+uplink time
    (tau_i^d + kappa/r_i(t)) that eq. (12) multiplies by gamma_i(t).
    """
    n = len(queues.Z)
    if not (len(a_i) == n == len(gammas) == len(device_time_per_gamma)):
        raise ValueError("a_i, gammas, device_time_per_gamma must have one entry per stream")
    if slot_seconds <= 0.0:
        raise ValueError("slot_seconds must be positive")

    new_z = [max(0.0, z + x_target * slot_seconds - a) for z, a in zip(queues.Z, a_i)]
    new_q = max(0.0, queues.Q + theta_f_seconds_per_token * verification_batch_tokens - slot_seconds)
    new_qi = [
        max(0.0, qi + gamma * dt - slot_seconds)
        for qi, gamma, dt in zip(queues.Qi, gammas, device_time_per_gamma)
    ]
    return VirtualQueues(Z=new_z, Q=new_q, Qi=new_qi)


def total_backlog(queues: VirtualQueues) -> float:
    """E[Z_i(t) + Q(t) + Q_i(t)] proxy (sum over the current state) used to
    check the O(V) backlog side of Theorem 3 (eq. 14)."""
    return sum(queues.Z) + queues.Q + sum(queues.Qi)
