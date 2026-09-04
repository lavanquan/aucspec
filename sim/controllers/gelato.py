"""GELATO baseline: Lyapunov drift-plus-penalty device-edge token offloading.

Closest competitor to auc_controller.py -- also drift-plus-penalty, but
optimizes a single goodput point rather than sweeping the whole frontier.
Read the GELATO paper before modifying; document every divergence from
auc_controller.py inline. See TASKS.md T3.5.

Built from GELATO: Generative Entropy- and Lyapunov-based Adaptive Token
Offloading for Device-Edge Speculative LLM Inference (arXiv:2605.10124),
fetched and read while writing this file. Re-check against the source before
citing exact theorem numbers in the paper writeup.

Key differences from auc_controller.py (all because GELATO solves a
different problem, not implementation shortcuts):
  - Constrained resource is per-device ENERGY, not per-user interactivity.
    Virtual queue Q_i(t+1) = [Q_i(t) + E_i(t) - Ebar]^+ (eq. 7) tracks
    average power against a budget Ebar; there is no eq.(12) Z_i(t)/Q(t)/
    Q_i(t) triple and no interactivity-floor constraint x.
  - Device-agnostic and uncoupled: one GelatoDevice per stream, each
    choosing its own gamma independently. No server-side batching knapsack
    and no radio waterfilling across streams (GELATO is a single
    device-edge pair in the source; we run N independent copies).
  - No closed form for the speculation budget: the source paper states
    "no closed-form solution exists" for this subproblem and evaluates
    U(gamma) = V*throughput(gamma) - Q*energy(gamma) exhaustively over
    {0,...,gamma_max} (Algorithm 1) -- unlike auc_controller.py's eq.(13).
  - Optimizes ONE operating point set by V; does not sweep a frontier.
  - Entropy-driven dynamic halting (a leaky-bucket over per-token draft/
    target logit entropy, Theta_i = max(0, Theta_{i-1} + H_i - H_th)) is an
    *additional* inner loop on top of the budget search. This simulator's
    Assumption 1 only exposes Bernoulli(alpha) accept/reject, not logit
    entropy, so `apply_entropy_halting` below takes an injected per-token
    proxy signal rather than real entropy -- a documented stand-in, not the
    paper's mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..core.acceptance import expected_tokens_per_round
from ..core.device import round_cycle_seconds
from ..core.server import RooflineParams, verification_latency


@dataclass(frozen=True)
class GelatoStreamParams:
    """Per-slot link/device parameters for one GELATO device-edge pair."""

    tau_d_seconds: float
    uplink_rate_bps: float
    kappa_bits_per_token: float
    delta_seconds: float
    draft_power_watts: float
    uplink_power_watts: float
    context_length: float = 0.0


@dataclass
class GelatoDevice:
    """One independent drift-plus-penalty controller (GELATO eq. 7, Lemma 1).
    There is one of these per stream; unlike AUCController, they never
    interact (see module docstring)."""

    gamma_max: int
    V: float
    energy_budget_watts: float
    alpha_hat: float
    roofline: RooflineParams
    entropy_leak_rate: float = 0.5  # H_th: per-token entropy subtracted from the bucket each step
    entropy_bucket_capacity: float = 1.0  # Theta_th: halt once backlog exceeds this
    Q: float = field(default=0.0, init=False)
    _entropy_backlog: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.V <= 0.0:
            raise ValueError("V must be positive")
        if self.gamma_max < 0:
            raise ValueError("gamma_max must be non-negative")

    def _throughput_and_energy(self, gamma: int, params: GelatoStreamParams) -> tuple[float, float]:
        t_v = verification_latency(self.roofline, [gamma], [params.context_length])
        cycle_seconds = round_cycle_seconds(
            gamma,
            params.tau_d_seconds,
            params.kappa_bits_per_token,
            params.uplink_rate_bps,
            queueing_delay_seconds=0.0,
            verification_latency_seconds=t_v,
            delta_seconds=params.delta_seconds,
        )
        expected_tokens = expected_tokens_per_round(gamma, self.alpha_hat)
        throughput = expected_tokens / cycle_seconds
        device_seconds = gamma * params.tau_d_seconds
        uplink_seconds = params.kappa_bits_per_token * gamma / params.uplink_rate_bps
        energy = params.draft_power_watts * device_seconds + params.uplink_power_watts * uplink_seconds
        return throughput, energy

    def choose_gamma(self, params: GelatoStreamParams) -> int:
        """gamma_k-tilde*, Lemma 1: exhaustive search over {0,...,gamma_max}
        for the budget maximizing U(gamma) = V*throughput(gamma) -
        Q*energy(gamma). The source paper states no closed form exists for
        this subproblem, so we do not attempt one either."""
        best_gamma = 0
        best_utility = float("-inf")
        for gamma in range(self.gamma_max + 1):
            throughput, energy = self._throughput_and_energy(gamma, params)
            utility = self.V * throughput - self.Q * energy
            if utility > best_utility:
                best_utility = utility
                best_gamma = gamma
        return best_gamma

    def apply_entropy_halting(self, budget_gamma: int, per_token_entropy_proxy: Sequence[float]) -> int:
        """gamma_k*: stop drafting at the first position where the
        leaky-bucket backlog Theta_i = max(0, Theta_{i-1} + H_i - H_th)
        exceeds the bucket capacity Theta_th. See the module docstring's
        note on per_token_entropy_proxy standing in for real logit entropy.
        """
        halted_at = budget_gamma
        theta = self._entropy_backlog
        for i, h in enumerate(per_token_entropy_proxy[:budget_gamma]):
            theta = max(0.0, theta + h - self.entropy_leak_rate)
            if theta > self.entropy_bucket_capacity:
                halted_at = i
                break
        self._entropy_backlog = theta
        return halted_at

    def update_queue(self, energy_used_watts: float) -> None:
        """Q_i(t+1) = [Q_i(t) + E_i(t) - Ebar]^+, eq. (7)."""
        self.Q = max(0.0, self.Q + energy_used_watts - self.energy_budget_watts)


@dataclass
class GelatoController:
    """N independent GelatoDevice instances, one per stream. Exposed as a
    single object so run_baselines.py can drive it alongside AUCController,
    but there is no shared server/radio subproblem here -- see module
    docstring."""

    devices: list[GelatoDevice]

    def decide_gammas(self, params_by_stream: Sequence[GelatoStreamParams]) -> list[int]:
        if len(params_by_stream) != len(self.devices):
            raise ValueError("params_by_stream must have one entry per device")
        return [device.choose_gamma(params) for device, params in zip(self.devices, params_by_stream)]

    def update_queues(self, energies_watts: Sequence[float]) -> None:
        if len(energies_watts) != len(self.devices):
            raise ValueError("energies_watts must have one entry per device")
        for device, energy in zip(self.devices, energies_watts):
            device.update_queue(energy)
