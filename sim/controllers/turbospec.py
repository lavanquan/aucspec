"""TurboSpec/SmartSpec-style baseline: closed-loop speculation control on goodput.

Represents arXiv:2406.14066. See TASKS.md T3.3.

Built from "TurboSpec: Closed-loop Speculation Control System for Optimizing
LLM Serving Goodput" (arXiv:2406.14066), fetched and read while writing this
file. Faithful to the source's core mechanism:
  - l(alpha,k) = (1-alpha^(k+1))/(1-alpha) is exactly our eq.(2)
    phi(gamma,alpha) -- reused directly from core.acceptance, not re-derived.
  - ArgMaxGoodput: search k in {0,...,K_max}, pick the k maximizing
    goodput(k) = expected_tokens(k,alpha_hat) / batch_latency(k), applied as
    ONE global speculation length for the whole batch (not per-stream) --
    unlike auc_controller.py's per-stream gamma_i(t).
  - alpha_hat is an exponential moving average of observed acceptance, not
    the UCB/Bernstein estimator of Theorem 4 -- TurboSpec makes no
    optimism/regret guarantee, it just tracks the empirical rate.

Deviation from the source: TurboSpec profiles batch latency
T_fwd(M,N_context,N_batched) = alpha*N_context + gamma*N_batched + delta via
offline linear regression. This simulator already has a physically-derived
roofline latency model (core.server.verification_latency, eq. 3) that serves
the same purpose more precisely, so we reuse it instead of fitting a
separate linear model -- the ArgMaxGoodput search itself is unchanged. There
is also no radio-waterfilling or knapsack-batching subproblem here:
TurboSpec assumes a fixed admitted batch and only controls k.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .auc_controller import SpeculationDecision, StreamLinkState
from ..core.acceptance import expected_tokens_per_round
from ..core.device import round_cycle_seconds
from ..core.server import RooflineParams, verification_latency


@dataclass
class MovingAverageAcceptance:
    """Exponential moving average of the observed token-level acceptance
    rate -- TurboSpec's alpha_hat, with no optimism/confidence bound
    (unlike Theorem 4's UCB/Bernstein estimator)."""

    decay: float = 0.98
    alpha_hat: float = 0.5

    def observe(self, accepted: int, gamma: int) -> None:
        if gamma <= 0:
            return
        empirical = min(1.0, accepted / gamma)
        self.alpha_hat = self.decay * self.alpha_hat + (1.0 - self.decay) * empirical


@dataclass
class TurboSpecController:
    """Search-based ArgMaxGoodput: one global speculation length k applied
    to the whole admitted batch each slot, chosen to maximize aggregate
    goodput (validated tokens per second)."""

    n_streams: int
    gamma_max: int
    tau_d_seconds: float
    kappa_bits_per_token: float
    uplink_rate_bps: float
    delta_seconds: float
    roofline: RooflineParams
    total_bandwidth_hz: float
    context_length: float = 0.0
    acceptance: MovingAverageAcceptance = field(default_factory=MovingAverageAcceptance)

    def __post_init__(self) -> None:
        if self.n_streams <= 0:
            raise ValueError("n_streams must be positive")
        if self.gamma_max < 0:
            raise ValueError("gamma_max must be non-negative")

    def _goodput(self, k: int, batch_size: int) -> float:
        t_v = verification_latency(self.roofline, [k] * batch_size, [self.context_length] * batch_size)
        cycle_seconds = round_cycle_seconds(
            k,
            self.tau_d_seconds,
            self.kappa_bits_per_token,
            self.uplink_rate_bps,
            queueing_delay_seconds=0.0,
            verification_latency_seconds=t_v,
            delta_seconds=self.delta_seconds,
        )
        expected_tokens_per_stream = expected_tokens_per_round(k, self.acceptance.alpha_hat)
        return batch_size * expected_tokens_per_stream / cycle_seconds

    def decide(self, t: int, links: Sequence[StreamLinkState]) -> SpeculationDecision:
        if len(links) != self.n_streams:
            raise ValueError("links must have one entry per stream")
        batch_size = self.n_streams  # ArgMaxGoodput assumes a fixed admitted batch (module docstring)
        best_k = 0
        best_goodput = float("-inf")
        for k in range(self.gamma_max + 1):
            goodput = self._goodput(k, batch_size)
            if goodput > best_goodput:
                best_goodput = goodput
                best_k = k

        gammas = [best_k] * self.n_streams
        selected = list(range(self.n_streams))
        share = self.total_bandwidth_hz / self.n_streams
        bandwidth = [share] * self.n_streams
        return SpeculationDecision(
            gammas=gammas,
            selected_batch=selected,
            bandwidth_allocation_hz=bandwidth,
            lambda_price=0.0,
            mu_prices=[0.0] * self.n_streams,
        )

    def observe(self, accepted_counts: Sequence[int], gamma: int) -> None:
        """Feed back this slot's actual per-stream accepted-prefix counts to
        update the moving-average alpha_hat used by the next decide() call."""
        for accepted in accepted_counts:
            self.acceptance.observe(accepted, gamma)
