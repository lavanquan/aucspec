"""GoodSpeed baseline: proportional-fair, log-utility policy.

Single fixed operating point on the AUC frontier -- the degenerate case
w = delta_{x0} in Proposition 1. Most important baseline; reuse the prior
GoodSpeed paper's code/results where available. See TASKS.md T3.2.

Per the formulation's "Relation to GoodSpeed" remark: GoodSpeed solves
max_x sum_i log(x_i), one tangency point of auc_controller.py's dual sweep
(Proposition 2(iv)) at multiplier eta_i(t) = 1/x_hat_i(t). We do not
re-derive a separate algorithm -- this file reuses auc_controller.py's
price_c/speculation_index/select_batch/radio_allocation machinery verbatim,
with two differences from AUCController:
  - the per-stream weight passed into price_c/speculation_index is
    1/x_hat_i(t) (each stream's own running-average interactivity), not the
    shared Lyapunov weight (V + Z_i(t));
  - there is no interactivity-floor constraint x and no Z_i(t) queue -- only
    the server/device-time queues Q(t), Q_i(t) create scarcity pricing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .auc_controller import (
    PendingVerification,
    SpeculationDecision,
    StreamLinkState,
    price_c,
    radio_allocation,
    select_batch,
    speculation_index,
)
from ..core.acceptance import expected_tokens_per_round
from ..metrics.virtual_queues import VirtualQueues, update_virtual_queues


@dataclass
class GoodSpeedController:
    """Proportional-fair (log-utility) baseline. Weight per stream is
    1/x_hat_i(t), the reciprocal of that stream's own running-average
    interactivity -- exactly d(log x_i)/dx_i, the marginal utility a
    log-utility (proportional-fair) allocation prices speculation by."""

    n_streams: int
    gamma_max: int
    kappa_bits_per_token: float
    theta_f_seconds_per_token: float
    total_bandwidth_hz: float
    gamma_budget: float
    alpha_hat: list[float]

    def __post_init__(self) -> None:
        if self.n_streams <= 0:
            raise ValueError("n_streams must be positive")
        self.queues = VirtualQueues.zeros(self.n_streams)  # Z unused (no x floor); Q, Qi price scarcity
        self._cumulative_tokens = [0.0] * self.n_streams
        self._elapsed_seconds = 1e-9  # avoid a divide-by-zero weight before any round has run

    def _running_x(self, i: int) -> float:
        return max(1e-6, self._cumulative_tokens[i] / self._elapsed_seconds)

    def decide(self, t: int, links: Sequence[StreamLinkState]) -> SpeculationDecision:
        if len(links) != self.n_streams:
            raise ValueError("links must have one entry per stream")
        lambda_price = self.queues.Q
        gammas: list[int] = []
        for i, link in enumerate(links):
            weight = 1.0 / self._running_x(i)  # d(log x_i)/dx_i
            mu_i = self.queues.Qi[i]
            c_i = price_c(
                lambda_price,
                mu_i,
                self.theta_f_seconds_per_token,
                link.tau_d_seconds,
                self.kappa_bits_per_token,
                link.uplink_rate_bps,
                V=weight,  # reuses eq.(13)'s (V + Z_i) slot for the fairness weight; Z_i is always 0 here
                z_i=0.0,
            )
            gammas.append(speculation_index(c_i, self.alpha_hat[i], self.gamma_max))

        pending = [
            PendingVerification(
                stream_index=i,
                gamma=gammas[i],
                weight=(1.0 / self._running_x(i)) * expected_tokens_per_round(gammas[i], self.alpha_hat[i]),
            )
            for i in range(self.n_streams)
        ]
        selected = select_batch(pending, self.gamma_budget)

        mu_prices = list(self.queues.Qi)
        spectral_effs = [link.spectral_efficiency for link in links]
        bandwidth = radio_allocation(
            mu_prices, self.kappa_bits_per_token, gammas, spectral_effs, self.total_bandwidth_hz
        )

        return SpeculationDecision(
            gammas=gammas,
            selected_batch=selected,
            bandwidth_allocation_hz=bandwidth,
            lambda_price=lambda_price,
            mu_prices=mu_prices,
        )

    def update_queues(
        self,
        decision: SpeculationDecision,
        tokens_delivered: Sequence[float],
        links: Sequence[StreamLinkState],
        verification_batch_tokens: float,
        slot_seconds: float,
    ) -> None:
        for i, tokens in enumerate(tokens_delivered):
            self._cumulative_tokens[i] += tokens
        self._elapsed_seconds += slot_seconds

        device_time_per_gamma = [
            link.tau_d_seconds + self.kappa_bits_per_token / link.uplink_rate_bps for link in links
        ]
        self.queues = update_virtual_queues(
            self.queues,
            x_target=0.0,  # no interactivity floor -- keeps Z at 0, unused by this controller
            a_i=tokens_delivered,
            theta_f_seconds_per_token=self.theta_f_seconds_per_token,
            verification_batch_tokens=verification_batch_tokens,
            gammas=decision.gammas,
            device_time_per_gamma=device_time_per_gamma,
            slot_seconds=slot_seconds,
        )
