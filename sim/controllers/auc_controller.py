"""AUC-frontier controller: our proposed inner-problem (Px) runtime policy.

Drift-plus-penalty controller (Theorem 3) decomposed into three subproblems:
  (i)   device-side speculation index gamma_i(t), closed form (eq. 13),
        priced by broadcast queues lambda(t) = Q(t), mu_i(t) = Q_i(t);
  (ii)  server-side batching: knapsack on token budget, greedy by
        omega_i / (gamma_i + 1);
  (iii) radio allocation: square-root waterfilling.
Virtual queues Z_i(t), Q(t), Q_i(t) per eq. (12). V is a config parameter
(needed for Exp 3), never hardcoded. Uses UCB/Bernstein censored learning
for alpha_hat_i(t) (Theorem 4) unless disabled -- see auc_controller_oracle.py.
See TASKS.md T2.1, T2.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from ..core.acceptance import expected_tokens_per_round
from ..core.channel import sqrt_waterfilling
from ..metrics.virtual_queues import VirtualQueues, update_virtual_queues


class AlphaEstimator(Protocol):
    """hat-alpha_i(t) source plugged into eq. (13): UCBBernsteinEstimator
    below (Theorem 4) for the full controller, or
    auc_controller_oracle.StaticAlphaEstimator for the cheap oracle."""

    def estimate(self, stream_index: int, t: int) -> float: ...


def price_c(
    lambda_price: float,
    mu_price: float,
    theta_f_seconds_per_token: float,
    tau_d_seconds: float,
    kappa_bits_per_token: float,
    uplink_rate_bps: float,
    V: float,
    z_i: float,
) -> float:
    """c_i(t), eq. (13): ratio of the marginal cost of one more drafted
    token (server-compute price plus device/uplink-time price) to the
    marginal-benefit weight (V + Z_i(t))."""
    if uplink_rate_bps <= 0.0:
        raise ValueError("uplink_rate_bps must be positive")
    denom = V + z_i
    if denom <= 0.0:
        raise ValueError("V + Z_i(t) must be positive")
    marginal_cost = lambda_price * theta_f_seconds_per_token + mu_price * (
        tau_d_seconds + kappa_bits_per_token / uplink_rate_bps
    )
    return marginal_cost / denom


def speculation_index(c_i: float, alpha_hat_i: float, gamma_max: int) -> int:
    """gamma_i(t), eq. (13): closed-form optimal stopping rule for the
    device-side speculation length, clamped to {0,...,gamma_max}."""
    if gamma_max < 0:
        raise ValueError("gamma_max must be non-negative")
    alpha_hat_i = min(1.0 - 1e-12, max(1e-12, float(alpha_hat_i)))
    if c_i <= 0.0:
        return gamma_max
    if c_i >= 1.0:
        return 0
    gamma = math.ceil(math.log(c_i) / math.log(alpha_hat_i)) - 1
    return max(0, min(gamma_max, int(gamma)))


@dataclass(frozen=True)
class PendingVerification:
    stream_index: int
    gamma: int
    weight: float  # omega_i(t) = (V + Z_i(t)) * phi(gamma_i, alpha_hat_i)


def select_batch(pending: Sequence[PendingVerification], gamma_budget: float) -> list[int]:
    """Server-side batching, muc 1.6(ii): token-budget knapsack
    max sum_{i in S} omega_i(t) s.t. Gamma(S) <= Gamma_bud(t), solved by the
    greedy rule ranked by omega_i/(gamma_i+1) -- optimal for the LP
    relaxation and within one job of integer-optimal. Returns the selected
    stream indices, in selection order.
    """
    if gamma_budget < 0.0:
        raise ValueError("gamma_budget must be non-negative")
    ranked = sorted(pending, key=lambda p: p.weight / (p.gamma + 1), reverse=True)
    selected: list[int] = []
    used = 0.0
    for item in ranked:
        cost = item.gamma + 1
        if used + cost <= gamma_budget:
            selected.append(item.stream_index)
            used += cost
    return selected


def radio_allocation(
    mu_prices: Sequence[float],
    kappa_bits_per_token: float,
    gammas: Sequence[int],
    spectral_efficiencies: Sequence[float],
    total_bandwidth_hz: float,
) -> list[float]:
    """w_i*(t), muc 1.6(iii): square-root waterfilling over per-stream costs
    mu_i(t) * kappa * gamma_i(t) / g_i(t) (Cauchy-Schwarz)."""
    if not (len(mu_prices) == len(gammas) == len(spectral_efficiencies)):
        raise ValueError("mu_prices, gammas, spectral_efficiencies must have the same length")
    costs = [
        mu * kappa_bits_per_token * gamma / g if g > 0.0 else 0.0
        for mu, gamma, g in zip(mu_prices, gammas, spectral_efficiencies)
    ]
    return sqrt_waterfilling(costs, total_bandwidth_hz)


@dataclass
class UCBBernsteinEstimator:
    """Per-stream censored acceptance-rate estimator, Theorem 4 (TASKS.md T2.2).

    Each round with gamma_i>=1 reveals a truncated-geometric observation:
    the accepted prefix contributes Bernoulli successes and, unless every
    drafted token was accepted (right-censoring), one observed failure.
    hat-alpha_i^ucb(t) is an optimistic upper-confidence estimate built from
    an empirical-Bernstein interval on these samples; with no samples yet it
    defaults to alpha_max (pure optimism), which per Remark 3 is what lets
    a binding interactivity constraint force exploration without a separate
    forced-exploration mechanism.
    """

    n_streams: int
    alpha_min: float
    alpha_max: float

    def __post_init__(self) -> None:
        self._successes = [0] * self.n_streams
        self._trials = [0] * self.n_streams

    def observe(self, stream_index: int, accepted: int, all_accepted: bool) -> None:
        if accepted < 0:
            raise ValueError("accepted must be non-negative")
        self._successes[stream_index] += accepted
        self._trials[stream_index] += accepted + (0 if all_accepted else 1)

    def estimate(self, stream_index: int, t: int) -> float:
        trials = self._trials[stream_index]
        if trials == 0:
            return self.alpha_max
        p_hat = self._successes[stream_index] / trials
        log_term = math.log(max(math.e, float(t) + 2.0))
        width = math.sqrt(2.0 * p_hat * (1.0 - p_hat) * log_term / trials) + 3.0 * log_term / trials
        return min(self.alpha_max, max(self.alpha_min, p_hat + width))


@dataclass(frozen=True)
class StreamLinkState:
    """Per-stream, per-slot link parameters the controller needs to decide."""

    tau_d_seconds: float
    uplink_rate_bps: float
    spectral_efficiency: float


@dataclass(frozen=True)
class SpeculationDecision:
    gammas: list[int]
    selected_batch: list[int]
    bandwidth_allocation_hz: list[float]
    lambda_price: float
    mu_prices: list[float]


class AUCController:
    """eq. (12)-(13) drift-plus-penalty controller. V is a required
    constructor argument, never hardcoded (TASKS.md T2.1)."""

    def __init__(
        self,
        n_streams: int,
        gamma_max: int,
        V: float,
        min_interactivity_x: float,
        kappa_bits_per_token: float,
        theta_f_seconds_per_token: float,
        total_bandwidth_hz: float,
        gamma_budget: float,
        alpha_estimator: AlphaEstimator,
    ) -> None:
        if V <= 0.0:
            raise ValueError("V must be positive")
        if n_streams <= 0:
            raise ValueError("n_streams must be positive")
        self.n_streams = n_streams
        self.gamma_max = gamma_max
        self.V = float(V)
        self.min_interactivity_x = float(min_interactivity_x)
        self.kappa_bits_per_token = kappa_bits_per_token
        self.theta_f_seconds_per_token = theta_f_seconds_per_token
        self.total_bandwidth_hz = total_bandwidth_hz
        self.gamma_budget = gamma_budget
        self.alpha_estimator = alpha_estimator
        self.queues = VirtualQueues.zeros(n_streams)

    def decide(self, t: int, links: Sequence[StreamLinkState]) -> SpeculationDecision:
        if len(links) != self.n_streams:
            raise ValueError("links must have one entry per stream")
        lambda_price = self.queues.Q
        gammas: list[int] = []
        alpha_hats: list[float] = []
        for i, link in enumerate(links):
            mu_i = self.queues.Qi[i]
            c_i = price_c(
                lambda_price,
                mu_i,
                self.theta_f_seconds_per_token,
                link.tau_d_seconds,
                self.kappa_bits_per_token,
                link.uplink_rate_bps,
                self.V,
                self.queues.Z[i],
            )
            alpha_hat = self.alpha_estimator.estimate(i, t)
            alpha_hats.append(alpha_hat)
            gammas.append(speculation_index(c_i, alpha_hat, self.gamma_max))

        pending = [
            PendingVerification(
                stream_index=i,
                gamma=gammas[i],
                weight=(self.V + self.queues.Z[i]) * expected_tokens_per_round(gammas[i], alpha_hats[i]),
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
        """Apply eq. (12) after observing the slot's outcome."""
        device_time_per_gamma = [
            link.tau_d_seconds + self.kappa_bits_per_token / link.uplink_rate_bps for link in links
        ]
        self.queues = update_virtual_queues(
            self.queues,
            x_target=self.min_interactivity_x,
            a_i=tokens_delivered,
            theta_f_seconds_per_token=self.theta_f_seconds_per_token,
            verification_batch_tokens=verification_batch_tokens,
            gammas=decision.gammas,
            device_time_per_gamma=device_time_per_gamma,
            slot_seconds=slot_seconds,
        )
