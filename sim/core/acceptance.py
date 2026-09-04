"""Bernoulli(alpha) acceptance generator: stationary and trace-driven alpha(t).

See docs/notation.md eq. (2), Assumption 1. TASKS.md T1.2. Trace-driven mode
backs the misspecification check in EXPERIMENTS.md (Assumption 1 robustness).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


def expected_tokens_per_round(gamma: int, alpha: float) -> float:
    """phi(gamma, alpha) = (1 - alpha^(gamma+1)) / (1 - alpha), eq. (2).

    E[A] where A is the number of tokens delivered in a round (accepted
    prefix plus one bonus token) when gamma tokens are drafted and each is
    accepted i.i.d. Bernoulli(alpha).
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    alpha = float(alpha)
    if not (0.0 <= alpha < 1.0):
        if alpha == 1.0:
            # Limit alpha -> 1: every draft is accepted, phi -> gamma + 1.
            return float(gamma + 1)
        raise ValueError("alpha must be in [0, 1]")
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def sample_round(gamma: int, alpha: float, rng: random.Random) -> tuple[int, bool]:
    """Draw one round's outcome under Assumption 1.

    Draws gamma i.i.d. Bernoulli(alpha) acceptance events and returns
    (tokens_delivered, all_accepted). tokens_delivered is the truncated
    geometric accepted-prefix length plus the +1 bonus token (eq. 2's A);
    all_accepted is True iff every one of the gamma drafts was accepted
    (the right-censoring case used by Theorem 4's censored-learning
    estimator: no failure was observed this round).
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    accepted = 0
    for _ in range(gamma):
        if rng.random() < alpha:
            accepted += 1
        else:
            break
    all_accepted = accepted == gamma
    tokens_delivered = accepted + 1  # +1 bonus token (correction or extra decode)
    return tokens_delivered, all_accepted


@dataclass
class StationaryAcceptance:
    """Per-stream Bernoulli(alpha) generator with a fixed, known alpha."""

    alpha: float
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.alpha) <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        self._rng = random.Random(self.seed)

    def sample(self, gamma: int) -> tuple[int, bool]:
        return sample_round(gamma, self.alpha, self._rng)

    def expected(self, gamma: int) -> float:
        return expected_tokens_per_round(gamma, self.alpha)


@dataclass
class TraceAcceptance:
    """Trace-driven alpha(t): a time-indexed sequence of acceptance rates
    measured from real accept/reject logs (TASKS.md T4.1-T4.3), used for the
    misspecification check (Assumption 1 robustness, EXPERIMENTS.md).

    `alpha_by_round` gives alpha_i(t) for each round index t; `window` is the
    sliding-window size (rounds) used when this alpha(t) was itself measured
    from raw accept/reject observations via `sliding_window_alpha`.
    """

    alpha_by_round: list[float]
    seed: int = 0
    window: int = 50
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for a in self.alpha_by_round:
            if not (0.0 <= float(a) <= 1.0):
                raise ValueError("every alpha_by_round entry must be in [0, 1]")
        self._rng = random.Random(self.seed)

    def alpha_at(self, round_index: int) -> float:
        if not self.alpha_by_round:
            raise ValueError("alpha_by_round is empty")
        clamped = min(max(round_index, 0), len(self.alpha_by_round) - 1)
        return self.alpha_by_round[clamped]

    def sample(self, gamma: int, round_index: int) -> tuple[int, bool]:
        return sample_round(gamma, self.alpha_at(round_index), self._rng)


def sliding_window_alpha(accept_flags: list[bool], window: int) -> list[float]:
    """Measure alpha_i(t) over a trailing sliding window of `window` token-level
    accept/reject observations. Returns one estimate per input position (using
    a shorter window at the start), for plotting alpha_i(t) drift (EXPERIMENTS.md
    Exp misspecification, Hinh 5a).
    """
    if window <= 0:
        raise ValueError("window must be positive")
    estimates: list[float] = []
    for idx in range(len(accept_flags)):
        start = max(0, idx + 1 - window)
        chunk = accept_flags[start : idx + 1]
        estimates.append(sum(1 for f in chunk if f) / len(chunk))
    return estimates
