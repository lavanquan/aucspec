"""Symmetric analytical N*(x) sanity model.

CAPACITY_AUC_NSTAR_IMPLEMENTATION.md Section 15. CPU-only. Not a claim
about the asynchronous real system — a structural check that the new
capacity metric has the `Theta(1/x)` scaling the aggregate-goodput
rectangle lacked, and that `gamma*` shifts across roofline regimes.

For N identical clients in a gated/synchronous, compute-bound verifier
approximation:

    x(N, gamma) = phi(gamma, alpha) / ( T0(gamma) + theta_f * N * (gamma + 1) )

with T0(gamma) collecting draft + network + non-amortised terms. Feasibility
of a common floor x needs x <= x(N, gamma) for some gamma, hence

    N*(x) ~= max_gamma  floor( ( phi(gamma, alpha)/x - T0(gamma) )
                                / ( theta_f * (gamma + 1) ) )_+
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence


def phi(gamma: int, alpha: float) -> float:
    """Expected useful committed tokens from speculation depth `gamma`
    under i.i.d. scalar acceptance `alpha` (Section 3.2)."""
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    a = min(1.0, max(0.0, float(alpha)))
    if gamma == 0:
        return 1.0
    if a >= 1.0:
        return float(gamma + 1)
    return (1.0 - a ** (gamma + 1)) / (1.0 - a)


@dataclass(frozen=True)
class SymmetricParams:
    alpha: float
    gamma_choices: Sequence[int]
    theta_f_s_per_token: float          # compute-bound per-verification-token time (s)
    theta0_s: float = 0.0               # verifier fixed overhead (s), NOT network RTT
    # T0(gamma): draft + network + non-amortised per-round latency (s). Default
    # is a simple affine draft+RTT model; callers pass their own for realism.
    t0_fn: Callable[[int], float] | None = None
    draft_s_per_token: float = 0.0
    rtt_s: float = 0.0

    def t0(self, gamma: int) -> float:
        if self.t0_fn is not None:
            return float(self.t0_fn(gamma))
        return self.theta0_s + self.rtt_s + self.draft_s_per_token * max(0, gamma)


def x_of_n_gamma(n: int, gamma: int, p: SymmetricParams) -> float:
    """Sustained per-client rate for `n` identical clients at depth `gamma`."""
    denom = p.t0(gamma) + p.theta_f_s_per_token * n * (gamma + 1)
    if denom <= 0.0:
        return math.inf
    return phi(gamma, p.alpha) / denom


def nstar_of_gamma(x: float, gamma: int, p: SymmetricParams) -> int:
    """Largest `n` with `x_of_n_gamma(n, gamma) >= x` (closed form)."""
    if x <= 0.0:
        raise ValueError("x must be positive (Section 0: do not integrate to 0)")
    num = phi(gamma, p.alpha) / x - p.t0(gamma)
    if num <= 0.0:
        return 0
    return max(0, int(math.floor(num / (p.theta_f_s_per_token * (gamma + 1)))))


def nstar(x: float, p: SymmetricParams) -> tuple[int, int]:
    """N*(x) = max over gamma of nstar_of_gamma; returns (N*, argmax gamma)."""
    best_n, best_gamma = -1, 0
    for g in p.gamma_choices:
        n = nstar_of_gamma(x, g, p)
        if n > best_n:
            best_n, best_gamma = n, int(g)
    return max(0, best_n), best_gamma


def nstar_curve(x_grid: Sequence[float], p: SymmetricParams) -> list[dict]:
    rows = []
    for x in x_grid:
        n, g = nstar(x, p)
        rows.append({"x": float(x), "nstar": n, "argmax_gamma": g})
    return rows


def scaling_exponent(x_grid: Sequence[float], p: SymmetricParams) -> float:
    """Fit log N*(x) ~ s * log(1/x) over the region where N* > 0 and
    theta_f dominates. Section 15 expects s ~ 1 (Theta(1/x))."""
    pts = [(x, nstar(x, p)[0]) for x in x_grid]
    pts = [(x, n) for x, n in pts if n > 0]
    if len(pts) < 3:
        raise ValueError("need >=3 points with N*>0 to fit a scaling exponent")
    lx = [math.log(1.0 / x) for x, _ in pts]
    ln = [math.log(n) for _, n in pts]
    m = len(lx)
    mean_lx = sum(lx) / m
    mean_ln = sum(ln) / m
    sxx = sum((v - mean_lx) ** 2 for v in lx)
    sxy = sum((a - mean_lx) * (b - mean_ln) for a, b in zip(lx, ln))
    return sxy / sxx if sxx > 1e-12 else float("nan")
