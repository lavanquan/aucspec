"""Symmetric gated reference model of AUC_SPEC Section 3 (eqs. 29-51).

CPU-only. N statistically identical streams at a common depth gamma and
scalar acceptance alpha; draft -> shared uplink -> one verifier batch of all
N proposals -> next cycle. Not a model of the asynchronous system: it gives
the testable predictions (roofline knee, ~1/x capacity in a loaded regime,
gamma* changes at regime boundaries) that the measured N*(x) / x*(N) curves
are compared against.

Units: times in ms, rates in tokens/s, bandwidth in Hz, payload in bits.
`nstar_symmetric.py` is the compute-bound-only special case of this model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def phi(gamma: int, alpha: float) -> float:
    """Eq. (29): expected useful committed tokens per round, phi(gamma, 1) = gamma + 1."""
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    a = min(1.0, max(0.0, float(alpha)))
    if a >= 1.0:
        return float(gamma + 1)
    return (1.0 - a ** (gamma + 1)) / (1.0 - a)


@dataclass(frozen=True)
class RooflineParams:
    """Eqs. (35)-(37), profiled rather than nominal hardware values.

    theta_0_ms       fixed verifier overhead (not network RTT)
    a_mem_ms         b_W / beta_mem: weight traffic time
    kv_ms_per_ctx    b_KV / beta_mem: KV traffic time per context token
    theta_f_ms       compute time per verifier-token position
    """

    theta_0_ms: float
    a_mem_ms: float
    kv_ms_per_ctx: float
    theta_f_ms: float

    def t_mem_ms(self, n: int, context_length: float) -> float:
        """Eq. (36)."""
        return self.a_mem_ms + self.kv_ms_per_ctx * n * context_length

    def t_comp_ms(self, n: int, gamma: int) -> float:
        """Eq. (37)."""
        return self.theta_f_ms * n * (gamma + 1)

    def t_v_ms(self, n: int, gamma: int, context_length: float) -> float:
        """Eq. (35)."""
        return self.theta_0_ms + max(self.t_mem_ms(n, context_length), self.t_comp_ms(n, gamma))

    def regime(self, n: int, gamma: int, context_length: float) -> str:
        if self.t_comp_ms(n, gamma) > self.t_mem_ms(n, context_length):
            return "compute"
        return "memory"


@dataclass(frozen=True)
class SymmetricSystem:
    """Everything that is fixed by the deployment xi in the gated model."""

    alpha: float
    gamma_choices: Sequence[int]
    roofline: RooflineParams
    context_length: float
    draft_ms_per_token: float = 0.0     # tau_d
    draft_fixed_ms: float = 0.0
    rtt_ms: float = 0.0                 # fixed network delay
    kappa_bits: float = 0.0             # uplink payload per speculative token
    bandwidth_hz: float = math.inf      # W, shared uplink
    spectral_efficiency: float = 1.0    # s, bit/s/Hz

    # ---- cycle-time terms -------------------------------------------------
    def t_base_ms(self, gamma: int) -> float:
        """Terms that do not scale with N (draft latency, fixed network delay)."""
        return self.draft_fixed_ms + self.draft_ms_per_token * gamma + self.rtt_ms

    def c_net_ms(self, gamma: int) -> float:
        """Eq. (40): per-stream communication slope gamma*kappa/(W s), in ms."""
        if math.isinf(self.bandwidth_hz):
            return 0.0
        return 1000.0 * gamma * self.kappa_bits / (self.bandwidth_hz * self.spectral_efficiency)

    def t_net_ms(self, n: int, gamma: int) -> float:
        """Eq. (30)."""
        return n * self.c_net_ms(gamma)

    def t_cyc_ms(self, n: int, gamma: int) -> float:
        """Eq. (31)."""
        return (
            self.t_base_ms(gamma)
            + self.t_net_ms(n, gamma)
            + self.roofline.t_v_ms(n, gamma, self.context_length)
        )

    # ---- rates and boundaries ---------------------------------------------
    def x_gamma(self, n: int, gamma: int) -> float:
        """Eq. (32), tokens/s."""
        return 1000.0 * phi(gamma, self.alpha) / self.t_cyc_ms(n, gamma)

    def x_sym(self, n: int) -> tuple[float, int]:
        """Eq. (33) and (51): best common rate and the depth attaining it
        (ties -> smaller depth)."""
        best_x, best_g = -math.inf, 0
        for g in sorted(self.gamma_choices):
            x = self.x_gamma(n, g)
            if x > best_x + 1e-12:
                best_x, best_g = x, int(g)
        return best_x, best_g

    def n_sym(self, x: float, n_max: int) -> int:
        """Eq. (34) by direct scan up to n_max (exact for any roofline shape).
        Returns n_max when every N <= n_max is feasible."""
        if x <= 0:
            raise ValueError("x must be positive")
        best = 0
        for n in range(1, int(n_max) + 1):
            if self.x_sym(n)[0] >= x:
                best = n
        return best

    # ---- roofline knee and affine regimes ---------------------------------
    def c_mem_ms(self) -> float:
        """Eq. (38)."""
        return self.roofline.kv_ms_per_ctx * self.context_length

    def c_comp_ms(self, gamma: int) -> float:
        """Eq. (38)."""
        return self.roofline.theta_f_ms * (gamma + 1)

    def n_knee(self, gamma: int) -> float | None:
        """Eq. (39); None when c_comp(gamma) <= c_mem (no compute crossing)."""
        slope = self.c_comp_ms(gamma) - self.c_mem_ms()
        if slope <= 0:
            return None
        return self.roofline.a_mem_ms / slope

    def affine(self, gamma: int, regime: str) -> tuple[float, float]:
        """(A_gamma, c_gamma) of eq. (43) for one regime: eqs. (41)/(42)."""
        if regime == "memory":
            return (
                self.t_base_ms(gamma) + self.roofline.theta_0_ms + self.roofline.a_mem_ms,
                self.c_net_ms(gamma) + self.c_mem_ms(),
            )
        if regime == "compute":
            return (
                self.t_base_ms(gamma) + self.roofline.theta_0_ms,
                self.c_net_ms(gamma) + self.c_comp_ms(gamma),
            )
        raise ValueError("regime must be 'memory' or 'compute'")

    def n_gamma_closed_form(self, x: float, gamma: int, regime: str) -> int:
        """Proposition 1, eq. (45): floor((phi/x - A)/c)_+, valid while the
        maximizing N stays inside `regime`."""
        if x <= 0:
            raise ValueError("x must be positive")
        a_ms, c_ms = self.affine(gamma, regime)
        if c_ms <= 0:
            raise ValueError("Proposition 1 needs c_gamma > 0")
        cycle_budget_ms = 1000.0 * phi(gamma, self.alpha) / x
        return max(0, math.floor((cycle_budget_ms - a_ms) / c_ms))

    def gamma_transitions(self, n_max: int) -> list[dict]:
        """N at which argmax_gamma x_gamma(N) changes (Section 3.4), with the
        verifier regime on each side."""
        out = []
        prev = self.x_sym(1)[1]
        for n in range(2, int(n_max) + 1):
            g = self.x_sym(n)[1]
            if g != prev:
                out.append({
                    "n": n, "gamma_before": prev, "gamma_after": g,
                    "regime_before": self.roofline.regime(n - 1, prev, self.context_length),
                    "regime_after": self.roofline.regime(n, g, self.context_length),
                })
                prev = g
        return out


def fit_roofline(
    batch_size: Sequence[float],
    verifier_tokens: Sequence[float],
    context_tokens: Sequence[float],
    wall_ms: Sequence[float],
    *,
    max_iter: int = 50,
) -> dict:
    """Least-squares fit of eqs. (35)-(37) to measured verify calls:

        T = theta_0 + max(a_mem + kv * S_ctx,  theta_f * Gamma)

    with S_ctx = total context tokens in the batch (= N L in the symmetric
    model) and Gamma = total verifier-token positions. Both branches are
    linear given an assignment of points to a regime, so the fit alternates
    a joint least-squares solve (shared theta_0) with reassigning each point
    to the branch that predicts the larger time. `batch_size` is only used
    for the per-regime summary.
    """
    import numpy as np

    b = np.asarray(batch_size, dtype=float)
    g = np.asarray(verifier_tokens, dtype=float)
    s = np.asarray(context_tokens, dtype=float)
    t = np.asarray(wall_ms, dtype=float)
    if not (len(b) == len(g) == len(s) == len(t)) or len(t) < 4:
        raise ValueError("need >= 4 measurements of equal length")

    def solve(mem_mask):
        # columns: theta_0, a_mem, kv, theta_f
        x = np.zeros((len(t), 4))
        x[:, 0] = 1.0
        x[mem_mask, 1] = 1.0
        x[mem_mask, 2] = s[mem_mask]
        x[~mem_mask, 3] = g[~mem_mask]
        keep = np.ones(4, dtype=bool)
        if mem_mask.all():
            keep[3] = False
        if not mem_mask.any():
            keep[1] = keep[2] = False
        coef = np.zeros(4)
        coef[keep] = np.linalg.lstsq(x[:, keep], t, rcond=None)[0]
        return np.maximum(coef, 0.0)

    def predict(coef):
        mem = coef[1] + coef[2] * s
        comp = coef[3] * g
        return coef[0] + np.maximum(mem, comp), mem >= comp

    # start: points with the most verifier work per context token are compute-bound
    ratio = g / np.maximum(s, 1.0)
    mem_mask = ratio <= np.median(ratio)
    for _ in range(max_iter):
        coef = solve(mem_mask)
        _, new_mask = predict(coef)
        if (new_mask == mem_mask).all():
            break
        mem_mask = new_mask
    coef = solve(mem_mask)
    pred, mem_mask = predict(coef)
    resid = t - pred
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return {
        "theta_0_ms": float(coef[0]),
        "a_mem_ms": float(coef[1]),
        "kv_ms_per_ctx": float(coef[2]),
        "theta_f_ms": float(coef[3]),
        "rmse_ms": float(math.sqrt(float((resid ** 2).mean()))),
        "r2": float(1.0 - float((resid ** 2).sum()) / ss_tot) if ss_tot > 0 else float("nan"),
        "n_points": int(len(t)),
        "n_memory_bound": int(mem_mask.sum()),
        "n_compute_bound": int((~mem_mask).sum()),
        "max_batch_size": float(b.max()),
        "max_verifier_tokens": float(g.max()),
        "compute_regime_sampled": bool((~mem_mask).sum() >= 3),
    }
