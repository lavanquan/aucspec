"""Certified capacity measurement layer (C1: audit items M3, M5, M6, M7, M8).  CPU-only, no scipy.

This module REPLACES the decision path of `capacity.classify_feasibility_final`, `monotone_scale_search_final`,
`xstar_search`, `area_from_nstar` and `area_from_xstar` (kept in `capacity.py` only to read historical
results).  What changed and why:

M3  Classifier.  The percentile bootstrap of K run-seed values cannot exceed the sample range and under-covers
    badly for K <= 7 (per-client coverage 0.76/0.91/0.96 for K = 3/5/7 at a nominal 0.995; false-FEASIBLE
    12.3 % / 4.7 % / 1.8 % when the weakest client sits exactly at x).  The default procedure is now a
    Bonferroni Student-t bound, with a minimum certification sample size K_min, an UNCERTAIN outcome below
    it, one-sided quantiles for the two one-sided decisions (documented below), and optional alpha-spending
    for sequential looks and boundary probes.  The percentile bootstrap survives only as a descriptive
    sensitivity comparison (`bootstrap_sensitivity`), never as a certificate.

    ASSUMPTIONS of the certificate (stated, not hidden): (A1) run-seeds are independent given a FIXED
    population (population_seed / run_seed separation); (A2) per-client mean rates over the window are
    approximately normal across run-seeds (Student-t is exact under normality and only approximately valid
    otherwise; heavy-tailed behaviour is quantified in results/c1_20261020/*); (A3) the window is
    long enough that edge-round noise is small.  Under A1-A3 each decision has error probability <= delta_eff;
    the family-wise error over a whole SEARCH is controlled only if `n_looks` and `n_probes` are set to the
    realised numbers of sequential looks and probes (alpha-spending); otherwise the guarantee is per decision.

M5/M7  Boundaries.  Only CERTIFIED points are endpoints.  UNCERTAIN and INVALID points never act as bounds.
    The largest certified-feasible N is an UNCONDITIONAL lower bound on controller-achieved capacity.  A finite
    upper bound needs either an exhaustive certified-infeasible tail (unconditional) or the monotonicity
    assumption "feasibility of THIS controller is down-closed in N" (conditional; it can be falsified by the
    probes themselves, in which case the conditional bound is withdrawn and the bound is reported UNRESOLVED).

M6  Staircase integration.  Lower integral uses the value at the RIGHT end of each grid cell, upper integral
    the value at the LEFT end (N*(x) nonincreasing in x is itself an assumption and is checked).

M8  Dual consistency between N*(x) and x*(N): exact identities (conjugacy, area equality) are only asserted
    when their prerequisites hold; otherwise interval consistency is reported.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

MIN_CERTIFICATION_SEEDS = 5

FEASIBLE, INFEASIBLE, UNCERTAIN, INVALID = "feasible", "infeasible", "uncertain", "invalid"


# --------------------------------------------------------------------- Student-t (no scipy)
def _betacf(a: float, b: float, x: float) -> float:
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def student_t_ppf(p: float, df: float) -> float:
    """Inverse CDF by bisection on the exact CDF (|error| < 1e-10)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -student_t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while student_t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------- M3 classifier
@dataclass
class CertifiedDecision:
    status: str                          # feasible | infeasible | uncertain | invalid
    reason: str
    seeds_used: int
    min_lcb: float = float("nan")        # min_i L_i  (feasible iff >= x)
    min_ucb: float = float("nan")        # min_i U_i  (infeasible iff < x)
    per_client_lcb: dict[int, float] = field(default_factory=dict)
    per_client_ucb: dict[int, float] = field(default_factory=dict)
    delta: float = 0.05
    delta_effective: float = 0.05
    quantile: float = float("nan")
    sided: str = "one"
    screening_status: str | None = None  # point-estimate hint used only to decide whether to add seeds
    certified: bool = False              # True only if K >= K_min and status is feasible/infeasible
    degenerate_zero_variance_clients: list[int] = field(default_factory=list)
    assumptions: tuple[str, ...] = (
        "A1 run-seeds independent given a fixed population",
        "A2 per-client mean rate approx normal across run-seeds",
        "A3 measurement window long enough that edge-round noise is small",
    )


def classify_certified(
    measurements: Sequence,
    x: float,
    n_clients: int,
    *,
    delta: float = 0.05,
    sided: str = "one",
    min_seeds: int = MIN_CERTIFICATION_SEEDS,
    n_looks: int = 1,
    n_probes: int = 1,
    sd_floor_frac: float = 0.01,
) -> CertifiedDecision:
    """Bonferroni Student-t classifier.

    Per client i: K run-seed rates r_i1..r_iK, mean m_i, sd s_i.  With alpha_c = delta_eff / N,
      sided="one":  L_i = m_i - t_{1-alpha_c,K-1} s_i/sqrt(K),  U_i = m_i + t_{1-alpha_c,K-1} s_i/sqrt(K)
                    (each is a simultaneous ONE-SIDED bound at total error delta_eff; FEASIBLE uses only the L_i,
                    INFEASIBLE only the U_i, so each claim has error <= delta_eff),
      sided="two":  the quantile t_{1-alpha_c/2}: simultaneous two-sided intervals (conservative variant).
    FEASIBLE iff min_i L_i >= x; INFEASIBLE iff min_i U_i < x (exists i with mu_i < x); else UNCERTAIN.
    Fewer than `min_seeds` valid seeds -> UNCERTAIN (reason "insufficient_seeds"), with a screening hint.
    A per-client sd below `sd_floor_frac * x` is raised to that floor (a handful of identical values from a
    near-deterministic run must not produce a zero-width, overconfident interval); affected clients are listed.
    Any INVALID measurement -> INVALID (never classified).
    """
    if sided not in ("one", "two"):
        raise ValueError("sided must be 'one' or 'two'")
    if any(getattr(m, "invalid", False) for m in measurements):
        return CertifiedDecision(INVALID, "at least one measurement is INVALID (workload exhausted)",
                                 len(measurements), delta=delta)
    k = len(measurements)
    delta_eff = delta / max(1, n_looks * n_probes)
    if k < 2:
        return CertifiedDecision(UNCERTAIN, "insufficient_seeds (K<2: no variance estimate)", k, delta=delta,
                                 delta_effective=delta_eff, sided=sided)
    alpha_c = delta_eff / max(1, n_clients)
    q = student_t_ppf(1.0 - (alpha_c if sided == "one" else alpha_c / 2.0), k - 1)
    lcb, ucb, zero_var = {}, {}, []
    for cid in range(n_clients):
        vals = [float(m.per_client_rates.get(cid, 0.0)) for m in measurements]
        mean = sum(vals) / k
        var = sum((v - mean) ** 2 for v in vals) / (k - 1)
        sd = math.sqrt(var)
        floor = sd_floor_frac * abs(x)
        if sd < floor:
            zero_var.append(cid)
            sd = floor
        half = q * sd / math.sqrt(k)
        lcb[cid], ucb[cid] = mean - half, mean + half
    min_l, min_u = min(lcb.values()), min(ucb.values())
    if min_l >= x:
        status, reason = FEASIBLE, "min_i L_i >= x"
    elif min_u < x:
        status, reason = INFEASIBLE, "min_i U_i < x"
    else:
        status, reason = UNCERTAIN, "confidence bounds straddle x"
    screening = status
    certified = k >= min_seeds
    if not certified and status in (FEASIBLE, INFEASIBLE):
        status, reason = UNCERTAIN, f"insufficient_seeds (K={k} < K_min={min_seeds}); screening says {screening}"
    return CertifiedDecision(
        status=status, reason=reason, seeds_used=k, min_lcb=min_l, min_ucb=min_u, per_client_lcb=lcb,
        per_client_ucb=ucb, delta=delta, delta_effective=delta_eff, quantile=q, sided=sided,
        screening_status=screening, certified=certified and status in (FEASIBLE, INFEASIBLE),
        degenerate_zero_variance_clients=zero_var,
    )


def bootstrap_sensitivity(measurements: Sequence, x: float, n_clients: int, *, delta: float = 0.05,
                          n_boot: int = 2000, rng_seed: int = 0) -> dict:
    """DESCRIPTIVE sensitivity comparison (percentile bootstrap, per-client alpha = delta/N).  Not a certificate."""
    k = len(measurements)
    if k < 2 or any(getattr(m, "invalid", False) for m in measurements):
        return {"certificate": False, "status": INVALID if k else UNCERTAIN}
    rng = random.Random(rng_seed)
    a = delta / max(1, n_clients)
    lo_i, hi_i = int((a / 2) * (n_boot - 1)), int((1 - a / 2) * (n_boot - 1))
    lcb, ucb = [], []
    for cid in range(n_clients):
        vals = [float(m.per_client_rates.get(cid, 0.0)) for m in measurements]
        boots = sorted(sum(vals[rng.randrange(k)] for _ in range(k)) / k for _ in range(n_boot))
        lcb.append(boots[lo_i])
        ucb.append(boots[hi_i])
    status = FEASIBLE if min(lcb) >= x else (INFEASIBLE if min(ucb) < x else UNCERTAIN)
    return {"certificate": False, "status": status, "min_lcb": min(lcb), "min_ucb": min(ucb), "seeds": k}


# --------------------------------------------------------------------- M5/M7 boundaries
@dataclass
class BoundaryResult:
    """Outcome of an N-axis search at one requirement x.  `probes` maps N -> status of the certified decision."""

    x_requirement: float
    probes: dict[int, str]
    hi_cap: int
    lower_bound_n: int                       # largest certified-feasible N (0 if none): UNCONDITIONAL
    upper_bound_unconditional: int | None    # set only if every N in (lower, hi_cap] is certified infeasible
    upper_bound_monotone: int | None         # (smallest certified-infeasible N) - 1, valid only under assumption M
    monotone_violations: list[tuple[int, int]] = field(default_factory=list)   # (feasible N, smaller infeasible N)
    gray_ns: list[int] = field(default_factory=list)
    invalid_ns: list[int] = field(default_factory=list)
    hit_search_cap: bool = False
    status: str = "unresolved_upper"         # exact_conditional | bracketed | unresolved_upper | cap_limited |
    #                                          lo_not_feasible | invalid | nonmonotone_unresolved
    exact_conditional: bool = False
    assumption: str = ("M: feasibility of this controller is down-closed in N (existential regions are "
                       "monotone; controller-achieved feasibility need not be)")

    @property
    def upper_bound_n(self) -> int | None:
        """Best available finite upper bound: unconditional if resolved, else conditional on M, else None."""
        return self.upper_bound_unconditional if self.upper_bound_unconditional is not None else self.upper_bound_monotone


def summarize_probes(x: float, probes: dict[int, str], hi_cap: int) -> BoundaryResult:
    feas = sorted(n for n, s in probes.items() if s == FEASIBLE)
    inf = sorted(n for n, s in probes.items() if s == INFEASIBLE)
    gray = sorted(n for n, s in probes.items() if s == UNCERTAIN)
    inval = sorted(n for n, s in probes.items() if s == INVALID)
    lower = feas[-1] if feas else 0
    violations = [(nf, ni) for nf in feas for ni in inf if nf > ni]
    unconditional = lower if (lower < hi_cap and all(probes.get(n) == INFEASIBLE for n in range(lower + 1, hi_cap + 1))) else None
    monotone = (inf[0] - 1) if (inf and not violations) else None
    cap = bool(feas) and lower >= hi_cap
    if inval:
        status = "invalid"
    elif not feas:
        status = "lo_not_feasible"
    elif violations:
        status = "nonmonotone_unresolved"
    elif cap:
        status = "cap_limited"
    elif monotone is not None and inf[0] == lower + 1:
        status = "exact_conditional"
    elif monotone is not None:
        status = "bracketed"
    else:
        status = "unresolved_upper"
    return BoundaryResult(
        x_requirement=float(x), probes=dict(probes), hi_cap=int(hi_cap), lower_bound_n=lower,
        upper_bound_unconditional=unconditional, upper_bound_monotone=monotone, monotone_violations=violations,
        gray_ns=gray, invalid_ns=inval, hit_search_cap=cap, status=status,
        exact_conditional=(status == "exact_conditional"),
    )


def certified_n_search(
    decide: Callable[[int], str],
    *,
    lo: int = 1,
    hi_cap: int,
    hi_hint: int | None = None,
    x_requirement: float = 0.0,
) -> BoundaryResult:
    """Monotone-ASSUMING search (assumption M) that never uses UNCERTAIN/INVALID points as bounds.

    `decide(n)` returns a status string.  Exponential bracketing from the hint, then bisection; at an
    UNCERTAIN midpoint the neighbours inside the open bracket are probed alternately (mid+1, mid-1, ...)
    to find a certified point; if the whole open bracket is gray the search stops and reports it.  Afterwards,
    if no certified-infeasible N is known, the search probes upward (doubling) to find one or reach the cap.
    An INVALID probe aborts the search immediately.
    """
    if hi_cap < lo:
        raise ValueError("hi_cap must be >= lo")
    probes: dict[int, str] = {}

    def ev(n: int) -> str:
        if n not in probes:
            probes[n] = decide(n)
        return probes[n]

    def done() -> BoundaryResult:
        return summarize_probes(x_requirement, probes, hi_cap)

    s = ev(lo)
    if s == INVALID:
        return done()
    if s != FEASIBLE:
        return done()

    lo_f = lo
    probe = min(hi_cap, int(hi_hint)) if (hi_hint is not None and hi_hint > lo_f) else min(hi_cap, lo_f + 1)
    hi_n: int | None = None
    while True:
        s = ev(probe)
        if s == INVALID:
            return done()
        if s == FEASIBLE:
            lo_f = probe
            if probe >= hi_cap:
                return done()
            probe = min(hi_cap, max(probe + 1, probe * 2))
        else:
            hi_n = probe          # certified infeasible OR gray: either way, "not certified feasible"
            break

    def bisect(lo_f: int, hi_n: int) -> tuple[int, int, bool]:
        """Narrow (lo_f, hi_n) with certified points only.  Returns (lo_f, hi_n, aborted_by_invalid)."""
        while hi_n - lo_f > 1:
            mid = (lo_f + hi_n) // 2
            s = ev(mid)
            if s == INVALID:
                return lo_f, hi_n, True
            if s == FEASIBLE:
                lo_f = mid
            elif s == INFEASIBLE:
                hi_n = mid
            else:
                resolved = False
                order = []
                for d in range(1, hi_n - lo_f):
                    order.extend([mid + d, mid - d])
                for c in order:
                    if not (lo_f < c < hi_n):
                        continue
                    sc = ev(c)
                    if sc == INVALID:
                        return lo_f, hi_n, True
                    if sc == FEASIBLE:
                        lo_f, resolved = c, True
                        break
                    if sc == INFEASIBLE:
                        hi_n, resolved = c, True
                        break
                if not resolved:
                    break          # whole open bracket is gray
        return lo_f, hi_n, False

    lo_f, hi_n, aborted = bisect(lo_f, hi_n)
    if aborted:
        return done()

    if not any(v == INFEASIBLE for v in probes.values()) and lo_f < hi_cap:
        p = min(hi_cap, max(lo_f + 1, hi_n + 1))
        while True:
            s = ev(p)
            if s == INVALID or s == INFEASIBLE or p >= hi_cap:
                break
            p = min(hi_cap, p * 2)
        if probes.get(p) == INFEASIBLE:
            bisect(lo_f, p)        # tighten the certified-infeasible end inside the gray zone
    return done()


# --------------------------------------------------------------------- x-axis search
@dataclass
class XBoundaryResult:
    n_active: int
    probes: dict[float, str]
    lower_feasible_x: float | None           # largest certified-feasible x probed
    upper_infeasible_x: float | None         # smallest certified-infeasible x probed
    gray_xs: list[float] = field(default_factory=list)
    invalid_xs: list[float] = field(default_factory=list)
    monotone_violations: list[tuple[float, float]] = field(default_factory=list)   # (feasible x, smaller infeasible x)
    status: str = "unresolved"               # exact_within_tolerance | bracketed | lo_not_feasible | hi_not_infeasible
    #                                          | invalid | nonmonotone_unresolved
    exact_within_tolerance: bool = False
    x_tolerance_tps: float = 0.1


def summarize_x_probes(n: int, probes: dict[float, str], tol: float) -> XBoundaryResult:
    feas = sorted(x for x, s in probes.items() if s == FEASIBLE)
    inf = sorted(x for x, s in probes.items() if s == INFEASIBLE)
    gray = sorted(x for x, s in probes.items() if s == UNCERTAIN)
    inval = sorted(x for x, s in probes.items() if s == INVALID)
    lower = feas[-1] if feas else None
    violations = [(xf, xi) for xf in feas for xi in inf if xf > xi]
    upper = None if violations else (inf[0] if inf else None)
    if inval:
        status = "invalid"
    elif not feas:
        status = "lo_not_feasible"
    elif violations:
        status = "nonmonotone_unresolved"
    elif upper is None:
        status = "hi_not_infeasible"
    elif upper - lower <= tol + 1e-12:
        status = "exact_within_tolerance"
    else:
        status = "bracketed"
    return XBoundaryResult(n_active=n, probes=dict(probes), lower_feasible_x=lower, upper_infeasible_x=upper,
                           gray_xs=gray, invalid_xs=inval, monotone_violations=violations, status=status,
                           exact_within_tolerance=(status == "exact_within_tolerance"), x_tolerance_tps=tol)


def certified_x_search(decide: Callable[[float], str], *, n_active: int, x_lo: float, x_hi: float,
                       x_tolerance_tps: float = 0.10) -> XBoundaryResult:
    probes: dict[float, str] = {}

    def ev(x: float) -> str:
        key = round(float(x), 9)
        if key not in probes:
            probes[key] = decide(key)
        return probes[key]

    def done() -> XBoundaryResult:
        return summarize_x_probes(n_active, probes, x_tolerance_tps)

    s_lo = ev(x_lo)
    if s_lo == INVALID or s_lo != FEASIBLE:
        return done()
    s_hi = ev(x_hi)
    if s_hi in (INVALID, FEASIBLE, UNCERTAIN):
        return done()
    lo, hi = x_lo, x_hi
    while hi - lo > x_tolerance_tps:
        mid = 0.5 * (lo + hi)
        s = ev(mid)
        if s == INVALID:
            return done()
        if s == FEASIBLE:
            lo = mid
        elif s == INFEASIBLE:
            hi = mid
        else:                                    # gray: probe the quarter points once, else stop
            moved = False
            for frac in (0.25, 0.75):
                c = lo + frac * (hi - lo)
                sc = ev(c)
                if sc == INVALID:
                    return done()
                if sc == FEASIBLE:
                    lo, moved = c, True
                elif sc == INFEASIBLE:
                    hi, moved = c, True
                if moved:
                    break
            if not moved:
                break
    return done()


# --------------------------------------------------------------------- M6 area / ICA
def _cells(xs: list[float], x_L: float, x_U: float):
    """Yield (a, b, k) for each sub-interval of [x_L, x_U] inside grid cell [xs[k], xs[k+1]]."""
    if x_L < xs[0] - 1e-12 or x_U > xs[-1] + 1e-12:
        raise ValueError(f"grid [{xs[0]}, {xs[-1]}] does not cover [{x_L}, {x_U}]: no extrapolation")
    knots = sorted({x_L, x_U, *[x for x in xs if x_L < x < x_U]})
    for a, b in zip(knots, knots[1:]):
        k = max(i for i in range(len(xs) - 1) if xs[i] <= a + 1e-12)
        yield a, b, k


def area_from_boundaries(results: Sequence[BoundaryResult], *, x_L: float, x_U: float, n_ref: float) -> dict:
    """Valid staircase bounds on  A = integral_{x_L}^{x_U} min(N*(x), n_ref) dx.

    Conditions (reported): x-monotone ("N*(x) nonincreasing in x", checked on the lower bounds) and, for the
    upper bound, assumption M per point.  Points with an unresolved upper bound contribute n_ref to the upper
    integral.  Any INVALID point makes the whole area INVALID (no number).
    """
    pts = sorted(results, key=lambda r: r.x_requirement)
    if not pts:
        raise ValueError("need at least one boundary result")
    if any(r.status == "invalid" for r in pts):
        return {"status": "INVALID", "area_lower": None, "area_upper": None, "ica_lower": None, "ica_upper": None,
                "invalid_x": [r.x_requirement for r in pts if r.status == "invalid"]}
    xs = [r.x_requirement for r in pts]
    lo = [min(float(n_ref), float(r.lower_bound_n)) for r in pts]
    up = [min(float(n_ref), float(r.upper_bound_n)) if r.upper_bound_n is not None else float(n_ref) for r in pts]
    x_viol = [(xs[i], xs[i + 1]) for i in range(len(pts) - 1) if lo[i + 1] > lo[i]]
    a_lo = a_up = a_disc = 0.0
    for a, b, k in _cells(xs, x_L, x_U):
        w = b - a
        a_lo += min(lo[k], lo[k + 1]) * w        # right end (= min with left if monotonicity is violated)
        a_up += max(up[k], up[k + 1]) * w        # left end  (= max with right if monotonicity is violated)
        a_disc += lo[k] * w                      # staircase through exact lower values, left end
    width = float(n_ref) * (x_U - x_L)
    return {
        "status": "OK",
        "area_lower": a_lo, "area_upper": a_up, "ica_lower": a_lo / width, "ica_upper": a_up / width,
        "discretisation_gap": max(0.0, a_disc - a_lo),
        "boundary_uncertainty_gap": max(0.0, a_up - a_disc),
        "unresolved_upper_points": [r.x_requirement for r in pts if r.upper_bound_n is None],
        "nonmonotone_points": [r.x_requirement for r in pts if r.status == "nonmonotone_unresolved"],
        "x_monotonicity_violations": x_viol,
        "conditions": ["lower bound: N*(x) nonincreasing in x between grid points",
                       "upper bound: assumption M per grid point (unresolved points use n_ref)"],
    }


def area_from_x_boundaries(results: Sequence[XBoundaryResult], *, x_L: float, x_U: float, n_ref: int) -> dict:
    """Dual column sum  A = sum_{N=1}^{n_ref} [min(x*(N), x_U) - x_L]^+  with certified endpoints only."""
    by_n = {r.n_active: r for r in results}
    if any(r.status == "invalid" for r in results):
        return {"status": "INVALID", "area_lower": None, "area_upper": None}
    low = up = 0.0
    missing = []
    for n in range(1, int(n_ref) + 1):
        r = by_n.get(n)
        if r is None:
            missing.append(n)
            up += max(0.0, x_U - x_L)
            continue
        low += max(0.0, min(r.lower_feasible_x if r.lower_feasible_x is not None else x_L, x_U) - x_L)
        hi_x = r.upper_infeasible_x if r.upper_infeasible_x is not None else x_U
        up += max(0.0, min(hi_x, x_U) - x_L)
    width = float(n_ref) * (x_U - x_L)
    return {"status": "OK", "area_lower": low, "area_upper": up, "ica_lower": low / width, "ica_upper": up / width,
            "missing_n": missing}


# --------------------------------------------------------------------- M8 dual consistency
def dual_consistency(n_results: Sequence[BoundaryResult], x_results: Sequence[XBoundaryResult], *,
                     x_L: float, x_U: float, n_ref: int, tol: float = 1e-9) -> dict:
    """Compare the N-axis and x-axis reconstructions of the same frontier.

    Pairwise interval checks (always valid under monotonicity in BOTH axes of the controller-achieved region):
      N <= Nlo(x_k)  (certified feasible at x_k)   =>  x*(N) >= x_k   i.e.  x_up(N) must not be < x_k
      N >  Nup(x_k)  (certified infeasible, M)      =>  x*(N) <  x_k   i.e.  x_lo(N) must not be >= x_k
    Exact identities (conjugacy N*(x) >= N <=> x*(N) >= x and area equality) are asserted only when every
    point is exact (exact_conditional / exact_within_tolerance), no gray/invalid/violations exist and the
    N-axis and x-axis grids are complete; otherwise only overlap of the two area intervals is reported.
    """
    bad: list[str] = []
    xr = {r.n_active: r for r in x_results}
    for nr in n_results:
        xk = nr.x_requirement
        for n in range(1, int(n_ref) + 1):
            r = xr.get(n)
            if r is None:
                continue
            if n <= nr.lower_bound_n and r.upper_infeasible_x is not None and r.upper_infeasible_x < xk - tol:
                bad.append(f"N={n} certified feasible at x={xk} but x*(N) upper {r.upper_infeasible_x} < x")
            ub = nr.upper_bound_n
            if ub is not None and n > ub and r.lower_feasible_x is not None and r.lower_feasible_x >= xk + tol:
                bad.append(f"N={n} certified infeasible at x={xk} but x*(N) lower {r.lower_feasible_x} >= x")
    a_n = area_from_boundaries(n_results, x_L=x_L, x_U=x_U, n_ref=n_ref) if n_results else None
    a_x = area_from_x_boundaries(x_results, x_L=x_L, x_U=x_U, n_ref=n_ref) if x_results else None
    overlap = None
    if a_n and a_x and a_n["status"] == "OK" and a_x["status"] == "OK":
        overlap = (a_n["area_lower"] <= a_x["area_upper"] + tol) and (a_x["area_lower"] <= a_n["area_upper"] + tol)
        if not overlap:
            bad.append("area intervals from the two axes do not overlap")
    prereq = bool(
        n_results and x_results and
        all(r.exact_conditional for r in n_results) and all(r.exact_within_tolerance for r in x_results) and
        not any(r.gray_ns or r.invalid_ns or r.monotone_violations for r in n_results) and
        not any(r.gray_xs or r.invalid_xs or r.monotone_violations for r in x_results) and
        set(range(1, int(n_ref) + 1)) <= set(xr)
    )
    exact_check = None
    if prereq and a_n and a_x:
        # with all points exact, the N-axis bounds collapse onto the staircase and the x-axis onto columns
        exact_check = {
            "area_N_axis": (a_n["area_lower"], a_n["area_upper"]),
            "area_x_axis": (a_x["area_lower"], a_x["area_upper"]),
            "note": "equality holds only up to grid/tolerance discretisation; compare via interval overlap",
        }
    return {"consistent": not bad, "violations": bad, "interval_overlap": overlap,
            "exact_identities_asserted": prereq, "exact_check": exact_check,
            "area_N_axis": a_n, "area_x_axis": a_x}
