"""N* capacity-search primitives (CPU-only, unit-testable).

Implements the outer-layer pieces of CAPACITY_AUC_NSTAR_IMPLEMENTATION.md:

- Section 11  sustained per-client service rate with a COMMON denominator
- Section 12  FeasibilityResult + queue-tail-slope estimator + classifier
- Section 13  monotone binary / exponential N (or m) search with caching
- Section 14  integer step-area bounds and normalized ICA

None of this imports the simulator or vLLM; the GPU run is injected as a
callable `run_candidate(scale, x) -> CandidateMeasurement`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence


# --------------------------------------------------------------------------- 11
def sustained_rates(
    useful_tokens_by_client: dict[int, float],
    admitted_client_ids: Iterable[int],
    t_measurement_s: float,
) -> dict[int, float]:
    """Section 11: x_hat_i = (useful committed tokens to client i during the
    measurement window) / T_measurement, with the SAME denominator for every
    admitted client. An admitted client that received no tokens has rate 0
    and is NOT dropped from the minimum.
    """
    if t_measurement_s <= 0.0:
        raise ValueError("t_measurement_s must be positive")
    rates: dict[int, float] = {}
    for cid in admitted_client_ids:
        rates[int(cid)] = float(useful_tokens_by_client.get(int(cid), 0.0)) / t_measurement_s
    return rates


def rate_summary(rates: dict[int, float]) -> dict[str, float]:
    vals = sorted(rates.values())
    if not vals:
        return {k: 0.0 for k in ("min", "p10", "median", "mean", "max")}
    n = len(vals)

    def q(p: float) -> float:
        if n == 1:
            return vals[0]
        pos = p * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return vals[lo] * (1 - frac) + vals[hi] * frac

    return {
        "min": vals[0],
        "p10": q(0.10),
        "median": q(0.50),
        "mean": sum(vals) / n,
        "max": vals[-1],
    }


# --------------------------------------------------------------------------- 12
def tail_slope(times_s: Sequence[float], values: Sequence[float], last_fraction: float = 0.5) -> float:
    """Least-squares slope (units: value per second) over the last
    `last_fraction` of the series. Section 12.1: used on Z_i(t), Q_s(t),
    Q_i(t) to detect a persistent positive drift => infeasible.
    """
    if len(times_s) != len(values):
        raise ValueError("times and values must have equal length")
    n = len(values)
    if n < 3:
        return 0.0
    start = int(n * (1.0 - last_fraction))
    start = max(0, min(start, n - 2))
    ts = [float(t) for t in times_s[start:]]
    vs = [float(v) for v in values[start:]]
    m = len(ts)
    mean_t = sum(ts) / m
    mean_v = sum(vs) / m
    sxx = sum((t - mean_t) ** 2 for t in ts)
    if sxx <= 1e-12:
        return 0.0
    sxy = sum((t - mean_t) * (v - mean_v) for t, v in zip(ts, vs))
    return sxy / sxx


@dataclass
class CandidateMeasurement:
    """What one GPU run of a fixed (scale/N, x) reports, per seed."""

    scale: int
    n_active: int
    x_requirement: float
    seed: int
    min_rate_tps: float
    per_client_rates: dict[int, float] = field(default_factory=dict)
    max_z_slope: float = 0.0            # tokens/s per s, worst interactivity-deficit queue
    max_device_queue_slope: float = 0.0
    server_queue_slope: float = 0.0
    goodput_tps: float = 0.0
    mean_gamma: float = 0.0
    mean_batch_size: float = 0.0
    mean_fill_ratio: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class FeasibilityResult:
    status: str                         # "feasible" | "infeasible" | "uncertain"
    min_rate_tps: float
    rate_margin_tps: float              # conservative min_rate - x
    max_z_slope: float
    max_device_queue_slope: float
    server_queue_slope: float
    seeds_used: int
    reason: str
    per_seed: list[dict] = field(default_factory=list)


def _median(values: Sequence[float]) -> float:
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0
    mid = n // 2
    return v[mid] if n % 2 else 0.5 * (v[mid - 1] + v[mid])


def classify_feasibility(
    measurements: Sequence[CandidateMeasurement],
    x: float,
    *,
    rate_tolerance_fraction: float = 0.03,
    queue_slope_tolerance: float = 1e-3,
) -> FeasibilityResult:
    """Section 12.2. Robust across seeds: the lower estimate of the
    minimum sustained rate is the WORST seed (conservative, distribution
    free) and the upper estimate is the median. This replaced a mean +/- 1
    standard error band that was too twitchy at 2-3 seeds and produced
    spurious "uncertain" verdicts near a thin margin. A point strictly
    inside the tolerance band -> "uncertain" (caller adds seeds).
    """
    if not measurements:
        raise ValueError("need at least one measurement")
    rates = [float(m.min_rate_tps) for m in measurements]
    k = len(rates)
    mean_r = sum(rates) / k
    lower = min(rates)            # worst seed
    upper = _median(rates)        # typical seed
    floor = x * (1.0 - rate_tolerance_fraction)

    # Section 12: the DECISIVE stability signal is the interactivity-deficit
    # queue Z_i. If the long-run per-user service rate meets the floor, Z_i
    # is rate-stable by construction. The server/device *price* queues are
    # expected to sit at a bounded non-zero level at any loaded operating
    # point (that non-zero price is what makes the drift-plus-penalty
    # controller trade off resources) -- a small positive tail slope on a
    # finite window that has not fully reached steady state does NOT mean
    # infeasible. We therefore gate feasibility on Z_i only, and keep the
    # resource-queue slopes as logged diagnostics.
    z_slope = max(float(m.max_z_slope) for m in measurements)
    dq_slope = max(float(m.max_device_queue_slope) for m in measurements)
    sq_slope = max(float(m.server_queue_slope) for m in measurements)
    deficit_queue_growing = z_slope > queue_slope_tolerance

    per_seed = [
        {"seed": m.seed, "min_rate_tps": m.min_rate_tps, "max_z_slope": m.max_z_slope}
        for m in measurements
    ]

    # If the lower estimate of the minimum sustained rate is comfortably
    # ABOVE the requirement (more than one tolerance band over x), the run
    # is feasible regardless of a small transient Z_i tail slope: Z_i
    # cannot diverge while per-user service exceeds per-user demand. The
    # queue-slope veto only applies in the marginal band, where a growing
    # Z_i is the tie-breaker toward "infeasible".
    comfortably_over = lower >= x * (1.0 + rate_tolerance_fraction)

    if comfortably_over:
        status = "feasible"
        reason = "min-rate lower bound exceeds x by more than the tolerance band"
    elif lower >= floor and not deficit_queue_growing:
        status = "feasible"
        reason = "lower rate estimate clears the floor; Z_i rate-stable"
    elif upper < floor or z_slope > 5 * queue_slope_tolerance:
        status = "infeasible"
        reason = (
            "upper rate estimate below floor"
            if upper < floor
            else "interactivity-deficit queue Z_i growing"
        )
    else:
        status, reason = "uncertain", "min_rate inside the tolerance band"

    return FeasibilityResult(
        status=status,
        min_rate_tps=mean_r,
        rate_margin_tps=lower - x,
        max_z_slope=z_slope,
        max_device_queue_slope=dq_slope,
        server_queue_slope=sq_slope,
        seeds_used=k,
        reason=reason,
        per_seed=per_seed,
    )


# --------------------------------------------------------------------------- 13
CANDIDATE_CACHE_SCHEMA_VERSION = 2  # bump when CandidateMeasurement's fixed-window
                                     # semantics change; see reject-on-mismatch below.


class CandidateCache:
    """`(x, scale, policy, seed, config_hash)` -> CandidateMeasurement on
    disk as JSON lines. The driver resumes without rerunning GPU points
    (Section 13.4). Section 19: versioned -- a record without a matching
    `schema_version` (e.g. an old cache predating the exact fixed-window
    denominator fix) is REJECTED at load time rather than silently reused,
    so a biased legacy rate can never leak into a final-mode result; the
    candidate is simply re-run.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._mem: dict[tuple, CandidateMeasurement] = {}
        self.rejected_legacy_records = 0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("schema_version") != CANDIDATE_CACHE_SCHEMA_VERSION:
                    self.rejected_legacy_records += 1
                    continue
                self._mem[self._key_from_rec(rec)] = CandidateMeasurement(**rec["measurement"])

    @staticmethod
    def _key(x: float, scale: int, policy: str, seed: int, config_hash: str) -> tuple:
        return (round(float(x), 6), int(scale), str(policy), int(seed), str(config_hash))

    @classmethod
    def _key_from_rec(cls, rec: dict) -> tuple:
        return cls._key(rec["x"], rec["scale"], rec["policy"], rec["seed"], rec["config_hash"])

    def get(self, x, scale, policy, seed, config_hash) -> CandidateMeasurement | None:
        return self._mem.get(self._key(x, scale, policy, seed, config_hash))

    def put(self, x, scale, policy, seed, config_hash, measurement: CandidateMeasurement) -> None:
        key = self._key(x, scale, policy, seed, config_hash)
        self._mem[key] = measurement
        rec = {
            "schema_version": CANDIDATE_CACHE_SCHEMA_VERSION,
            "x": float(x),
            "scale": int(scale),
            "policy": str(policy),
            "seed": int(seed),
            "config_hash": str(config_hash),
            "measurement": asdict(measurement),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")


def monotone_scale_search(
    feasibility_fn: Callable[[int], str],
    *,
    lo_feasible: int = 1,
    hi_cap: int,
    hi_hint: int | None = None,
) -> tuple[int, dict]:
    """Section 13.2: find the largest scale `m` that is feasible.

    `feasibility_fn(m)` returns "feasible" | "infeasible" | "uncertain".
    On "uncertain" the bracket is NOT moved — the caller is expected to
    have already refined (added seeds) so a persistent "uncertain" is
    treated as the boundary and reported as such.

    Returns `(m_star, trace)`. `m_star` is the last confirmed feasible
    scale. `trace["boundary"]` records the first infeasible scale seen.
    """
    if hi_cap < lo_feasible:
        raise ValueError("hi_cap must be >= lo_feasible")
    trace: dict = {"evaluated": [], "boundary_first_infeasible": None}

    def _ev(m: int) -> str:
        s = feasibility_fn(m)
        trace["evaluated"].append({"scale": m, "status": s})
        return s

    # exponential bracketing up from a known-feasible lo
    lo = lo_feasible
    if _ev(lo) != "feasible":
        # even the smallest candidate is not feasible
        trace["note"] = "lo_feasible was not feasible"
        return 0, trace

    # First probe: the monotonicity hint from the previous (smaller) x if
    # it is strictly above lo, else lo+1. The hint is ALWAYS verified --
    # never assumed feasible -- so hi_hint == hi_cap still gets tested.
    hi: int | None = None
    if hi_hint is not None and hi_hint > lo:
        probe = min(hi_cap, int(hi_hint))
    else:
        probe = min(hi_cap, lo + 1)

    while True:
        st = _ev(probe)
        if st == "feasible":
            lo = probe
            if probe >= hi_cap:
                return hi_cap, trace
            probe = min(hi_cap, max(probe + 1, probe * 2))
        else:
            if st == "infeasible":
                trace["boundary_first_infeasible"] = probe
            hi = probe
            break

    # binary search in (lo, hi]. An "uncertain" verdict (the caller has
    # already escalated seeds and still can't confirm) is treated as "not
    # confirmed feasible" -> move the ceiling down and keep narrowing,
    # rather than stopping and collapsing N* to the last clean feasible.
    # The trace records how many uncertain points were hit.
    trace["uncertain_scales"] = []
    while hi - lo > 1:
        mid = (lo + hi) // 2
        st = _ev(mid)
        if st == "feasible":
            lo = mid
        elif st == "infeasible":
            trace["boundary_first_infeasible"] = mid
            hi = mid
        else:  # uncertain
            trace["uncertain_scales"].append(mid)
            hi = mid
    return lo, trace


# --------------------------------------------------------------------------- 14
def step_area_bounds(x_grid: Sequence[float], nstar: Sequence[float]) -> dict:
    """Section 14: monotone Riemann bounds for the integer, nonincreasing
    N*(x) curve. `A_upper` uses the left (higher) value on each interval,
    `A_lower` the right (lower) value. `A_step` is their midpoint.
    """
    if len(x_grid) != len(nstar):
        raise ValueError("x_grid and nstar must have equal length")
    if len(x_grid) < 2:
        return {"area_lower": 0.0, "area_upper": 0.0, "area_step_estimate": 0.0}
    xs = [float(x) for x in x_grid]
    ns = [float(n) for n in nstar]
    if any(b < a for a, b in zip(xs, xs[1:])):
        raise ValueError("x_grid must be non-decreasing")
    lower = sum(ns[k + 1] * (xs[k + 1] - xs[k]) for k in range(len(xs) - 1))
    upper = sum(ns[k] * (xs[k + 1] - xs[k]) for k in range(len(xs) - 1))
    return {
        "area_lower": lower,
        "area_upper": upper,
        "area_step_estimate": 0.5 * (lower + upper),
    }


def ica(
    x_grid: Sequence[float],
    nstar: Sequence[float],
    *,
    x_L: float,
    x_U: float,
    n_ref: float,
) -> float:
    """Section 14 / 0: normalized interactivity-concurrency area in [0,1].

    ICA = (1 / (N_ref * (x_U - x_L))) * integral_{x_L}^{x_U} min(N*(x), N_ref) dx

    The integral uses the same monotone step-envelope (left value on each
    sub-interval) as `step_area_bounds`' upper bound, clipped to the
    [x_L, x_U] window and capped at N_ref. `x_L` must be > 0 (or a finite
    N_ref cap applied) — enforced by the caller passing a positive x_L.
    """
    if x_U <= x_L:
        raise ValueError("x_U must exceed x_L")
    if n_ref <= 0:
        raise ValueError("n_ref must be positive")
    if x_L <= 0:
        raise ValueError("x_L must be strictly positive (Section 0)")
    pts = sorted(zip((float(x) for x in x_grid), (float(n) for n in nstar)))
    xs = [p[0] for p in pts]
    ns = [min(float(n_ref), p[1]) for p in pts]

    def n_at(x: float) -> float:
        # step function: value carried forward from the last grid point <= x
        if x <= xs[0]:
            return ns[0]
        for k in range(len(xs) - 1):
            if xs[k] <= x < xs[k + 1]:
                return ns[k]
        return ns[-1]

    # integrate over [x_L, x_U] on the union of grid points and window edges
    knots = sorted({x_L, x_U, *[x for x in xs if x_L < x < x_U]})
    area = 0.0
    for a, b in zip(knots, knots[1:]):
        area += n_at(a) * (b - a)
    return area / (n_ref * (x_U - x_L))


def bootstrap_ica(
    seed_indexed_nstar: dict[int, Sequence[float]],
    x_grid: Sequence[float],
    *,
    x_L: float,
    x_U: float,
    n_ref: float,
    n_boot: int = 1000,
    rng_seed: int = 0,
) -> dict:
    """Section 14: resample the SEED axis, rebuild N*(x) per resample
    (element-wise min across the drawn seeds, the conservative
    reconstruction), integrate ICA, and report the 95% CI. Callers that
    have per-seed feasibility should pass per-seed N*(x) vectors here.
    """
    import random as _random

    seeds = list(seed_indexed_nstar.keys())
    if not seeds:
        raise ValueError("need at least one seed vector")
    rng = _random.Random(rng_seed)
    vals: list[float] = []
    for _ in range(n_boot):
        draw = [rng.choice(seeds) for _ in seeds]
        stacked = [seed_indexed_nstar[s] for s in draw]
        nstar = [min(vec[i] for vec in stacked) for i in range(len(x_grid))]
        vals.append(ica(x_grid, nstar, x_L=x_L, x_U=x_U, n_ref=n_ref))
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return {"ica_mean": sum(vals) / len(vals), "bootstrap_ci_95": [lo, hi]}


# ===========================================================================
# CAPACITY_AWARE_FRAMEWORK_CODEX_IMPLEMENTATION.md "final mode" additions.
#
# The functions above (classify_feasibility, monotone_scale_search,
# step_area_bounds/ica) are kept unchanged and are now documented as
# PILOT/SANITY-ONLY (Section 9.2: "A one-seed mode may remain for --pilot,
# but it must be labelled heuristic/sanity-only and must not produce
# headline ICA."). Everything below is the exact-SLO, statistically
# interpretable final-mode path (Sections 9-11).
# ===========================================================================


def _bonferroni_bootstrap_ci(
    values: Sequence[float],
    confidence_delta: float,
    n_clients: int,
    *,
    n_boot: int = 2000,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """Section 9.1's documented bootstrap alternative to the Bonferroni
    Student-t formula (no scipy dependency). Resamples `values` (one
    client's rate across K independent seeds) with replacement, and reads
    off the (alpha/2, 1-alpha/2) percentiles where
    `alpha = confidence_delta / n_clients` is the per-client Bonferroni
    budget for SIMULTANEOUS coverage across `n_clients` clients.

    K < 2: no interval can be formed; caller must treat the result as
    heuristic-only, never as a certified bound (Section 9.2).
    """
    import random as _random

    k = len(values)
    if k < 2:
        v = float(values[0]) if values else 0.0
        return v, v
    rng = _random.Random(rng_seed)
    boots = []
    for _ in range(n_boot):
        draw = [values[rng.randrange(k)] for _ in range(k)]
        boots.append(sum(draw) / k)
    boots.sort()
    alpha = confidence_delta / max(1, n_clients)
    lo_idx = max(0, min(n_boot - 1, int((alpha / 2.0) * (n_boot - 1))))
    hi_idx = max(0, min(n_boot - 1, int((1.0 - alpha / 2.0) * (n_boot - 1))))
    return boots[lo_idx], boots[hi_idx]


@dataclass
class FinalFeasibilityResult:
    """Section 9.1: the exact-SLO classifier. `status` is one of
    "feasible" | "infeasible" | "uncertain" | "heuristic_only" (K=1: no
    statistical certification is possible, Section 9.2's test
    requirement)."""

    status: str
    per_client_lcb: dict[int, float]
    per_client_ucb: dict[int, float]
    min_lcb: float          # min_i L_i -- feasible iff this is >= x
    min_ucb: float          # min_i U_i -- infeasible iff this is <  x
    seeds_used: int
    confidence_delta: float
    reason: str


def classify_feasibility_final(
    measurements: Sequence["CandidateMeasurement"],
    x: float,
    n_clients: int,
    *,
    confidence_delta: float = 0.05,
    n_boot: int = 2000,
) -> FinalFeasibilityResult:
    """Section 9.1, verbatim:

        FEASIBLE    iff min_i L_i >= x
        INFEASIBLE  iff min_i U_i <  x
        UNCERTAIN   otherwise

    `x` is the EXACT requirement -- no `(1-eps)*x` floor anywhere in this
    function (Section 0 rule 7). Queue-tail slopes are NOT read here; they
    are secondary diagnostics logged separately (Section 9.3) and must
    never veto a rate-based verdict.
    """
    k = len(measurements)
    if k < 2:
        return FinalFeasibilityResult(
            status="heuristic_only",
            per_client_lcb={}, per_client_ucb={},
            min_lcb=float("nan"), min_ucb=float("nan"),
            seeds_used=k, confidence_delta=confidence_delta,
            reason="K=1: cannot form a statistical confidence bound (Section 9.2)",
        )
    per_client_lcb: dict[int, float] = {}
    per_client_ucb: dict[int, float] = {}
    for cid in range(n_clients):
        vals = [float(m.per_client_rates.get(cid, 0.0)) for m in measurements]
        lo, hi = _bonferroni_bootstrap_ci(vals, confidence_delta, n_clients, n_boot=n_boot)
        per_client_lcb[cid] = lo
        per_client_ucb[cid] = hi
    min_lcb = min(per_client_lcb.values()) if per_client_lcb else 0.0
    min_ucb = min(per_client_ucb.values()) if per_client_ucb else 0.0

    if min_lcb >= x:
        status, reason = "feasible", "min_i L_i >= x"
    elif min_ucb < x:
        status, reason = "infeasible", "min_i U_i < x"
    else:
        status, reason = "uncertain", "confidence bounds straddle x"

    return FinalFeasibilityResult(
        status=status, per_client_lcb=per_client_lcb, per_client_ucb=per_client_ucb,
        min_lcb=min_lcb, min_ucb=min_ucb, seeds_used=k,
        confidence_delta=confidence_delta, reason=reason,
    )


@dataclass
class SearchDecision:
    n: int
    status: str
    min_lcb: float
    min_ucb: float
    seeds_used: int


@dataclass
class NStarResult:
    """Section 10.1. A feasible candidate exactly at the search cap is a
    LOWER bound, never reported as `N* = N_cap` (Section 0 rule 8)."""

    x_requirement: float
    last_confirmed_feasible_n: int
    first_confirmed_infeasible_n: int | None
    lower_bound_n: int
    upper_bound_n: int | None
    exact: bool
    hit_search_cap: bool
    uncertain_n: list[int] = field(default_factory=list)
    trace: list[SearchDecision] = field(default_factory=list)


def monotone_scale_search_final(
    feasibility_fn: Callable[[int], SearchDecision],
    *,
    lo_feasible: int = 1,
    hi_cap: int,
    hi_hint: int | None = None,
    x_requirement: float = 0.0,
) -> NStarResult:
    """Section 10.1 steps 1-4 (local re-certification, step 5, is the
    caller's job -- see Gate B in the spec, which re-runs {N*-1,N*,N*+1}
    on real GPU data). `feasibility_fn(n)` returns a `SearchDecision` with
    status in {feasible, infeasible, uncertain}; "uncertain" is expected
    to already reflect the caller's seed-escalation policy (Section 9.2's
    `min_final_seeds` near the boundary), so this function does NOT retry
    -- it narrows the bracket through uncertain points and reports them.
    """
    if hi_cap < lo_feasible:
        raise ValueError("hi_cap must be >= lo_feasible")
    trace: list[SearchDecision] = []
    uncertain_n: list[int] = []

    def _ev(n: int) -> SearchDecision:
        d = feasibility_fn(n)
        trace.append(d)
        if d.status == "uncertain":
            uncertain_n.append(n)
        return d

    lo_decision = _ev(lo_feasible)
    if lo_decision.status != "feasible":
        return NStarResult(
            x_requirement=x_requirement,
            last_confirmed_feasible_n=0,
            first_confirmed_infeasible_n=lo_feasible if lo_decision.status == "infeasible" else None,
            lower_bound_n=0, upper_bound_n=lo_feasible if lo_decision.status == "infeasible" else None,
            exact=lo_decision.status == "infeasible",
            hit_search_cap=False, uncertain_n=uncertain_n, trace=trace,
        )

    lo = lo_feasible
    probe = min(hi_cap, int(hi_hint)) if (hi_hint is not None and hi_hint > lo) else min(hi_cap, lo + 1)
    hi: int | None = None
    while True:
        d = _ev(probe)
        if d.status == "feasible":
            lo = probe
            if probe >= hi_cap:
                # Section 0 rule 8: cap-feasible is a lower bound, not N*.
                return NStarResult(
                    x_requirement=x_requirement,
                    last_confirmed_feasible_n=lo, first_confirmed_infeasible_n=None,
                    lower_bound_n=hi_cap, upper_bound_n=None,
                    exact=False, hit_search_cap=True,
                    uncertain_n=uncertain_n, trace=trace,
                )
            probe = min(hi_cap, max(probe + 1, probe * 2))
        else:
            hi = probe
            break

    # binary search in (lo, hi]
    first_infeasible = hi if trace[-1].status == "infeasible" else None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        d = _ev(mid)
        if d.status == "feasible":
            lo = mid
        elif d.status == "infeasible":
            first_infeasible = mid
            hi = mid
        else:
            # uncertain: narrow the ceiling but do not certify past it
            hi = mid

    exact = first_infeasible == lo + 1
    return NStarResult(
        x_requirement=x_requirement,
        last_confirmed_feasible_n=lo,
        first_confirmed_infeasible_n=first_infeasible,
        lower_bound_n=lo,
        upper_bound_n=first_infeasible if first_infeasible is not None else hi,
        exact=exact,
        hit_search_cap=False,
        uncertain_n=uncertain_n,
        trace=trace,
    )


@dataclass
class XStarResult:
    """Section 10.2: the dual search axis at fixed N."""

    n_active: int
    lower_feasible_x: float
    upper_infeasible_x: float | None
    estimate_x: float
    exact_within_tolerance: bool
    trace: list[SearchDecision] = field(default_factory=list)


def xstar_search(
    feasibility_fn: Callable[[float], SearchDecision],
    *,
    n_active: int,
    x_lo: float,
    x_hi: float,
    x_tolerance_tps: float = 0.10,
) -> XStarResult:
    """Bisect `[x_lo, x_hi]` at fixed `n_active` until the bracket width is
    <= `x_tolerance_tps`. Requires `feasibility_fn(x_lo)` feasible and
    `feasibility_fn(x_hi)` infeasible (caller should widen the bracket with
    exponential search first if that does not hold)."""
    trace: list[SearchDecision] = []

    def _ev(x: float) -> SearchDecision:
        d = feasibility_fn(x)
        trace.append(d)
        return d

    lo_d = _ev(x_lo)
    hi_d = _ev(x_hi)
    if lo_d.status != "feasible":
        return XStarResult(n_active=n_active, lower_feasible_x=x_lo, upper_infeasible_x=x_lo,
                            estimate_x=x_lo, exact_within_tolerance=False, trace=trace)
    if hi_d.status == "feasible":
        return XStarResult(n_active=n_active, lower_feasible_x=x_hi, upper_infeasible_x=None,
                            estimate_x=x_hi, exact_within_tolerance=False, trace=trace)

    lo, hi = x_lo, x_hi
    while hi - lo > x_tolerance_tps:
        mid = 0.5 * (lo + hi)
        d = _ev(mid)
        if d.status == "feasible":
            lo = mid
        elif d.status == "infeasible":
            hi = mid
        else:
            # uncertain: stop narrowing, report the current bracket
            break

    return XStarResult(
        n_active=n_active, lower_feasible_x=lo, upper_infeasible_x=hi,
        estimate_x=0.5 * (lo + hi), exact_within_tolerance=(hi - lo) <= x_tolerance_tps,
        trace=trace,
    )


def area_from_nstar(
    results: Sequence[NStarResult],
    *,
    x_L: float,
    x_U: float,
    n_ref: float,
) -> dict:
    """Section 11: dual-bound capacity area from a sequence of `NStarResult`
    (one per swept x, sorted by x). NO extrapolation outside `[x_L, x_U]`
    -- the swept grid must already cover that range (raises otherwise).
    Cap-limited or interval-valued points widen `area_upper` using
    `n_ref` as the unresolved upper bound and `area_lower` using the last
    CONFIRMED feasible count.
    """
    pts = sorted(results, key=lambda r: r.x_requirement)
    if not pts:
        raise ValueError("need at least one NStarResult")
    xs = [r.x_requirement for r in pts]
    if xs[0] > x_L or xs[-1] < x_U:
        raise ValueError(
            f"swept grid [{xs[0]}, {xs[-1]}] does not cover [{x_L}, {x_U}] -- "
            "Section 11: do not extrapolate the ICA window beyond the measured grid"
        )
    lower_n = [min(float(n_ref), float(r.lower_bound_n)) for r in pts]
    upper_n = [
        min(float(n_ref), float(r.lower_bound_n))
        if r.exact
        else min(float(n_ref), float(r.upper_bound_n) if r.upper_bound_n is not None else float(n_ref))
        for r in pts
    ]
    # restrict to [x_L, x_U] and integrate the step function (carry the
    # value at the last grid point <= the evaluation point)
    knots = sorted({x_L, x_U, *[x for x in xs if x_L < x < x_U]})

    def _step_area(ns: list[float]) -> float:
        def n_at(x: float) -> float:
            if x <= xs[0]:
                return ns[0]
            for k in range(len(xs) - 1):
                if xs[k] <= x < xs[k + 1]:
                    return ns[k]
            return ns[-1]
        return sum(n_at(a) * (b - a) for a, b in zip(knots, knots[1:]))

    area_lower = _step_area(lower_n)
    area_upper = _step_area(upper_n)
    width = n_ref * (x_U - x_L)
    return {
        "area_lower": area_lower, "area_upper": area_upper,
        "ica_lower": area_lower / width, "ica_upper": area_upper / width,
        "any_cap_limited": any(r.hit_search_cap for r in pts),
        "any_interval_valued": any((not r.exact) and (not r.hit_search_cap) for r in pts),
    }


def area_from_xstar(
    results: Sequence[XStarResult],
    *,
    x_L: float,
    x_U: float,
    n_ref: int,
) -> dict:
    """Section 11's dual column-sum check:
    `A_C = sum_{N=1}^{N_ref} [min(x*(N), x_U) - x_L]^+`, using the ESTIMATE
    (midpoint of the certified bracket) for each `N`. Integer sum over
    concurrency -- never trapezoidal interpolation over fractional N."""
    by_n = {r.n_active: r for r in results}
    total_lower = 0.0
    total_upper = 0.0
    for n in range(1, int(n_ref) + 1):
        r = by_n.get(n)
        if r is None:
            continue
        x_est = r.estimate_x
        x_hi_bound = r.upper_infeasible_x if r.upper_infeasible_x is not None else x_U
        total_lower += max(0.0, min(r.lower_feasible_x, x_U) - x_L)
        total_upper += max(0.0, min(x_hi_bound, x_U) - x_L)
        del x_est
    width = n_ref * (x_U - x_L)
    return {
        "area_lower": total_lower, "area_upper": total_upper,
        "ica_lower": total_lower / width, "ica_upper": total_upper / width,
    }
