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


def classify_feasibility(
    measurements: Sequence[CandidateMeasurement],
    x: float,
    *,
    rate_tolerance_fraction: float = 0.03,
    queue_slope_tolerance: float = 1e-3,
) -> FeasibilityResult:
    """Section 12.2. Conservative: uses the across-seed mean minus one
    standard error as the lower estimate and mean plus one standard error
    as the upper estimate of min_rate. A point strictly inside the
    tolerance band -> "uncertain" (caller adds seeds).
    """
    if not measurements:
        raise ValueError("need at least one measurement")
    rates = [float(m.min_rate_tps) for m in measurements]
    k = len(rates)
    mean_r = sum(rates) / k
    if k >= 2:
        var = sum((r - mean_r) ** 2 for r in rates) / (k - 1)
        se = math.sqrt(var / k)
    else:
        se = 0.0
    lower = mean_r - se
    upper = mean_r + se
    floor = x * (1.0 - rate_tolerance_fraction)

    z_slope = max(float(m.max_z_slope) for m in measurements)
    dq_slope = max(float(m.max_device_queue_slope) for m in measurements)
    sq_slope = max(float(m.server_queue_slope) for m in measurements)
    queues_growing = (
        z_slope > queue_slope_tolerance
        or dq_slope > queue_slope_tolerance
        or sq_slope > queue_slope_tolerance
    )

    per_seed = [
        {"seed": m.seed, "min_rate_tps": m.min_rate_tps, "max_z_slope": m.max_z_slope}
        for m in measurements
    ]

    if lower >= floor and not queues_growing:
        status, reason = "feasible", "lower rate estimate clears the floor; no queue drift"
    elif upper < floor or z_slope > 5 * queue_slope_tolerance:
        status = "infeasible"
        reason = (
            "upper rate estimate below floor"
            if upper < floor
            else "interactivity-deficit queue growing"
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
class CandidateCache:
    """`(x, scale, policy, seed, config_hash)` -> CandidateMeasurement on
    disk as JSON lines. The driver resumes without rerunning GPU points
    (Section 13.4)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._mem: dict[tuple, CandidateMeasurement] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
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

    hi = hi_hint if hi_hint is not None else lo
    step = max(1, lo)
    while hi < hi_cap:
        nxt = min(hi_cap, max(hi + 1, hi * 2 if hi > 0 else step))
        st = _ev(nxt)
        if st == "feasible":
            lo = nxt
            hi = nxt
            if nxt >= hi_cap:
                return hi_cap, trace
        else:
            if st == "infeasible":
                trace["boundary_first_infeasible"] = nxt
            hi = nxt
            break
    else:
        # never found an infeasible point up to the cap
        return min(lo, hi_cap), trace

    # binary search in (lo, hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        st = _ev(mid)
        if st == "feasible":
            lo = mid
        elif st == "infeasible":
            trace["boundary_first_infeasible"] = mid
            hi = mid
        else:  # uncertain -> treat as the boundary, stop narrowing
            trace["note"] = f"uncertain at scale {mid}; reporting {lo} as N*"
            break
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
