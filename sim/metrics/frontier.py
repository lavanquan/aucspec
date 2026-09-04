"""Measure C(xi) empirically by sweeping admitted concurrency / controller V.

Records (Y, min_i x_i) at each operating point, per Definition 2, eq. (5).
TASKS.md T1.6.

Phase 1 only provides the plumbing to turn a list of measured operating
points into a frontier (sorted, deduplicated by x, upper envelope). Sweeping
a live controller's V (the general case) is Phase 2's job, once
sim/controllers/auc_controller.py exists; sweeping gamma in the symmetric
closed-form instance (eq. 10-11, EXPERIMENTS.md Exp 2a) can already be done
with sim/core/device.py's symmetric_operating_point and does not need a
controller.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatingPoint:
    """One measured (min_i x_i, Y) point of the achievable region C(xi)."""

    min_interactivity: float
    goodput: float
    label: str | None = None


def build_frontier(points: list[OperatingPoint]) -> list[OperatingPoint]:
    """Reduce measured operating points to an empirical estimate of Y*(x;xi),
    eq. (5): for each distinct x, keep only the best (max) Y observed, sorted
    by x ascending.

    Proposition 2(ii) says the true Y*(.;xi) is concave and nonincreasing;
    this function does not enforce that shape, since a single baseline point
    is not expected to lie on the curve at all -- only a dense sweep from the
    AUC-controller should approximate it. A concave-hull filter for that case
    is not implemented here; add one if/when Exp 1 needs it.
    """
    best_goodput_by_x: dict[float, OperatingPoint] = {}
    for point in points:
        if point.goodput < 0.0 or point.min_interactivity < 0.0:
            raise ValueError("goodput and min_interactivity must be non-negative")
        existing = best_goodput_by_x.get(point.min_interactivity)
        if existing is None or point.goodput > existing.goodput:
            best_goodput_by_x[point.min_interactivity] = point
    return sorted(best_goodput_by_x.values(), key=lambda p: p.min_interactivity)


def sweep_symmetric_gamma(
    batch_size: int,
    alpha: float,
    gamma_choices: list[int],
    tau_d_seconds: float,
    kappa_bits_per_token: float,
    uplink_rate_bps: float,
    delta_seconds: float,
    roofline,
) -> list[OperatingPoint]:
    """Closed-form frontier of the symmetric saturated instance (eq. 10):
    sweep gamma at a fixed batch size B, per EXPERIMENTS.md Exp 2a.
    """
    from ..core.device import symmetric_operating_point

    points = []
    for gamma in gamma_choices:
        x, y = symmetric_operating_point(
            batch_size,
            gamma,
            alpha,
            tau_d_seconds,
            kappa_bits_per_token,
            uplink_rate_bps,
            delta_seconds,
            roofline,
        )
        points.append(OperatingPoint(min_interactivity=x, goodput=y, label=f"gamma={gamma}"))
    return points
