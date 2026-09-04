"""Trapezoidal integration of the measured frontier: AUC(xi) = integral Y*(x;xi) dx.

See docs/notation.md eq. (6). Headline metric used in every cross-method
summary table (CLAUDE.md section 4). TASKS.md T1.7.
"""

from __future__ import annotations

from .frontier import OperatingPoint


def trapezoidal_auc(points: list[OperatingPoint]) -> float:
    """AUC(xi) = integral_0^{x_bar(xi)} Y*(x;xi) dx, eq. (6), via the
    trapezoidal rule over frontier points sorted by x ascending.

    Requires at least 2 points (otherwise there is no interval to integrate
    over, and the area is defined to be 0). Points must be sorted by
    ascending min_interactivity -- use frontier.build_frontier first.
    """
    if len(points) < 2:
        return 0.0
    auc = 0.0
    for left, right in zip(points[:-1], points[1:]):
        if right.min_interactivity < left.min_interactivity:
            raise ValueError("points must be sorted by ascending min_interactivity")
        width = right.min_interactivity - left.min_interactivity
        auc += 0.5 * (left.goodput + right.goodput) * width
    return auc


def normalized_auc(points: list[OperatingPoint]) -> float:
    """AUC-hat(xi) = AUC(xi) / (x_bar(xi) * Y*(0;xi)), eq. (6)'s
    normalization, in (0, 1]. Unit-free, enables cross-system comparison
    (CLAUDE.md section 2). Points must be sorted by ascending
    min_interactivity, with the first point at x = 0 (i.e. the unconstrained
    maximum-goodput point Y*(0;xi)).
    """
    if len(points) < 2:
        return 0.0
    if points[0].min_interactivity != 0.0:
        raise ValueError("normalized_auc requires the first point at x = 0 (Y*(0;xi))")
    x_bar = points[-1].min_interactivity
    y_star_0 = points[0].goodput
    if x_bar <= 0.0 or y_star_0 <= 0.0:
        return 0.0
    return trapezoidal_auc(points) / (x_bar * y_star_0)
