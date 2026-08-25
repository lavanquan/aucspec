from __future__ import annotations

import math

from edge_specsim.metrics import trapezoidal_auc, upper_concave_hull


def test_trapezoidal_auc_matches_expected_sum() -> None:
    xs = [0.0, 1.0, 2.0, 4.0]
    ys = [10.0, 8.0, 6.0, 2.0]

    expected = (
        0.5 * (10.0 + 8.0) * (1.0 - 0.0)
        + 0.5 * (8.0 + 6.0) * (2.0 - 1.0)
        + 0.5 * (6.0 + 2.0) * (4.0 - 2.0)
    )

    assert math.isclose(trapezoidal_auc(xs, ys), expected)


def test_trapezoidal_auc_is_zero_with_fewer_than_two_points() -> None:
    assert trapezoidal_auc([2.0], [5.0]) == 0.0


def test_upper_concave_hull_removes_points_dominated_by_time_sharing() -> None:
    xs = [0.0, 1.0, 2.0]
    ys = [10.0, 8.0, 9.0]

    # The middle point lies below the chord from (0,10) to (2,9),
    # so time-sharing dominates it and the hull should skip it.
    assert upper_concave_hull(xs, ys) == [0, 2]


def test_upper_concave_hull_keeps_concave_points() -> None:
    xs = [0.0, 1.0, 2.0]
    ys = [4.0, 6.0, 7.0]

    assert upper_concave_hull(xs, ys) == [0, 1, 2]
