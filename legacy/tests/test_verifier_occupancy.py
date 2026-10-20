"""C1.4: exact verifier occupancy = sum of service-interval overlaps with the measurement window / window length."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "c1" / "analyze_rq1_pilot.py"
spec = importlib.util.spec_from_file_location("an_occ", SCRIPT)
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)

START, END = 30000.0, 150000.0          # 120 s window


def frame(intervals, members=2):
    rows = []
    for b, (s, f) in enumerate(intervals):
        for m in range(members):        # several requests share one batch: it must be counted ONCE
            rows.append({"verification_batch_id": b, "batch_start_ms": s, "verify_finish_ms": f})
    return pd.DataFrame(rows)


def occ(intervals):
    return an.exact_occupancy(frame(intervals), START, END)


def test_fully_inside_and_idle_periods():
    r = occ([(40000.0, 41000.0), (60000.0, 60500.0)])
    assert r["utilisation"] == pytest.approx(1500.0 / 120000.0) and r["n_batches_overlapping_window"] == 2 and r["serialisation_ok"]


def test_batch_crossing_the_window_start_is_clipped():
    r = occ([(29000.0, 31000.0)])
    assert r["busy_ms"] == pytest.approx(1000.0) and r["utilisation"] == pytest.approx(1000.0 / 120000.0)


def test_batch_crossing_the_window_end_is_clipped():
    r = occ([(149000.0, 152000.0)])
    assert r["busy_ms"] == pytest.approx(1000.0)


def test_batches_outside_the_window_do_not_count():
    r = occ([(1000.0, 5000.0), (150000.0, 153000.0), (151000.0 + 2000.0, 160000.0)])
    assert r["busy_ms"] == 0.0 and r["n_batches_overlapping_window"] == 0


def test_back_to_back_batches_give_full_occupancy():
    r = occ([(20000.0, 60000.0), (60000.0, 100000.0), (100000.0, 160000.0)])
    assert r["utilisation"] == pytest.approx(1.0) and r["serialisation_ok"]


def test_a_batch_that_straddles_the_window_edge_is_counted_even_if_no_client_receives_inside():
    # service [149000, 152000]; its clients receive after 150000 (outside the window) -> the legacy accounting would miss it
    r = occ([(100000.0, 101000.0), (149000.0, 152000.0)])
    assert r["n_batches_overlapping_window"] == 2 and r["busy_ms"] == pytest.approx(2000.0)


def test_overlapping_service_intervals_are_flagged_not_silently_summed():
    r = occ([(40000.0, 50000.0), (45000.0, 55000.0)])
    assert r["serialisation_ok"] is False


def test_legacy_and_corrected_utilisation_differ_when_batches_straddle_or_wall_exceeds_the_window_overlap():
    from test_rq1_pilot_analysis import synthetic_run
    df = synthetic_run(4, 3.0)
    df["verify_finish_ms"] = df["batch_start_ms"] + df["verification_batch_wall_ms"]
    m = an.run_metrics(df, 4)
    assert "verifier_utilisation_legacy" in m and m["verifier_batches_overlapping_window"] > 0
    assert 0.0 <= m["verifier_utilisation"] <= 1.0 + 1e-9
