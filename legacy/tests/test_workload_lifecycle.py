"""Adversarial tests for the workload-lifecycle exhaustion criterion (audit items M1/M2)."""
from pathlib import Path

import pandas as pd
import pytest

from edge_specsim.capacity import (
    CandidateMeasurement,
    classify_feasibility,
    classify_feasibility_final,
)
from edge_specsim.workload_lifecycle import (
    ACTIVE,
    EXHAUSTED,
    NO_WORK,
    UNKNOWN,
    assess_from_dataframe,
    assess_workload,
)

START, END = 5000.0, 25000.0  # ms, the B6 smoke window


def rnd(draft_start, receive, tokens=3, complete=False, assigned=2, pending=0, finished=None):
    return {
        "draft_start_ms": draft_start,
        "client_receive_ms": receive,
        "committed_token_count": tokens,
        "client_samples_assigned": assigned,
        "sample_finished": complete if finished is None else finished,
        "client_samples_pending": pending,
        "client_work_complete": complete,
    }


def status_of(rows, **kw):
    a = assess_workload({0: rows}, admitted_client_ids=[0], measurement_start_ms=START, measurement_end_ms=END, **kw)
    return a, a.clients[0]


def legacy_span_guard(win: pd.DataFrame, meas_s: float) -> list[int]:
    """The heuristic removed by C1 (kept only to show before/after)."""
    out = []
    for cid, g in win.groupby("client_id"):
        span_s = (g["client_receive_ms"].max() - g["client_receive_ms"].min()) / 1000.0
        if span_s < 0.85 * meas_s:
            out.append(int(cid))
    return out


def test_sparse_but_ongoing_rounds_are_not_exhaustion():
    # one round every 9 s: only 3 in-window rounds, span 18 s < 0.85*20 s, yet work remains for a long time
    rows = [rnd(t, t + 8000.0, complete=False, pending=3) for t in range(0, 90000, 9000)]
    a, c = status_of(rows)
    assert c.status == ACTIVE and not a.invalid
    win = pd.DataFrame([dict(r, client_id=0) for r in rows if START <= r["client_receive_ms"] <= END])
    assert legacy_span_guard(win, 20.0) == [0], "the old span heuristic mis-fires on this pattern"


def test_genuine_early_exhaustion():
    rows = [rnd(t, t + 1800.0, complete=False, pending=1) for t in range(0, 9000, 1800)]
    rows.append(rnd(9000.0, 10800.0, complete=True))  # all work done at 10.8 s < 25 s
    a, c = status_of(rows)
    assert c.status == EXHAUSTED and a.invalid and a.exhausted_clients == [0]
    assert c.completion_ms == 10800.0


@pytest.mark.parametrize(
    "completion,expected",
    [(END - 1.0, EXHAUSTED), (END, ACTIVE), (END + 1.0, ACTIVE)],
)
def test_completion_just_before_at_and_after_the_boundary(completion, expected):
    rows = [rnd(1000.0, 3000.0, complete=False, pending=1), rnd(completion - 1800.0, completion, complete=True)]
    _, c = status_of(rows)
    assert c.status == expected


def test_in_flight_request_at_measurement_end_keeps_client_active():
    # final round straddles the end: drafted at 24.0 s, delivered at 26.5 s; it also completes the workload
    rows = [rnd(1000.0, 3000.0, complete=False, pending=1), rnd(24000.0, 26500.0, complete=True)]
    a, c = status_of(rows)
    assert c.status == ACTIVE and c.in_flight_at_end and not a.invalid


def test_client_with_zero_in_window_tokens_but_pending_work_is_active_not_invalid():
    # one very long round spans the whole window; zero in-window receives
    rows = [rnd(1000.0, 3000.0, complete=False, pending=1), rnd(3000.0, 40000.0, complete=False, pending=1)]
    a, c = status_of(rows)
    assert c.in_window_tokens == 0 and c.in_window_rounds == 0
    assert c.status == ACTIVE and c.in_flight_at_end and not a.invalid


def test_client_without_any_rows_is_not_silently_dropped():
    a = assess_workload({}, admitted_client_ids=[0, 1], measurement_start_ms=START, measurement_end_ms=END)
    assert [c.status for c in a.clients] == [UNKNOWN, UNKNOWN]
    assert not a.verifiable and not a.invalid
    a2 = assess_workload({}, admitted_client_ids=[0], measurement_start_ms=START, measurement_end_ms=END,
                         expected_samples_per_client=0)
    assert a2.clients[0].status == NO_WORK and a2.invalid


def test_assigned_zero_samples_is_invalid():
    a, c = status_of([rnd(0.0, 1000.0, assigned=0)])
    assert c.status == NO_WORK and a.invalid


def test_work_completed_before_warmup_end_is_exhausted_even_with_zero_window_rows():
    rows = [rnd(0.0, 2000.0, complete=True)]
    a, c = status_of(rows)
    assert c.status == EXHAUSTED and c.in_window_rounds == 0 and a.invalid


def test_no_measurement_window_means_no_assessment():
    a = assess_workload({0: [rnd(0.0, 1.0)]}, admitted_client_ids=[0], measurement_start_ms=0.0, measurement_end_ms=None)
    assert a.clients == [] and not a.invalid


def test_b6_smoke_false_positive_pattern():
    """The recorded B6 smoke (N=8, window 5-25 s, 6 questions per client, clients running 107-128 s)."""
    df = pd.read_csv(Path(__file__).parent / "data_b6_smoke_compact.csv")
    win = df[df["in_measurement_window"].astype(bool)]
    # before: 5 of 8 clients flagged by the span heuristic (the INVALID verdict recorded in the B6 report)
    assert len(legacy_span_guard(win, 20.0)) >= 5
    # after: nobody is exhausted -- every client still had work at the end of the window
    a = assess_from_dataframe(df, admitted_client_ids=range(8), measurement_start_ms=START,
                              measurement_end_ms=END, expected_samples_per_client=6)
    assert not a.invalid and a.verifiable
    assert all(c.status == ACTIVE for c in a.clients)
    assert all(c.last_event_ms > 100000.0 for c in a.clients), "clients ran >100 s virtual time"


def test_legacy_rows_without_columns_and_without_expectation_are_unknown_not_exhausted():
    df = pd.read_csv(Path(__file__).parent / "data_b6_smoke_compact.csv")
    a = assess_from_dataframe(df, admitted_client_ids=range(8), measurement_start_ms=START, measurement_end_ms=END)
    assert not a.verifiable and not a.invalid


def test_inferred_exhaustion_on_legacy_rows():
    """A legacy log whose clients finish all expected samples before the window end is flagged as inferred."""
    df = pd.read_csv(Path(__file__).parent / "data_b6_smoke_compact.csv")
    a = assess_from_dataframe(df, admitted_client_ids=range(8), measurement_start_ms=START,
                              measurement_end_ms=130000.0, expected_samples_per_client=6)
    assert a.invalid and all(c.basis == "inferred_from_rows" for c in a.clients)


# ---- INVALID propagation through the measurement/classification layer -------------------------------
def meas(rates, invalid=False):
    return CandidateMeasurement(scale=2, n_active=2, x_requirement=1.0, seed=1, min_rate_tps=min(rates.values()),
                                per_client_rates=rates, invalid=invalid)


def test_invalid_is_not_infeasible_in_any_classifier():
    bad = meas({0: 0.1, 1: 0.1}, invalid=True)  # would look INFEASIBLE if classified
    good = meas({0: 5.0, 1: 5.0})
    assert classify_feasibility([bad, good, good], 1.0).status == "invalid"
    assert classify_feasibility_final([bad, good, good], 1.0, 2).status == "invalid"
    # a single invalid seed poisons the candidate: it is excluded from K by the caller, never averaged in
    assert classify_feasibility_final([good, good, good], 1.0, 2, n_boot=200).status != "invalid"
