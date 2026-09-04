import pytest

from sim.workloads.synthetic_trace import (
    generate_drifting_trace,
    generate_stationary_trace,
    write_trace_jsonl,
)
from sim.workloads.trace_loader import (
    TraceRecord,
    build_trace_acceptance,
    domain_alpha_summary,
    group_by_stream,
    lag1_autocorrelation,
    load_trace_jsonl,
    round_level_alpha_series,
    token_accept_flags,
    token_level_alpha_series,
)


def test_token_accept_flags_matches_censoring_convention():
    # accepted=3, all_accepted=True -> 3 successes, no censored failure.
    records = [TraceRecord("s0", "chat", 0, gamma=3, accepted=3, all_accepted=True)]
    assert token_accept_flags(records) == [True, True, True]
    # accepted=2, all_accepted=False -> 2 successes then 1 observed failure.
    records = [TraceRecord("s0", "chat", 1, gamma=3, accepted=2, all_accepted=False)]
    assert token_accept_flags(records) == [True, True, False]


def test_token_accept_flags_rejects_negative_accepted():
    with pytest.raises(ValueError):
        token_accept_flags([TraceRecord("s0", "chat", 0, gamma=1, accepted=-1, all_accepted=False)])


def test_group_by_stream_sorts_by_round_index():
    records = [
        TraceRecord("a", "chat", 2, 1, 1, True),
        TraceRecord("a", "chat", 0, 1, 1, True),
        TraceRecord("b", "code", 0, 1, 1, True),
    ]
    grouped = group_by_stream(records)
    assert list(grouped.keys()) == ["a", "b"]
    assert [r.round_index for r in grouped["a"]] == [0, 2]


def test_round_level_alpha_series_length_and_bounds():
    records = generate_stationary_trace("s0", "chat", alpha=0.7, num_rounds=50, gamma=4, seed=1)
    series = round_level_alpha_series(records, window_rounds=10)
    assert len(series) == 50
    assert all(0.0 <= a <= 1.0 for a in series)


def test_round_level_alpha_series_rejects_bad_window():
    with pytest.raises(ValueError):
        round_level_alpha_series([], window_rounds=0)


def test_build_trace_acceptance_round_trips_through_core_acceptance():
    records = generate_stationary_trace("s0", "chat", alpha=0.7, num_rounds=50, gamma=4, seed=1)
    trace_acc = build_trace_acceptance(records, window_rounds=10, seed=2)
    assert len(trace_acc.alpha_by_round) == 50
    # sample() should not raise for any in-range round index.
    trace_acc.sample(gamma=3, round_index=10)


def test_load_trace_jsonl_round_trips_write_and_read(tmp_path):
    records = generate_stationary_trace("s0", "chat", alpha=0.6, num_rounds=5, gamma=2, seed=0)
    path = tmp_path / "trace.jsonl"
    write_trace_jsonl(records, path)
    loaded = load_trace_jsonl(path)
    assert loaded == records


def test_domain_alpha_summary_separates_domains():
    chat = generate_stationary_trace("chat0", "chat", alpha=0.9, num_rounds=3000, gamma=4, seed=1)
    code = generate_stationary_trace("code0", "code", alpha=0.4, num_rounds=3000, gamma=4, seed=2)
    summary = domain_alpha_summary(chat + code)
    assert summary["chat"] > summary["code"]
    assert summary["chat"] == pytest.approx(0.9, abs=0.03)
    assert summary["code"] == pytest.approx(0.4, abs=0.03)


def test_lag1_autocorrelation_near_zero_for_stationary_iid_trace():
    records = generate_stationary_trace("s0", "chat", alpha=0.7, num_rounds=5000, gamma=4, seed=7)
    flags = token_accept_flags(records)
    ac = lag1_autocorrelation(flags)
    assert abs(ac) < 0.05  # i.i.d. by construction: no meaningful lag-1 correlation


def test_lag1_autocorrelation_detects_drifting_trace():
    # A strong, monotone drift from alpha=0.1 to alpha=0.95 creates a clear
    # trend, which shows up as strong positive lag-1 autocorrelation.
    records = generate_drifting_trace("s0", "chat", alpha_start=0.1, alpha_end=0.95, num_rounds=5000, gamma=4, seed=7)
    flags = token_accept_flags(records)
    ac = lag1_autocorrelation(flags)
    assert ac > 0.05  # detectably different from the stationary i.i.d. case


def test_lag1_autocorrelation_handles_degenerate_input():
    assert lag1_autocorrelation([]) == 0.0
    assert lag1_autocorrelation([True]) == 0.0
    assert lag1_autocorrelation([True, True, True]) == 0.0  # zero variance


def test_token_level_alpha_series_matches_core_sliding_window():
    records = generate_stationary_trace("s0", "chat", alpha=0.6, num_rounds=20, gamma=3, seed=3)
    series = token_level_alpha_series(records, window=10)
    assert len(series) == len(token_accept_flags(records))
