"""The pre-registered RQ1 decision rules on synthetic pilots."""
import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "c1" / "analyze_rq1_pilot.py"
spec = importlib.util.spec_from_file_location("analyze_rq1_pilot", SCRIPT)
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)


def synthetic_run(n, x_each, *, window=(30000.0, 150000.0), tokens_per_round=3, complete_at=None, wall_ratio=1.0, seed=0):
    """Every client delivers tokens at rate x_each (tok/s); one round per tokens_per_round/x_each seconds."""
    rows = []
    period = tokens_per_round / x_each * 1000.0
    for c in range(n):
        t = 1000.0 + 37.0 * c
        k = 0
        while t < window[1] + 10000.0:
            t += period
            rows.append(dict(client_id=c, round_id=k, draft_start_ms=t - period, client_receive_ms=t, committed_token_count=tokens_per_round,
                             verification_batch_id=int(t // 200), verification_batch_size=min(n, 4), verification_batch_wall_ms=100.0,
                             batch_start_ms=t - 100.0, server_arrival_ms=t - 300.0, draft_latency_ms=44.0,
                             virtual_round_latency_ms=period, wall_clock_time_ms=t * wall_ratio,
                             experiment_warmup_end_ms=window[0], experiment_measurement_end_ms=window[1],
                             client_samples_assigned=64, sample_finished=False, client_samples_pending=63,
                             client_work_complete=bool(complete_at is not None and t >= complete_at)))
            k += 1
    return pd.DataFrame(rows)


def runs(rate_fn, ns=(1, 2, 4, 8, 12, 16), seeds=3, **kw):
    out = {}
    for n in ns:
        out[n] = []
        for s in range(seeds):
            m = an.run_metrics(synthetic_run(n, rate_fn(n) * (1 + 0.01 * (s - 1)), **kw), n)
            out[n].append(m)
    return out


def test_goodput_plateau_with_inverse_n_floor_supports_rq1():
    v = an.rq1_verdict(runs(lambda n: min(9.0, 24.0 / n)))        # Y = min(9N, 24): plateau at N>=3
    assert v["verdict"].startswith("SUPPORTS"), v["verdict"]
    assert abs(v["N8_to_N16"]["x_min_ratio"] - 0.5) < 0.05


def test_flat_floor_with_plateau_falsifies_the_concealment_claim():
    v = an.rq1_verdict(runs(lambda n: 2.0, ns=(1, 2, 4, 8, 12, 16)))   # Y grows linearly -> unsaturated
    assert v["verdict"].startswith("INCONCLUSIVE")


def test_floor_that_does_not_fall_while_goodput_plateaus_is_a_falsification():
    r = runs(lambda n: 3.0)
    for n in (8, 16):                                  # force a plateau in Y with an unchanged floor
        for m in r[n]:
            m["Y"] = 24.0
    v = an.rq1_verdict(r)
    assert v["verdict"].startswith("FALSIFIES"), v["verdict"]


def test_goodput_that_falls_is_flagged_as_pathology():
    v = an.rq1_verdict(runs(lambda n: 40.0 / (n * n) if n >= 8 else 9.0))
    assert v["verdict"].startswith("PATHOLOGY"), v["verdict"]


def test_exhausted_runs_are_excluded_and_missing_points_make_the_pilot_inconclusive():
    r = runs(lambda n: min(9.0, 24.0 / n))
    r[16] = [an.run_metrics(synthetic_run(16, 1.5, complete_at=100000.0), 16)]
    assert r[16][0]["invalid"]
    v = an.rq1_verdict(r)
    assert v["invalid_runs_by_N"] == {16: 1} and v["verdict"].startswith("INCONCLUSIVE")


def test_metrics_use_the_exact_window_denominator_and_report_components():
    df = synthetic_run(4, 5.0)
    m = an.run_metrics(df, 4)
    assert abs(m["x_min"] - 5.0) < 0.2 and abs(m["Y"] - 20.0) < 0.8
    assert m["batch_wall_ms_mean"] == 100.0 and m["draft_ms_mean"] == 44.0 and abs(m["real_over_virtual"] - 1.0) < 1e-9


def test_pilot_gate_stops_on_causality_violation_and_slow_real_time(tmp_path):
    gate_spec = importlib.util.spec_from_file_location("pilot_gate", SCRIPT.parent / "pilot_gate.py")
    pg = importlib.util.module_from_spec(gate_spec)
    gate_spec.loader.exec_module(pg)
    csv = tmp_path / "r.csv"
    synthetic_run(8, 3.0).to_csv(csv, index=False)
    chk = tmp_path / "chk.json"
    chk.write_text('{"n_violations": 0}')
    ok, why = pg.gate(csv, 8, chk)
    assert ok, why
    chk.write_text('{"n_violations": 2}')
    assert not pg.gate(csv, 8, chk)[0]
    slow = tmp_path / "slow.csv"
    synthetic_run(8, 3.0, wall_ratio=4.0).to_csv(slow, index=False)
    ok, why = pg.gate(slow, 8, None)
    assert not ok and any("real/virtual" in w for w in why)
