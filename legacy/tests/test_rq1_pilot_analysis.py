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
            m["seed"] = s + 1
            out[n].append(m)
    return out


def test_goodput_plateau_with_inverse_n_floor_supports_rq1():
    v = an.rq1_verdict(runs(lambda n: min(9.0, 24.0 / n)))        # Y = min(9N, 24): plateau at N>=3
    assert v["verdict"].startswith("SUPPORTS"), v["verdict"]
    assert abs(v["N8_to_N16"]["x_min_ratio_mean"] - 0.5) < 0.05


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
    bad = an.run_metrics(synthetic_run(16, 1.5, complete_at=100000.0), 16)
    bad["seed"] = 1
    r[16] = [bad]
    assert bad["invalid"]
    v = an.rq1_verdict(r)
    assert v["invalid_seeds_by_N"] == {16: [1]} and v["verdict"].startswith("INCONCLUSIVE")


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


def _good():
    return runs(lambda n: min(9.0, 24.0 / n))


def test_pairing_is_by_explicit_seed_not_position():
    r = _good()
    ref = an.rq1_verdict(r)
    r[16] = list(reversed(r[16]))                       # shuffled input order
    r[8] = [r[8][1], r[8][2], r[8][0]]
    sh = an.rq1_verdict(r)
    assert sh["verdict"] == ref["verdict"] and sh["N8_to_N16"]["per_seed"] == ref["N8_to_N16"]["per_seed"]
    # a different seed's run in the same position must NOT be silently paired: perturb seed 2 at N=16 only
    r2 = _good()
    for m in r2[16]:
        if m["seed"] == 2:
            m["Y"] *= 1.5
            m["x_min"] *= 1.5
    v = an.rq1_verdict(r2)
    assert v["N8_to_N16"]["per_seed"][2]["Y_ratio"] > 1.4 and v["N8_to_N16"]["per_seed"][1]["Y_ratio"] < 1.05


def test_duplicate_seed_records_are_rejected():
    import pytest
    r = _good()
    r[8].append(dict(r[8][0]))
    with pytest.raises(ValueError):
        an.rq1_verdict(r)


def test_missing_seed_gives_partial_inconclusive_with_diagnostics():
    r = _good()
    r[16] = [m for m in r[16] if m["seed"] != 3]
    v = an.rq1_verdict(r)
    assert v["verdict"].startswith("INCONCLUSIVE(partial)")
    assert v["N8_to_N16"]["valid_pairs"] == [1, 2] and v["N8_to_N16"]["missing_or_invalid_required_seeds"] == [3]
    assert v["N8_to_N16"]["per_seed"][1]["x_min_ratio"] < 0.6          # partial numbers retained as diagnostics


def test_single_seed_never_yields_a_substantive_verdict():
    r = {n: [m for m in ms if m["seed"] == 1] for n, ms in _good().items()}
    assert an.rq1_verdict(r)["verdict"].startswith("INCONCLUSIVE(partial)")


def test_invalid_intermediate_run_does_not_break_pairing_of_the_others():
    r = _good()
    bad = next(m for m in r[4] if m["seed"] == 2)
    bad["invalid"], bad["invalid_reason"] = True, "x"
    v = an.rq1_verdict(r)
    assert v["invalid_seeds_by_N"] == {4: [2]} and v["verdict"].startswith("SUPPORTS")
    assert v["per_N"][4]["seeds"] == [1, 3]


def test_run_without_seed_is_rejected():
    import pytest
    r = _good()
    del r[8][0]["seed"]
    with pytest.raises(ValueError):
        an.rq1_verdict(r)


def test_gate_csv_and_artifact_paths_match_what_the_driver_really_writes(tmp_path, monkeypatch):
    """Run the real run_one() with a fake simulator subprocess and look the outputs up the way the job script does."""
    import json as _json
    import subprocess
    import yaml as _yaml
    drv_spec = importlib.util.spec_from_file_location("rcf_paths", SCRIPT.parents[1] / "run_capacity_frontier.py")
    rcf = importlib.util.module_from_spec(drv_spec)
    drv_spec.loader.exec_module(rcf)
    cfg = _yaml.safe_load((SCRIPT.parents[2] / "configs" / "rq1_pilot_fixed_gamma2_tritonfa33.yaml").read_text())
    scratch = tmp_path / "scratch"
    (scratch / "TAG" / "raw").mkdir(parents=True)
    root = tmp_path / "repo"
    (root / "results" / "capacity_frontier" / "TAG").mkdir(parents=True)
    monkeypatch.setattr(rcf, "REPO_ROOT", root)
    run_dir_raw = root / "results" / "capacity_frontier" / "TAG" / "raw"
    run_dir_raw.symlink_to(scratch / "TAG" / "raw", target_is_directory=True) if hasattr(run_dir_raw, "symlink_to") else None
    rcf._ACTIVE_POLICY = cfg["controller"]["policy"]

    def fake_sim(cmd, cwd=None, capture_output=False):
        run_cfg = _yaml.safe_load(Path(cmd[cmd.index("--config") + 1]).read_text())
        out = Path(cwd) / run_cfg["simulation"]["output_csv"]
        df = synthetic_run(2, 3.0)
        df["useful_tokens"] = df["committed_token_count"]
        for col, v in (("gamma", 2), ("server_queue", 0.0), ("z_queue", 0.0), ("device_queue", 0.0)):
            df[col] = v
        df["in_measurement_window"] = (df["client_receive_ms"] >= 30000.0) & (df["client_receive_ms"] <= 150000.0)
        df.to_csv(out, index=False)
        art = out.parent / (out.stem + "_artifacts")
        art.mkdir()
        (art / "causality_check.json").write_text(_json.dumps({"n_violations": 0}))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(rcf.subprocess, "run", fake_sim)
    cache = rcf.CandidateCache(tmp_path / "cache.jsonl")
    rcf.run_one(cfg, 2, 3.0, 1, 120.0, "h", cache, root / "results" / "capacity_frontier" / "TAG" / "raw", False)
    gate_spec = importlib.util.spec_from_file_location("pilot_gate2", SCRIPT.parent / "pilot_gate.py")
    pg = importlib.util.module_from_spec(gate_spec)
    gate_spec.loader.exec_module(pg)
    csv = pg.csv_path(scratch, "TAG", 2, 1)
    assert csv.exists(), csv
    ok, why = pg.gate(csv, 2, Path(str(csv).replace(".csv", "_artifacts")) / "causality_check.json")
    assert ok, why
