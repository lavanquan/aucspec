"""C1.4: the ORIGINAL gate is unchanged (real/virtual <= 3 is a failure); the EXPLORATORY_V2 protocol records the ratio as telemetry only
while every scientific-validity requirement stays fail-closed."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("pg_v2", ROOT / "scripts" / "c1" / "pilot_gate.py")
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)

WIN = (30000.0, 150000.0)


def valid_run(n, cap=16, wall=200.0, ratio=1.5):
    rows, bid = [], 0
    for k in range(170):                                   # one cohort per virtual second
        t0 = 1000.0 * k + 200.0
        clients = list(range(n))
        for i in range(0, n, cap):
            members = clients[i:i + cap]
            start = t0 + (i // cap) * wall
            fin = start + wall
            for c in members:
                rows.append(dict(client_id=c, round_id=k, sample_id=f"s{c}_{k // 20}", draft_start_ms=t0 - 150.0,
                                 client_receive_ms=fin + 25.0, committed_token_count=3, useful_tokens=3,
                                 verification_batch_id=bid, verification_batch_size=len(members), verification_batch_wall_ms=wall,
                                 verification_batch_service_ms=wall, batch_start_ms=start, verify_finish_ms=fin,
                                 server_arrival_ms=t0 - 50.0, draft_latency_ms=44.0, virtual_round_latency_ms=1000.0,
                                 wall_clock_time_ms=(fin + 25.0) * ratio, experiment_warmup_end_ms=WIN[0],
                                 experiment_measurement_end_ms=WIN[1], client_samples_assigned=64, sample_finished=False,
                                 client_samples_pending=63, client_work_complete=False, verifier_max_batch_size=cap))
            bid += 1
    return pd.DataFrame(rows)


def write(tmp_path, df, *, runtime=None, viol=0, name="fixed_gamma_n24_x3.0_s1"):
    csv = tmp_path / f"{name}.csv"
    df.to_csv(csv, index=False)
    art = tmp_path / f"{name}_artifacts"
    art.mkdir(exist_ok=True)
    (art / "causality_check.json").write_text(json.dumps({"n_violations": viol, "n_events": 10}))
    live = {"n_attention_layers": 28, "n_expected_layers": 28, "layers_triton_flash_attn": 28, "layers_naive_sdpa": 0,
            "effective_max_logprobs": 1, "prefix_caching": False}
    ri = {"identity": pg.FROZEN_IDENTITY, "manifest_sha256": pg.FROZEN_MANIFEST_SHA, "live_engine": live}
    if runtime:
        ri = runtime(ri)
    (art / "run_config.json").write_text(json.dumps({"runtime_identity": ri}))
    return csv, art


def v2(tmp_path, df, **kw):
    csv, art = write(tmp_path, df, **kw)
    return pg.gate_exploratory_v2(csv, df.client_id.nunique(), art / "causality_check.json", 16, art / "run_config.json")


def test_a_valid_slow_run_passes_v2_but_fails_the_original_gate_on_the_ratio(tmp_path):
    df = valid_run(24, ratio=3.07)
    ok, why, tel = v2(tmp_path, df)
    assert ok, why
    assert tel["real_over_virtual"] == pytest.approx(3.07, rel=0.02) and tel["ratio_is_telemetry_only"]
    csv, art = write(tmp_path, df, name="orig")
    ok0, why0 = pg.gate(csv, 24, art / "causality_check.json", 16)
    assert not ok0 and any("real/virtual" in w for w in why0)          # original protocol unchanged


def test_original_gate_still_passes_a_fast_run(tmp_path):
    csv, art = write(tmp_path, valid_run(16, ratio=1.2), name="fast")
    assert pg.gate(csv, 16, art / "causality_check.json", 16)[0]


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.assign(batch_start_ms=d.batch_start_ms.where(d.verification_batch_id != 3, d.batch_start_ms - 120.0)), "overlap"),
    (lambda d: d.assign(verification_batch_service_ms=d.verification_batch_service_ms + 5.0), "batch_wall accounting"),
    (lambda d: pd.concat([d, d.iloc[:1]], ignore_index=True), "duplicated"),
    (lambda d: d.drop(d.index[10]).reset_index(drop=True), "dropped or duplicated"),
    (lambda d: d.assign(useful_tokens=d.useful_tokens + 1), "useful_tokens"),
    (lambda d: d.assign(committed_token_count=d.committed_token_count.astype(float).where(d.index != 5, float("nan"))), "non-finite"),
    (lambda d: d.assign(verifier_max_batch_size=32), "max_batch_size"),
    (lambda d: d.assign(client_work_complete=d.client_receive_ms > 100000.0), "INVALID"),
])
def test_data_level_validity_failures_stay_fail_closed(tmp_path, mutate, needle):
    ok, why, _ = v2(tmp_path, mutate(valid_run(24)))
    assert not ok and any(needle in w for w in why), why


def test_runtime_identity_and_attention_fallback_are_fail_closed(tmp_path):
    def naive(ri):
        ri["live_engine"] = dict(ri["live_engine"], layers_triton_flash_attn=27, layers_naive_sdpa=1)
        return ri

    ok, why, _ = v2(tmp_path, valid_run(24), runtime=naive)
    assert not ok and any("fallback" in w for w in why)
    ok, why, _ = v2(tmp_path, valid_run(24), runtime=lambda ri: dict(ri, manifest_sha256="0" * 64), name="fixed_gamma_n24_x3.0_s2")
    assert not ok and any("manifest" in w for w in why)
    csv, art = write(tmp_path, valid_run(24), name="fixed_gamma_n24_x3.0_s3")
    (art / "run_config.json").unlink()
    assert not pg.gate_exploratory_v2(csv, 24, art / "causality_check.json", 16, art / "run_config.json")[0]


def test_causality_violation_and_too_few_rounds_stay_fail_closed(tmp_path):
    ok, why, _ = v2(tmp_path, valid_run(24), viol=2)
    assert not ok and any("causality" in w for w in why)
    short = valid_run(24)
    short = short[short.round_id < 33]                                   # ~3 s inside the window
    ok, why, _ = v2(tmp_path, short, name="fixed_gamma_n24_x3.0_s9")
    assert not ok and any("20 in-window" in w for w in why)


def test_cli_selects_the_protocol_only_through_the_explicit_environment_variable(tmp_path, monkeypatch, capsys):
    import subprocess
    import sys
    csv, art = write(tmp_path, valid_run(24, ratio=3.5), name="fixed_gamma_n24_x3.0_s1")
    env = {"EXPECT_MAX_BATCH": "16", "PATH": "", "SYSTEMROOT": "C:\\Windows"}
    import os
    base = dict(os.environ)
    base.update(env)
    r0 = subprocess.run([sys.executable, str(ROOT / "scripts" / "c1" / "pilot_gate.py"), str(csv), "24"], env=base, capture_output=True, text=True)
    assert r0.returncode == 3 and '"protocol": "original"' in r0.stdout
    base["GATE_PROTOCOL"] = "exploratory_v2"
    r1 = subprocess.run([sys.executable, str(ROOT / "scripts" / "c1" / "pilot_gate.py"), str(csv), "24"], env=base, capture_output=True, text=True)
    assert r1.returncode == 0 and '"protocol": "exploratory_v2"' in r1.stdout and "ratio_is_telemetry_only" in r1.stdout
