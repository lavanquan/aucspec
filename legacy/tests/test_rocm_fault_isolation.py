"""CPU-only tests for the ROCm fault-isolation helpers.

Includes a check that the code-derived model of vLLM 0.9.1's ROCm naive-SDPA prefill (it receives the
cumulative `query_seq_start_loc` where it expects per-sequence lengths) predicts exactly which batch
slots were wrong in the earlier audit data (results/audit_20261019/summary_row_attribution.json).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import rocm_fi_lib as lib  # noqa: E402


def test_b1_and_b2_are_correct_for_any_lengths():
    assert lib.naive_sdpa_status([162]) == ["correct"]
    assert lib.naive_sdpa_status([162, 162]) == ["correct", "correct"]
    assert lib.naive_sdpa_status([100, 230]) == ["correct", "correct"]


def test_b4_equal_lengths_third_sequence_contaminated():
    assert lib.naive_sdpa_status([162] * 4) == ["correct", "correct", "contaminated", "correct"]


def test_b16_equal_lengths_predicts_the_observed_good_slots():
    st = lib.naive_sdpa_status([162] * 16)
    good = {i for i, s in enumerate(st) if s == "correct"}
    assert good == {0, 1, 3, 6, 10, 15}
    assert all(s == "contaminated" for i, s in enumerate(st) if i not in good)


def test_prediction_matches_earlier_audit_data():
    p = REPO / "results" / "audit_20261019" / "summary_row_attribution.json"
    if not p.exists():
        pytest.skip("audit data not present")
    wrong = json.loads(p.read_text())["B16_vs_B1"]["wrong_by_ctx_slot_in_group"]
    pred_bad = {i for i, s in enumerate(lib.naive_sdpa_status([162] * 16)) if s != "correct"}
    agree = 0
    for c in range(32):
        observed_bad = wrong[str(c)] >= 3  # wrong at >= 3 of the 4 candidate positions
        agree += observed_bad == ((c % 16) in pred_bad)
    assert agree >= 31  # 31 of 32 contexts; the single exception is a 3-of-4 case on a near tie


def test_b4_prediction_matches_earlier_audit_data():
    p = REPO / "results" / "audit_20261019" / "summary_row_attribution.json"
    if not p.exists():
        pytest.skip("audit data not present")
    wrong = json.loads(p.read_text())["B4_vs_B1"]["wrong_by_ctx_slot_in_group"]
    pred_bad = {i for i, s in enumerate(lib.naive_sdpa_status([162] * 4)) if s != "correct"}
    assert all((wrong[str(c)] >= 3) == ((c % 4) in pred_bad) for c in range(32) if c % 4 != 1)


def test_fraction_of_contaminated_requests_in_the_n10_smoke_batch_mix():
    counts = {1: 3063, 2: 2553, 3: 1378, 4: 375, 5: 74, 6: 16, 7: 1}
    bad = sum(n * sum(s != "correct" for s in lib.naive_sdpa_status([160] * b)) for b, n in counts.items())
    tot = sum(n * b for b, n in counts.items())
    assert tot == 14276 and 0.12 < bad / tot < 0.15


def test_perm_and_hash_are_deterministic():
    assert lib.make_perm(32, 1) == lib.make_perm(32, 1) and lib.make_perm(32, 1) != lib.make_perm(32, 2)
    assert sorted(lib.make_perm(16, 3)) == list(range(16))
    assert lib.token_hash([1, 2, 3]) == lib.token_hash([1, 2, 3]) != lib.token_hash([1, 2, 4])


def test_agreement_gap_filter_and_gate():
    rows = [{"match": True, "fp32_gap": 2.0}, {"match": False, "fp32_gap": 0.1}, {"match": True, "fp32_gap": 1.0}]
    assert lib.agreement(rows, 0.0)["rate"] == pytest.approx(2 / 3)
    assert lib.agreement(rows, lib.GAP_OK) == {"n": 2, "rate": 1.0}
    assert lib.passes_gate({"a": 1.0, "b": 0.995}) and not lib.passes_gate({"a": 1.0, "b": 0.98}) \
        and not lib.passes_gate({"a": None})


def test_slot_consistency_separates_input_from_slot():
    a = {0: False, 1: True, 2: False, 3: True}
    b = {0: False, 1: True, 2: True, 3: False}
    sa = {0: 0, 1: 1, 2: 2, 3: 3}
    sb = {0: 3, 1: 2, 2: 1, 3: 0}
    r = lib.slot_consistency(a, b, sa, sb)
    assert (r["both_wrong"], r["both_correct"], r["wrong_only_in_order_a"], r["wrong_only_in_order_b"]) == (1, 1, 1, 1)


def test_orchestrator_score_and_scripts_compile():
    import py_compile

    for f in ("rocm_fi_prepare.py", "rocm_fi_run_config.py", "rocm_fi_orchestrate.py"):
        py_compile.compile(str(REPO / "scripts" / f), doraise=True)
    import rocm_fi_orchestrate as orch

    ok = {"status": "ok", "dev": {"gate_inputs": {"B16_ident_vs_fp32_gap>=0.5": 0.5, "B4_ident_vs_fp32_gap>=0.5": 0.9}}}
    assert orch.score(ok) == pytest.approx(0.7) and orch.score({"status": "failed"}) == -1.0
