"""C2 Phase C: the offline verifier-cost diagnostics (loaded-run traces; not a hardware calibration; not activated in production)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "c2" / "verifier_cost_model_offline.json"


@pytest.fixture(scope="module")
def res():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "c2" / "verifier_cost_model.py")], check=True, capture_output=True)
    return json.loads(OUT.read_text())


def test_label_and_scope(res):
    assert "not an independent hardware calibration" in res["label"] and res["n_batches"] > 4000


def test_context_aware_model_beats_logical_positions_in_held_out_runs(res):
    lo = res["leave_one_run_out"]
    ctx, pos = lo["M2 full-context tokens (processed)"], lo["M1 theta_f*(gamma+1) positions (+intercept)"]
    assert ctx["mae_ms"] < 0.85 * pos["mae_ms"]                                   # ~46 vs ~63 ms
    assert ctx["mape"] < 0.15


def test_speculative_depth_effect_is_not_identified_by_these_traces(res):
    ci = res["coefficients"]["M3 ctx_sum + B + gamma_sum"]["gamma_sum"]["ci90_runblock_bootstrap"]
    assert ci[0] < 0 < ci[1]                                                      # gamma was ~constant (2): its cost cannot be estimated
    assert res["observed_ranges"]["gamma_sum"][1] / res["observed_ranges"]["B"][1] <= 2.0 + 1e-9


def test_extrapolation_beyond_the_training_range_is_poorly_calibrated(res):
    ood = res["ood"]["train B<=8 -> test B>=12 (batch-size extrapolation)"]
    assert all(v["pi90_coverage"] < 0.75 for v in ood.values())                   # nominal 90% intervals under-cover when extrapolating batch size


def test_marginal_request_cost_depends_on_context_unlike_the_theta_f_abstraction(res):
    m = res["marginal_request_cost_ms"]
    assert m["ctx=450,gamma=2"]["M3_marginal_ms"] > 2.0 * m["ctx=100,gamma=2"]["M3_marginal_ms"]
    assert m["ctx=450,gamma=2"]["theta_f_abstraction_ms"] == pytest.approx(m["ctx=100,gamma=2"]["theta_f_abstraction_ms"])
