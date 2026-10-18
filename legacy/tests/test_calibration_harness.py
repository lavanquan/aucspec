"""CPU-only tests for the calibration harness core and the fitting code (synthetic data, no GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import calibrate_verifier as cal  # noqa: E402
import fit_verifier_cost as fit  # noqa: E402


def test_schedule_counts_and_budget():
    s = cal.build_schedule(1)
    c = cal.expected_call_counts(s)
    assert (c["warmup"], c["timed"], c["val"]) == (225, 1500, 300)
    assert len(s["cells"]) == 75
    assert c["total_verifier_and_prep_calls"] == 2037 <= 2500
    per_cell = Counter((it["phase"], it["cell"]) for it in s["items"] if it["phase"] != "val")
    for cell in s["cells"]:
        assert per_cell[("warmup", cell["cell"])] == 3 and per_cell[("timed", cell["cell"])] == 20


def test_schedule_randomised_deterministic_and_phased():
    a, b, c = cal.build_schedule(1), cal.build_schedule(1), cal.build_schedule(2)
    assert [i["item_id"] for i in a["items"]] == [i["item_id"] for i in b["items"]]
    assert [i["item_id"] for i in a["items"]] != [i["item_id"] for i in c["items"]]
    phases = [i["phase"] for i in a["items"]]
    assert phases == sorted(phases, key=["warmup", "timed", "val"].index)
    rounds = {}
    for it in a["items"]:
        if it["phase"] == "timed":
            rounds.setdefault(it["rep"], []).append(it["cell"])
    orders = [tuple(v) for v in rounds.values()]
    assert all(len(set(o)) == 75 for o in orders) and len(set(orders)) > 1  # each round is a fresh permutation
    for v in a["val_cells"]:
        assert len(set(v["pool_idx"])) == v["B"] and len(v["gammas"]) == v["B"]
    assert Counter(v["B"] for v in a["val_cells"]) == {2: 8, 4: 8, 8: 7, 16: 7}


class FakeResult(SimpleNamespace):
    pass


def _fake_verify(wall_s=0.0, bad_cells=()):
    state = {"n": 0}

    def verify(payload):
        state["n"] += 1
        out = []
        for ctx, draft, _ in payload:
            g = len(draft.token_ids)
            acc = g - 1 if (g > 0 and state["n"] in bad_cells) else g
            out.append(FakeResult(accepted_length=acc, reached_eos=False, target_batch_wall_ms=10.0))
        return out

    return verify, state


def _payload_maker(items):
    def make(it):
        B = it["B"]
        g = it.get("gamma", 0)
        gammas = it.get("gammas", [g] * B)
        payload = [([0] * 100, SimpleNamespace(token_ids=[1] * gg), 96) for gg in gammas]
        return payload, {"gammas": gammas, "context_lens": [100] * B}

    return make


def test_run_schedule_records_resume_and_malformed():
    s = cal.build_schedule(3, warmup_reps=1, timed_reps=1, val_cells=4, val_reps=1)
    items = s["items"][:20]
    verify, st = _fake_verify(bad_cells={3, 4, 5, 6, 7, 8})
    recs = []
    n, early = cal.run_schedule(items, _payload_maker(items), verify, lambda: 4.0, recs.append, set())
    assert n == 20 and not early and len(recs) == 20
    r = recs[0]
    assert r["logical_tokens"] == sum(g + 1 for g in r["gamma_requested"])
    assert r["processed_tokens"] == r["sum_context"] + r["sum_gamma"]
    assert r["generate_ms"] == 4.0 and r["host_ms"] == pytest.approx(r["wall_ms"] - 4.0)
    bad = [x for x in recs if not x["well_formed"]]
    assert bad and all(sum(x["gamma_requested"]) > 0 for x in bad)  # only cells with proposals can be malformed
    done = {x["item_id"] for x in recs[:10]}
    more = []
    n2, _ = cal.run_schedule(items, _payload_maker(items), verify, lambda: 0.0, more.append, done)
    assert n2 == 10 and {x["item_id"] for x in more}.isdisjoint(done)


def test_run_schedule_deadline():
    s = cal.build_schedule(3, warmup_reps=1, timed_reps=1, val_cells=4, val_reps=1)
    t = {"v": 0.0}

    def clock():
        t["v"] += 1.0
        return t["v"]

    verify, _ = _fake_verify()
    recs = []
    n, early = cal.run_schedule(s["items"], _payload_maker(s["items"]), verify, lambda: 0.0, recs.append, set(),
                                deadline_s=10.0, clock=clock)
    assert early and n < len(s["items"])


# ----------------------------------------------------------------- fitting
def _synthetic(theta0=120.0, a=3.0, c=0.15, b=0.0, noise=3.0, knee=None, d=0.0, seed=0):
    rng = np.random.default_rng(seed)
    s = cal.build_schedule(1)
    recs = []
    for it in s["items"]:
        if it["phase"] == "warmup":
            continue
        if it["phase"] == "timed":
            B, L, gammas = it["B"], cal.CONTEXT_QUANTILES[it["Lq"]], [it["gamma"]] * it["B"]
            ctx = [L] * B
        else:
            B = it["B"]
            ctx = [int(rng.integers(100, 230)) for _ in range(B)]
            gammas = it["gammas"]
        SL, Sg = sum(ctx), sum(gammas)
        T = theta0 + a * B + c * SL + b * Sg + noise * rng.standard_normal()
        if knee is not None:
            T += d * max(0.0, SL + Sg - knee)
        gen = 0.9 * T
        recs.append({"item_id": it["item_id"], "phase": it["phase"], "rep": it["rep"], "cell": it["cell"], "B": B,
                     "well_formed": True, "context_lens": ctx, "gamma_requested": gammas, "gamma_realized_accepted": gammas, "sum_context": SL, "sum_gamma": Sg, "processed_tokens": SL + Sg,
                     "wall_ms": T, "stamped_batch_wall_ms": T, "generate_ms": gen, "host_ms": T - gen})
    return recs


def _run_fit(tmp_path, recs):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r) for r in recs))
    out = tmp_path / "fit.json"
    subprocess.run([sys.executable, str(REPO / "scripts" / "fit_verifier_cost.py"), str(raw), "--out", str(out),
                    "--n-boot", "200"], check=True, capture_output=True)
    return json.loads(out.read_text())


def test_fit_recovers_affine_and_zero_draft_cost(tmp_path):
    res = _run_fit(tmp_path, _synthetic(b=0.0))
    co = res["affine"]["coef"]
    assert co["theta0_ms"] == pytest.approx(120.0, abs=6.0)
    assert co["per_context_token_ms"] == pytest.approx(0.15, abs=0.01)
    assert res["draft_token_coefficient_indistinguishable_from_zero"] is True
    assert res["regime_test"]["supports_distinct_regimes"] is False
    assert res["held_out"]["heterogeneous_validation"]["test"]["rmse_ms"] < 6.0
    assert abs(res["paired_depth"]["pooled_ms_per_draft_token_per_request"]) < 0.5


def test_fit_detects_positive_draft_cost_and_paired_agrees(tmp_path):
    res = _run_fit(tmp_path, _synthetic(b=0.4))
    assert res["draft_token_coefficient_indistinguishable_from_zero"] is False
    assert res["components"]["marginal_cost_per_draft_token_ms"]["estimate"] == pytest.approx(0.4, abs=0.1)
    assert res["paired_depth"]["pooled_ms_per_draft_token_per_request"] == pytest.approx(0.4, abs=0.15)


def test_fit_flags_a_real_knee(tmp_path):
    res = _run_fit(tmp_path, _synthetic(knee=1500.0, d=0.2, noise=1.0))
    assert res["regime_test"]["supports_distinct_regimes"] is True
    assert res["regime_test"]["delta_bic_broken_minus_affine"] < -10
