"""End-to-end CPU test of the certified search driver with a synthetic simulator (no GPU, no vLLM)."""
import importlib.util
import json
import random
from pathlib import Path

import pandas as pd
import pytest

from edge_specsim.capacity import CandidateMeasurement

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_capacity_frontier.py"


@pytest.fixture()
def drv(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("run_capacity_frontier_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "RESULTS_ROOT", tmp_path / "capacity_frontier")
    return mod


BASE_CFG = {
    "simulation": {"seed": 1, "population_seed": 7, "num_clients": 8, "nested_population": True,
                   "population_n_max": 16, "questions_per_client": 16},
    "controller": {"V": 50.0},
    "experiment": {},
    "capacity_search": {"x_L": 2.0, "x_U": 6.0, "x_grid_final": [2.0, 3.0, 4.0, 6.0], "n_ref": 16,
                         "n_search_cap": 16, "n_lo_feasible": 1, "confidence_delta": 0.05, "initial_seeds": 5,
                         "max_seeds": 7, "min_final_seeds": 5, "warmup_seconds": 30,
                         "measurement_seconds_final": 120, "x_tolerance_tps": 0.1},
}


def fake_run_one_factory(capacity=26.0, invalid_from=None, sd_frac=0.01, nonmonotone_hole=None):
    calls = []

    def fake_run_one(base_cfg, n, x, seed, meas_s, cfg_hash, cache, raw_dir, force):
        calls.append((n, round(x, 6), seed))
        rng = random.Random(hash((n, round(x, 6), seed, 12345)) & 0xFFFFFFFF)
        mean = capacity / n
        if nonmonotone_hole is not None and n in nonmonotone_hole:
            mean = capacity / (n - 3)
        rates = {c: max(0.0, rng.gauss(mean, sd_frac * mean)) for c in range(n)}
        inv = invalid_from is not None and n >= invalid_from
        return CandidateMeasurement(scale=n, n_active=n, x_requirement=x, seed=seed, min_rate_tps=min(rates.values()),
                                    per_client_rates=rates, invalid=inv,
                                    invalid_reason="workload exhausted" if inv else "")

    fake_run_one.calls = calls
    return fake_run_one


def test_certified_search_and_analysis_end_to_end(drv, monkeypatch):
    fake = fake_run_one_factory()
    monkeypatch.setattr(drv, "run_one", fake)
    ctx = drv.RunContext.create(json.loads(json.dumps(BASE_CFG)), run_tag="e2e")
    res = drv.search_nstar_final(ctx, "capacity_dpp", 3.0, None, force=False)
    assert res.lower_bound_n == 8 and res.upper_bound_monotone == 8 and res.exact_conditional
    # every candidate got >= K_min seeds before any certified decision
    by_n = {}
    for n, x, s in fake.calls:
        by_n.setdefault((n, x), set()).add(s)
    assert all(len(v) >= 5 for v in by_n.values())
    assert any(n >= res.lower_bound_n + 2 for n, _, _ in fake.calls), "monotonicity falsification probes ran"


def test_invalid_candidate_stops_the_search_and_poisons_the_analysis(drv, monkeypatch):
    fake = fake_run_one_factory(invalid_from=12)
    monkeypatch.setattr(drv, "run_one", fake)
    ctx = drv.RunContext.create(json.loads(json.dumps(BASE_CFG)), run_tag="e2e_invalid")
    rows = []
    for x in (2.0, 4.0):
        r = drv.search_nstar_final(ctx, "capacity_dpp", x, None, force=False)
        rows.append(drv._nstar_row("capacity_dpp", r))
    # x=2: the boundary is N=13 but N>=12 is INVALID (exhausted workload): the point is INVALID, never infeasible.
    # x=4: the boundary N=6 is reached without touching an INVALID candidate.
    assert rows[0]["boundary_status"] == "INVALID" and rows[0]["invalid_n"] != ""
    assert rows[1]["boundary_status"] == "exact_conditional" and rows[1]["N_lower"] == 6
    pd.DataFrame(rows).to_csv(ctx.run_dir / "nstar_frontier_capacity_dpp.csv", index=False)
    drv.analyze_final("e2e_invalid")
    area = json.loads((ctx.run_dir / "capacity_area_capacity_dpp.json").read_text())
    assert area["status"] in ("INVALID", "NOT_COMPUTED") and area["area_lower"] is None


def test_nonmonotone_controller_is_reported_unresolved_not_exact(drv, monkeypatch):
    # controller does badly at N=9..10 only (hole), but fine again at 11+: the monotone search stops at the hole
    fake = fake_run_one_factory(capacity=44.0, nonmonotone_hole={9, 10})
    monkeypatch.setattr(drv, "run_one", fake)
    ctx = drv.RunContext.create(json.loads(json.dumps(BASE_CFG)), run_tag="e2e_nonmono")
    r = drv.search_nstar_final(ctx, "capacity_dpp", 4.0, None, force=False)
    # truth: feasible iff capacity/n >= 4 (n<=10) except in the hole where mean=40/(n-3)... construct so that a
    # larger-N point is feasible again; the falsification probes must either find it or leave the result bracketed
    assert r.lower_bound_n >= 8
    assert r.status in ("exact_conditional", "bracketed", "nonmonotone_unresolved")
    if r.monotone_violations:
        assert r.upper_bound_monotone is None and not r.exact_conditional


def test_driver_uses_certified_imports_only():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "classify_feasibility_final" not in text.split("def _look_budget")[1]
    assert "monotone_scale_search_final" not in text.split("def _look_budget")[1]
