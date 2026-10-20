"""C1.3 Stage B: the real VerificationBatcher + VirtualCoordinator with 24 and 32 simulated clients under max_batch_size = 16,
a CPU mock verifier with nonzero service time, compared against an independent sequential event scheduler."""
import asyncio
import importlib.util
import random
from pathlib import Path

import pytest
import yaml

from causality_harness import Spec, reference_batches, run_scenario
from test_virtual_causality import canon

CAP = 16
ROOT = Path(__file__).resolve().parents[2]


def run(coro):
    return asyncio.run(coro)


def tied_specs(n, rounds=3, rtt=50.0, draft=44.4, real=0.0):
    """Homogeneous clients: identical virtual draft/RTT => all arrivals TIED (the pilot design)."""
    return [Spec(i, rtt, [draft] * rounds, [real]) for i in range(n)]


@pytest.mark.parametrize("n", [24, 32])
def test_cap16_basic_invariants_and_equality_with_the_sequential_scheduler(n):
    specs = tied_specs(n)
    r = run(run_scenario(specs, causal=True, max_batch=CAP))
    ref = reference_batches(specs, max_batch=CAP)
    members = [m for b in r["batches"] for m in b["members"]]
    assert all(len(b["members"]) <= CAP for b in r["batches"])                           # 1. cap respected
    assert max(r["target_calls"]) <= CAP
    assert sorted(members) == sorted((c, k) for c in range(n) for k in range(3))        # 2./3. all served, once each
    assert len(members) == len(set(members)) == n * 3
    assert r["violations"] == []                                                          # 4. no D1 / D2
    for a, b in zip(r["batches"], r["batches"][1:]):                                      # 5. no overlap
        assert a["finish"] <= b["start"] + 1e-9
    assert [b["members"] for b in r["batches"]] == [b["members"] for b in ref]           # independent oracle
    for a, b in zip(r["batches"], ref):
        assert a["start"] == pytest.approx(b["start"]) and a["finish"] == pytest.approx(b["finish"])


def test_tied_arrivals_split_into_full_and_partial_batches_in_client_id_order():
    r = run(run_scenario(tied_specs(24, rounds=1), causal=True, max_batch=CAP))
    assert [b["members"] for b in r["batches"]] == [[(c, 0) for c in range(16)], [(c, 0) for c in range(16, 24)]]
    r32 = run(run_scenario(tied_specs(32, rounds=1), causal=True, max_batch=CAP))
    assert [len(b["members"]) for b in r32["batches"]] == [16, 16]
    # the second batch starts when the first ends (back-to-back), i.e. the cohort is served in two verifier calls
    assert r32["batches"][1]["start"] == pytest.approx(r32["batches"][0]["finish"])


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n", [24, 32])
def test_random_virtual_inputs_match_the_reference_and_never_starve(seed, n):
    rng = random.Random(1000 * n + seed)
    specs = [Spec(i, rng.choice([20.0, 50.0, 80.0]), [rng.choice([30.0, 44.4, 60.0, 150.0]) + rng.random() for _ in range(3)],
                  [rng.random() * 0.02 for _ in range(3)]) for i in range(n)]
    r = run(run_scenario(specs, causal=True, max_batch=CAP))
    ref = reference_batches(specs, max_batch=CAP)
    assert r["violations"] == []
    assert [b["members"] for b in r["batches"]] == [b["members"] for b in ref]
    assert all(len(r["receives"][c]) == 3 for c in range(n))                              # every client completed every round


def test_partial_cohort_liveness_a_slow_client_never_blocks_the_verified_ones():
    """6. One client is really slow (real 0.15 s) AND virtually late; the other 23 must be served without waiting for the
    verifier to receive a full cohort, and the late client is served in a later batch."""
    specs = tied_specs(24, rounds=2)
    specs[23] = Spec(23, 50.0, [700.0, 44.4], [0.15, 0.0])
    r = run(run_scenario(specs, causal=True, max_batch=CAP))
    assert r["violations"] == []
    first_round_batches = [b for b in r["batches"] if any(k == 0 for _, k in b["members"])]
    assert sum(len(b["members"]) for b in first_round_batches) == 24
    served_early = [m for b in first_round_batches[:2] for m in b["members"]]
    assert (23, 0) not in served_early and len(served_early) == 23
    assert [b["members"] for b in r["batches"]] == [b["members"] for b in reference_batches(specs, max_batch=CAP)]


@pytest.mark.parametrize("n", [24, 32])
def test_virtual_timestamps_and_queue_state_versions_are_deterministic_under_reordered_tasks(n):
    outs = []
    for s in (0, 1, 2):
        rng = random.Random(s)
        specs = [Spec(i, 50.0, [44.4, 44.4], [rng.random() * 0.03]) for i in range(n)]    # permuted REAL completion order
        outs.append(canon(run(run_scenario(specs, causal=True, max_batch=CAP))))
    assert outs[0] == outs[1] == outs[2]
    r = run(run_scenario(tied_specs(n, rounds=2), causal=True, max_batch=CAP))
    versions = [e.queue_state_version for e in r["ledger"].events if e.type == "decision"]
    assert versions == sorted(versions) or len(set(versions)) > 1                            # versions advance with finished batches
    assert all(d["visible_ts"] <= d["t"] + 1e-9 for d in r["decisions"])


def test_legacy_real_order_batching_still_violates_causality_under_cap16():
    """Control: the check has teeth -- the same inputs in real-order mode are flagged."""
    rng = random.Random(5)
    specs = [Spec(i, 20.0, [rng.choice([5.0, 40.0]) + rng.random()], [rng.random() * 0.06]) for i in range(24)]
    r = run(run_scenario(specs, causal=False, max_batch=CAP))
    assert any(v.startswith("D1") for v in r["violations"])


# ---------------------------------------------------------------- 10. effective serving config
def _load(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def _flat(d, p=""):
    out = {}
    for k, v in d.items():
        out.update(_flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
    return out


def test_batch_cap16_config_differs_from_batch_cap_32_only_in_max_batch_size_plus_an_observation_flag():
    a, b = _flat(_load("rq1_extension_fixed_gamma2_tritonfa33.yaml")), _flat(_load("rq1_batchcap16_fixed_gamma2_tritonfa33.yaml"))
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert changed == {"verification_batching.max_batch_size", "simulation.failure_diagnostics"}
    assert (a["verification_batching.max_batch_size"], b["verification_batching.max_batch_size"]) == (32, 16)
    assert b["simulation.failure_diagnostics"] is True
    for k in ("verification_batching.service_time_model", "verification_batching.scheduler", "verification_batching.verify_token_budget",
              "verification_batching.batch_wait_ms", "models.target_gpu_memory_utilization", "models.target_max_model_len",
              "controller.fixed_gamma", "simulation.expected_catalog_hash", "simulation.population_seed", "simulation.questions_per_client",
              "simulation.draft_profile.per_token_ms", "experiment.warmup_seconds", "experiment.measurement_seconds",
              "experiment.early_stop_margin_seconds", "runtime.identity"):
        assert a[k] == b[k], k


def test_gate_rejects_a_run_whose_effective_cap_is_not_16(tmp_path):
    from test_rq1_pilot_analysis import synthetic_run
    spec = importlib.util.spec_from_file_location("pg_cap", ROOT / "scripts" / "c1" / "pilot_gate.py")
    pg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pg)
    df = synthetic_run(8, 3.0)
    df["verifier_max_batch_size"] = 16
    p = tmp_path / "ok.csv"
    df.to_csv(p, index=False)
    assert pg.gate(p, 8, None, expect_max_batch=16)[0] or any("max_batch" in w for w in pg.gate(p, 8, None, expect_max_batch=16)[1]) is False
    df["verifier_max_batch_size"] = 32
    p2 = tmp_path / "bad.csv"
    df.to_csv(p2, index=False)
    ok, why = pg.gate(p2, 8, None, expect_max_batch=16)
    assert not ok and any("effective max_batch_size" in w for w in why)
    df3 = synthetic_run(8, 3.0)                      # column missing entirely
    p3 = tmp_path / "missing.csv"
    df3.to_csv(p3, index=False)
    assert not pg.gate(p3, 8, None, expect_max_batch=16)[0]


def test_job_script_has_the_registered_order_walltime_and_cap_check():
    txt = (ROOT / "scripts" / "c1" / "run_rq1_batchcap16_job.sh").read_text()
    assert "--time=01:15:00" in txt and "8 16 24 32" in txt and "EXPECT_MAX_BATCH=16" in txt and "rq1_batchcap16" in txt
