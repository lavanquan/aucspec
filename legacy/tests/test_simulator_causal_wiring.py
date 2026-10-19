"""Integration test of the C1 simulator wiring with stubbed target/draft workers (needs torch to import the module;
no GPU, no vLLM weights).  Exercises the real EdgeSpecSimulator._make_clients / _run_client / _run_round code."""
import asyncio
import random
import sys
import types
from collections import deque
from dataclasses import dataclass

import pytest

# The simulator module imports vLLM (target_client) and torch/transformers (draft_worker) at import time. The wiring
# under test does not need them, so lightweight stand-ins are injected BEFORE the import (CPU-only, any machine).
_STUBBED = []
if "edge_specsim.target_client" not in sys.modules:
    _STUBBED.append("edge_specsim.target_client")
    _tc = types.ModuleType("edge_specsim.target_client")
    _tc.VLLMCandidateVerifier = object
    sys.modules["edge_specsim.target_client"] = _tc
if "edge_specsim.draft_worker" not in sys.modules:
    _STUBBED.append("edge_specsim.draft_worker")
    _dw = types.ModuleType("edge_specsim.draft_worker")

    @dataclass
    class DraftRequest:
        client_id: int
        context_token_ids: list
        gamma: int
        temperature: float

    _dw.DraftRequest, _dw.DraftWorker, _dw.DraftKVState = DraftRequest, object, object
    sys.modules["edge_specsim.draft_worker"] = _dw

from causality_harness import MockTarget, default_wall  # noqa: E402
from edge_specsim import simulator as simmod  # noqa: E402
from edge_specsim.controller import OnlineController  # noqa: E402
from edge_specsim.models import DraftResult, PromptSample, VerificationResult, stamp_batch_timing  # noqa: E402
from edge_specsim.verification_batcher import VerificationBatcher  # noqa: E402
from edge_specsim.virtual_causality import EventLedger, VirtualCoordinator  # noqa: E402

EdgeSpecSimulator = simmod.EdgeSpecSimulator


def teardown_module(module):
    """Do not leak the stand-ins into other test modules of the same session."""
    for name in _STUBBED + ["edge_specsim.simulator"]:
        sys.modules.pop(name, None)

CLIENTS_CFG = {
    "rtt_ms_range": [15.0, 100.0], "uplink_mbps_range": [5.0, 50.0],
    "draft_tokens_per_second_range": [20.0, 60.0], "draft_fixed_latency_ms_range": [5.0, 20.0],
}


class StubWorker:
    device = "cpu"

    def __init__(self, delay_rng):
        self.delay_rng = delay_rng

    async def reset_client(self, cid): ...
    async def confirmed_length(self, cid): return None
    async def commit(self, cid, tokens): ...

    async def generate(self, request):
        await asyncio.sleep(self.delay_rng.random() * 0.01)     # real completion order jitter
        g = request.gamma
        return DraftResult(request.client_id, list(range(g)), [None] * g, "", 5.0 + 3 * g, 0,
                           distribution_payload="delta_proposal")


class StubTarget:
    vocab_size = 1000

    def __init__(self):
        self._pref = {}

    def build_context_token_ids(self, prompt, history):
        return [1, 2, 3] + list(history)

    def set_confirmed_prefix(self, cid, toks): self._pref[cid] = list(toks)
    def confirmed_prefix_hash(self, cid): return hash(tuple(self._pref.get(cid, [])))


class TwoTokenTarget(MockTarget):
    def verify_batch_sync(self, payload):
        self.calls.append(len(payload))
        results = [VerificationResult(accepted_length=0, committed_token_ids=[7, 8], committed_text="x",
                                      target_latency_ms=0.0) for _ in payload]
        stamp_batch_timing(results, self.wall_fn(len(payload)))
        return results


def build_sim(n, *, run_seed, causal, population_seed=7, qpc=3, max_new=9, delay_seed=0, policy="capacity_dpp",
              early_stop_margin_s=None, draft_mode="simulation"):
    sim = object.__new__(EdgeSpecSimulator)
    sim.cfg = {"simulation": {"seed": run_seed, "num_clients": n, "nested_population": True,
                              "population_n_max": 16, "questions_per_client": qpc},
               "clients": dict(CLIENTS_CFG), "models": {"draft": "d", "target": "t"}, "dataset": {}}
    if population_seed is not None:
        sim.cfg["simulation"]["population_seed"] = population_seed
    sim.learning_cfg = {}
    sim._nested_population, sim._nested_qpc, sim._nested_n_max = True, qpc, 16
    sim.population_fingerprint = ""
    sim.workers = [StubWorker(random.Random(delay_seed))]
    sim.samples = [PromptSample(f"s{i}", "ds", "p", max_new) for i in range(16 * qpc)]
    sim.clients = sim._make_clients(sim.samples, num_clients=n)
    sim.target = StubTarget()
    sim.records = []
    sim.detailed_log = False
    sim.total_rounds = 0
    sim.wall_clock_start = 0.0
    sim.virtual_system_time_ms = 0.0
    sim.virtual_target_available_at_ms = 0.0
    sim.draft_execution_mode = draft_mode
    sim.early_stop_margin_ms = None if early_stop_margin_s is None else early_stop_margin_s * 1000.0
    if draft_mode == "isolated_profile":
        sim.cfg["simulation"]["draft_profile"] = {"overhead_ms": 0.0, "per_token_ms": 22.0}
    sim.sampling_temperature = 0.0
    sim.measurement_start_ms, sim.measurement_end_ms = 500.0, 5000.0
    sim.batch_wait_ms = 40.0
    sim.causal_virtual_time = causal
    sim.coordinator = VirtualCoordinator(sim.batch_wait_ms, EventLedger())
    sim.controller = OnlineController(
        gamma_choices=[0, 1, 2, 3], V=50.0, min_tps=3.0, server_price_scale=1.0, policy=policy, fixed_gamma=2,
        seed=run_seed, proposal_vocab_size=1000, proposal_distribution_payload="delta_proposal")
    sim.verifier = VerificationBatcher(
        TwoTokenTarget(default_wall), max_batch_size=8, batch_wait_ms=sim.batch_wait_ms, verify_token_budget=100,
        service_time_model="batch_wall", coordinator=sim.coordinator, causal=causal,
        on_batch_complete=sim._on_batch_complete if causal else None)
    for c in sim.clients:
        sim.coordinator.register_client(c.client_id, c.rtt_ms)
    sim._write_results = lambda: None
    return sim


def run(sim):
    asyncio.run(asyncio.wait_for(sim.run(), 60))
    return sim


def vkey(rec):
    return (rec["client_id"], rec["round_id"], rec["sample_id"], rec["verification_batch_id"],
            round(rec["client_receive_ms"], 6), round(rec["verify_finish_ms"], 6), rec["gamma"])


def test_population_is_fixed_across_run_seeds_and_nested_in_n():
    a = build_sim(4, run_seed=1, causal=False).clients
    b = build_sim(4, run_seed=2, causal=False).clients
    c = build_sim(8, run_seed=3, causal=False).clients
    fields = ("rtt_ms", "uplink_mbps", "downlink_mbps", "draft_tokens_per_second", "draft_fixed_latency_ms",
              "uplink_base_snr_db", "snr_phase_rad")
    for i in range(4):
        for f in fields:
            assert getattr(a[i], f) == getattr(b[i], f) == getattr(c[i], f), (i, f)
    assert [list(x.prompt_queue) for x in a] == [list(x.prompt_queue) for x in c[:4]]


def test_legacy_mode_still_redraws_the_population_per_seed():
    a = build_sim(4, run_seed=1, causal=False, population_seed=None).clients
    b = build_sim(4, run_seed=2, causal=False, population_seed=None).clients
    assert a[0].rtt_ms != b[0].rtt_ms        # the documented legacy behaviour (historical results stay readable)


def test_causal_run_is_causally_consistent_and_deterministic_under_permuted_real_order():
    runs = [run(build_sim(4, run_seed=1, causal=True, delay_seed=s)) for s in (0, 1, 2)]
    keys = [sorted(vkey(r) for r in s.records) for s in runs]
    assert keys[0] == keys[1] == keys[2]
    for s in runs:
        assert s.coordinator.ledger.check_causality() == []
        assert all(r["decision_visible_update_ms"] <= r["decision_virtual_ms"] + 1e-9 for r in s.records)
        assert all(r["causal_virtual_time"] for r in s.records)


def test_lifecycle_columns_mark_the_last_round_of_each_client_complete():
    s = run(build_sim(3, run_seed=1, causal=True, qpc=2, max_new=6))
    for cid in range(3):
        recs = sorted((r for r in s.records if r["client_id"] == cid), key=lambda r: r["client_receive_ms"])
        assert recs[0]["client_samples_assigned"] == 2
        assert [r["client_work_complete"] for r in recs].count(True) == 1 and recs[-1]["client_work_complete"]
        assert sum(r["sample_finished"] for r in recs) == 2


def test_legacy_real_order_mode_with_ledger_exposes_d1_in_the_real_simulator_code():
    found = 0
    for seed in range(4):
        s = run(build_sim(6, run_seed=1, causal=False, delay_seed=seed))
        found += sum(v.startswith("D1") for v in s.coordinator.ledger.check_causality())
    assert found > 0


def test_early_stop_keeps_window_behaviour_and_leaves_unfinished_work_active():
    full = run(build_sim(3, run_seed=1, causal=True, qpc=6, max_new=24))
    es = run(build_sim(3, run_seed=1, causal=True, qpc=6, max_new=24, early_stop_margin_s=0.5))
    end = es.measurement_end_ms
    inside = lambda s: sorted(vkey(r) for r in s.records if r["client_receive_ms"] <= end)
    assert inside(full) == inside(es)                       # identical inside the window
    assert len(es.records) < len(full.records)
    assert all(r["client_receive_ms"] <= end + 500.0 + 3000.0 for r in es.records)
    assert not any(r["client_work_complete"] for r in es.records)       # stopped, not exhausted
    from edge_specsim.workload_lifecycle import assess_workload
    rows = {}
    for r in es.records:
        rows.setdefault(r["client_id"], []).append(r)
    a = assess_workload(rows, admitted_client_ids=range(3), measurement_start_ms=es.measurement_start_ms,
                        measurement_end_ms=end)
    assert not a.invalid


def test_isolated_profile_draft_latency_is_independent_of_n_and_of_real_contention():
    a = run(build_sim(2, run_seed=1, causal=True, draft_mode="isolated_profile", delay_seed=0))
    b = run(build_sim(6, run_seed=1, causal=True, draft_mode="isolated_profile", delay_seed=5))
    la = {(r["client_id"], r["gamma"]) for r in a.records if r["gamma"] > 0 and r["client_id"] < 2}
    lb = {(r["client_id"], r["gamma"]) for r in b.records if r["gamma"] > 0 and r["client_id"] < 2}
    assert la == lb
    for s in (a, b):
        for r in s.records:
            if r["gamma"] > 0:
                c = s.clients[r["client_id"]]
                assert r["draft_latency_ms"] == pytest.approx(c.draft_fixed_latency_ms + c.draft_speed_multiplier * 22.0 * r["gamma"])


def test_fixed_gamma_pilot_policy_runs_through_the_real_round_code():
    s = run(build_sim(4, run_seed=1, causal=True, policy="fixed_gamma", early_stop_margin_s=0.5, qpc=4, max_new=24))
    assert s.coordinator.ledger.check_causality() == []
    assert {r["selected_gamma"] for r in s.records} == {2}


def test_pilot_config_gives_identical_static_hardware_across_clients_n_and_run_seeds():
    import yaml
    from pathlib import Path
    pilot = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "rq1_pilot_fixed_gamma2_tritonfa33.yaml").read_text())
    saved = dict(CLIENTS_CFG)
    CLIENTS_CFG.clear()
    CLIENTS_CFG.update(pilot["clients"])
    try:
        sims = [build_sim(n, run_seed=rs, causal=False, population_seed=pilot["simulation"]["population_seed"])
                for n, rs in ((2, 1), (8, 2), (16, 3))]
    finally:
        CLIENTS_CFG.clear()
        CLIENTS_CFG.update(saved)
    hw = ("rtt_ms", "uplink_mbps", "downlink_mbps", "packet_loss", "draft_tokens_per_second", "draft_fixed_latency_ms",
          "draft_speed_multiplier", "uplink_base_snr_db", "downlink_base_snr_db", "uplink_snr_jitter_db",
          "downlink_snr_jitter_db", "snr_period_ms", "snr_block_duration_ms", "wireless_weight",
          "diagnostic_priority_multiplier", "channel_model", "draft_device_class")
    ref = {f: getattr(sims[0].clients[0], f) for f in hw}
    for sim in sims:
        for c in sim.clients:
            for f in hw:
                assert getattr(c, f) == ref[f], (f, c.client_id)
    assert ref["draft_speed_multiplier"] == 1.0
    # explicitly NOT homogeneous: channel phase (stochastic fading) and the prompt block
    assert len({c.snr_phase_rad for c in sims[2].clients}) == 16
    assert len({tuple(s.sample_id for s in c.prompt_queue) for c in sims[2].clients}) == 16
