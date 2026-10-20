"""CPU closed-loop harness for the C2 controller-mechanism study.

Runs the REAL EdgeSpecSimulator round code (clients, controller, causal coordinator, batcher + scheduler, bandwidth allocator, virtual queues,
lifecycle, early stop) with stand-ins only for the models:
  * target verifier: Bernoulli acceptance coupled across policies (the u-draw for token position p of (client, sample) is fixed by the seed, so two policies
    that propose the same tokens see the same acceptance outcomes) and a PHYSICAL batch service time wall = a + b * processed_tokens * (1 + noise), with
    (a, b) fitted offline from the C1 traces (results/c2/verifier_cost_model_offline.json);
  * draft worker: instant (virtual draft time comes from the isolated profile, as in the pilots).
Nothing here touches a GPU.  Ablations are subclasses used ONLY by the harness; production code is unchanged."""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import math
import random
import sys
import types
from dataclasses import dataclass, field

if "edge_specsim.target_client" not in sys.modules:
    _tc = types.ModuleType("edge_specsim.target_client")
    _tc.VLLMCandidateVerifier = object
    sys.modules["edge_specsim.target_client"] = _tc
if "edge_specsim.draft_worker" not in sys.modules:
    _dw = types.ModuleType("edge_specsim.draft_worker")

    @dataclass
    class DraftRequest:
        client_id: int
        context_token_ids: list
        gamma: int
        temperature: float

    _dw.DraftRequest, _dw.DraftWorker, _dw.DraftKVState = DraftRequest, object, object
    sys.modules["edge_specsim.draft_worker"] = _dw

from edge_specsim import simulator as simmod  # noqa: E402
from edge_specsim.controller import OnlineController  # noqa: E402
from edge_specsim.expectations import expected_useful_tokens_from_profile  # noqa: E402
from edge_specsim.models import DraftResult, PromptSample, VerificationResult, stamp_batch_timing  # noqa: E402
from edge_specsim.verification_batcher import VerificationBatcher  # noqa: E402
from edge_specsim.verification_queue import VerificationQueue  # noqa: E402
from edge_specsim.virtual_causality import EventLedger, VirtualCoordinator  # noqa: E402

EdgeSpecSimulator = simmod.EdgeSpecSimulator
WALL_A_MS, WALL_B_MS_PER_TOKEN = 52.0, 0.19            # fitted offline: results/c2/verifier_cost_model_offline.json (M2)
PROMPT_LEN = 112                                      # observed: mean processed tokens per request 162 = prompt + ~48 generated


def _u(*key) -> float:
    h = hashlib.blake2b(repr(key).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2.0 ** 64


class StubWorker:
    device = "cpu"

    def __init__(self, delay_s=0.0, seed=0):
        self.delay_s, self.rng = delay_s, random.Random(seed)

    async def reset_client(self, cid): ...
    async def confirmed_length(self, cid): return None
    async def commit(self, cid, tokens): ...

    async def generate(self, request):
        if self.delay_s:
            await asyncio.sleep(self.rng.random() * self.delay_s)
        g = request.gamma
        return DraftResult(request.client_id, list(range(g)), [None] * g, "", 1.0, 0, distribution_payload="delta_proposal")


class StubTarget:
    vocab_size = 151936
    effective_max_logprobs = 1

    def __init__(self):
        self._pref = {}

    prompt_lens = None

    def build_context_token_ids(self, prompt, history):
        key = int(hashlib.blake2b(prompt.encode(), digest_size=4).hexdigest(), 16)
        cid = int(prompt.split("-")[0][1:])
        plen = PROMPT_LEN if self.prompt_lens is None else int(self.prompt_lens[cid])
        return [key] + [1] * (plen - 1) + list(history)

    def set_confirmed_prefix(self, cid, toks): self._pref[cid] = list(toks)
    def confirmed_prefix_hash(self, cid): return hash(tuple(self._pref.get(cid, [])))


class PhysicalTarget:
    VERIFIER_SEMANTICS = "V1"

    def __init__(self, alphas, seed, a=WALL_A_MS, b=WALL_B_MS_PER_TOKEN, noise=0.08, prompt_lens=None):
        self.alphas, self.seed, self.a, self.b, self.noise = alphas, seed, a, b, noise
        self.prompt_lens = prompt_lens
        self.calls = []

    def verify_batch_sync(self, payload):
        results, processed, ids = [], 0, []
        prompt_lens = self.prompt_lens
        for ctx, draft, remaining in payload:
            cid, g = draft.client_id, len(draft.token_ids)
            pos0 = len(ctx) - (PROMPT_LEN if prompt_lens is None else int(prompt_lens[cid]))
            processed += len(ctx) + g
            ids.append((cid, len(ctx)))
            acc = 0
            for k in range(g):
                if _u(self.seed, cid, ctx[0], pos0 + k) < self.alphas[cid]:
                    acc += 1
                else:
                    break
            inspected = min(g, acc + 1)
            committed = list(range(min(acc + 1, max(1, remaining))))
            results.append(VerificationResult(accepted_length=acc, committed_token_ids=committed, committed_text="x", target_latency_ms=0.0,
                                              acceptance_probabilities=[float(self.alphas[cid])] * inspected))
        z = (_u(self.seed, "wall", tuple(ids)) - 0.5) * 2.0
        wall = (self.a + self.b * processed) * (1.0 + self.noise * z)
        self.calls.append(len(payload))
        stamp_batch_timing(results, wall)
        return results


def ablated_controller_class():
    class Ablated(OnlineController):
        """Harness-only: switches mechanisms off one at a time; with no switch it must equal production (tested)."""

        def __init__(self, *a, ablate=frozenset(), theta_override=None, rescale=None, **k):
            super().__init__(*a, **k)
            self.ablate = frozenset(ablate)
            self.theta_override = theta_override
            self.rescale = dict(rescale or {})            # unit-sensitivity tests: multiply a term by a constant
            self.decision_log = []

        def set_verifier_theta_f_signal(self, v):
            super().set_verifier_theta_f_signal(self.theta_override if self.theta_override is not None else v)

        def score_with(self, client, gamma, profile, ablate=None, rescale=None, qi_override=None, qs_override=None):
            ab = self.ablate if ablate is None else ablate
            rs = self.rescale if rescale is None else rescale
            z = 0.0 if ("z" in ab or not self.use_virtual_queues) else client.z_queue
            qs = 0.0 if ("qs" in ab or not self.use_virtual_queues) else (self.server_queue if qs_override is None else qs_override)
            qi = 0.0 if ("qi" in ab or not self.use_virtual_queues) else (client.device_queue if qi_override is None else qi_override)
            phi = expected_useful_tokens_from_profile(profile) if gamma > 0 else 1.0
            return ((self.V + z) * phi
                    - rs.get("qs", 1.0) * qs * self.current_verifier_theta_f_ms_per_token * (gamma + 1)
                    - rs.get("qi", 1.0) * qi * gamma * self._capacity_dpp_device_cost_per_token_ms(client))

        def capacity_dpp_score(self, client, gamma, acceptance_profile):
            return self.score_with(client, gamma, acceptance_profile)

        def argmax_under(self, client, ablate, **kw):
            best, bs = 0, self.score_with(client, 0, [], ablate, **kw)
            for g in self.gamma_choices:
                if g == 0:
                    continue
                s = self.score_with(client, g, client.positional_acceptance_profile(g, True), ablate, **kw)
                if s > bs + 1e-12:
                    best, bs = g, s
            return best

        def observe_device_service(self, client_id, edge_finish_ms, gamma, tau_d_ms, kappa_over_r_ms):
            self._qi_last = getattr(self, "_qi_last", {})
            self._qi_last[client_id] = edge_finish_ms
            return super().observe_device_service(client_id, edge_finish_ms, gamma, tau_d_ms, kappa_over_r_ms)

        def observe_server_batch(self, batch_id, verify_finish_ms, theta_f_ms_per_token, verifier_token_cost):
            self._qs_last = verify_finish_ms
            return super().observe_server_batch(batch_id, verify_finish_ms, theta_f_ms_per_token, verifier_token_cost)

        def _ages(self, client):
            qi_age = max(0.0, (client.virtual_time_ms - getattr(self, "_qi_last", {}).get(client.client_id, client.virtual_time_ms)) / self.slot_ms)
            qs_age = max(0.0, (client.virtual_time_ms - getattr(self, "_qs_last", client.virtual_time_ms)) / self.slot_ms)
            return qi_age, qs_age

        def choose_gamma(self, client, total_rounds):
            client.refresh_learning_stats(total_rounds)
            if self.policy == "capacity_dpp":
                self.decision_log.append({
                    "client": client.client_id, "t": client.virtual_time_ms, "z": client.z_queue, "qi": client.device_queue, "qs": self.server_queue,
                    "theta": self.current_verifier_theta_f_ms_per_token, "alpha_ucb": client.alpha_ucb,
                    "full": self.argmax_under(client, frozenset()), "no_z": self.argmax_under(client, frozenset({"z"})),
                    "no_qi": self.argmax_under(client, frozenset({"qi"})), "no_qs": self.argmax_under(client, frozenset({"qs"})),
                    "benefit_only": self.argmax_under(client, frozenset({"z", "qi", "qs"})),
                    "qs_x0.1": self.argmax_under(client, frozenset(), rescale={"qs": 0.1}), "qs_x10": self.argmax_under(client, frozenset(), rescale={"qs": 10.0}),
                    "qi_x0.1": self.argmax_under(client, frozenset(), rescale={"qi": 0.1}), "qi_x10": self.argmax_under(client, frozenset(), rescale={"qi": 10.0}),
                    "qi_age_slots": self._ages(client)[0], "qs_age_slots": self._ages(client)[1],
                    "stale_drained": self.argmax_under(
                        client, frozenset(),
                        qi_override=max(0.0, client.device_queue - self.slot_ms * int(self._ages(client)[0])),
                        qs_override=max(0.0, self.server_queue - self.slot_ms * int(self._ages(client)[1])))})
            return super().choose_gamma(client, total_rounds)

    return Ablated


@dataclass
class Spec:
    n: int
    policy: str = "capacity_dpp"                 # capacity_dpp | fixed_gamma
    fixed_gamma: int = 2
    scheduler: str = "capacity_knapsack"
    max_batch: int = 16
    token_budget: int = 100
    ablate: frozenset = frozenset()
    theta_f: float | None = None                  # None = production (0 in fixed-budget mode)
    rescale: dict = field(default_factory=dict)
    x_req: float = 3.0
    V: float = 50.0
    alphas: list | None = None
    hetero_hw: bool = False
    seed: int = 1
    window_s: float = 120.0
    warmup_s: float = 30.0
    margin_s: float = 10.0
    delay_s: float = 0.0005
    log_batches: bool = True
    causal: bool = True                           # False = historical real-order batching/state (control for the future-information test)
    wireless: dict | None = None
    batch_wait_ms: float = 150.0
    prompt_lens: list | None = None               # per-client prompt length (tokens); default PROMPT_LEN for everybody


HOMOG_CLIENTS = {"rtt_ms_range": [50.0, 50.0], "uplink_mbps_range": [20.0, 20.0], "downlink_mbps_range": [40.0, 40.0],
                 "draft_tokens_per_second_range": [40.0, 40.0], "draft_fixed_latency_ms_range": [10.0, 10.0],
                 "uplink_base_snr_db_range": [14.0, 14.0], "downlink_base_snr_db_range": [16.0, 16.0],
                 "uplink_snr_jitter_db_range": [2.0, 2.0], "downlink_snr_jitter_db_range": [2.0, 2.0], "snr_period_ms_range": [1000.0, 1000.0],
                 "packet_loss_range": [0.0, 0.0], "draft_speed_multiplier_range": [1.0, 1.0], "snr_block_duration_ms_range": [100.0, 100.0],
                 "shared_wireless": {"uplink_allocator": "proportional", "downlink_allocator": "proportional",
                                     "total_uplink_bandwidth_mhz": 40.0, "total_downlink_bandwidth_mhz": 80.0}}
HETERO_CLIENTS = dict(HOMOG_CLIENTS, rtt_ms_range=[20.0, 100.0], uplink_mbps_range=[5.0, 50.0], draft_tokens_per_second_range=[20.0, 60.0],
                      draft_fixed_latency_ms_range=[5.0, 20.0], uplink_base_snr_db_range=[8.0, 18.0])


def build(spec: Spec):
    Ablated = ablated_controller_class()
    n = spec.n
    alphas = spec.alphas or [0.92] * n          # observed tokens/round 2.77 at gamma=2 => per-token acceptance ~0.92
    sim = object.__new__(EdgeSpecSimulator)
    clients_cfg = dict(HETERO_CLIENTS if spec.hetero_hw else HOMOG_CLIENTS)
    if spec.wireless:
        clients_cfg["shared_wireless"] = dict(clients_cfg["shared_wireless"], **spec.wireless)
    sim.cfg = {"simulation": {"seed": spec.seed, "num_clients": n, "nested_population": True, "population_n_max": n, "questions_per_client": 64,
                              "population_seed": 20261020, "draft_profile": {"overhead_ms": 0.0, "per_token_ms": 22.2}},
               "clients": clients_cfg, "models": {"draft": "d", "target": "t"}, "dataset": {}}
    sim.learning_cfg = {}
    sim._nested_population, sim._nested_qpc, sim._nested_n_max = True, 64, n
    sim.population_fingerprint = ""
    sim.workers = [StubWorker(spec.delay_s, spec.seed)]
    sim.samples = [PromptSample(f"s{i}", "ds", f"c{i // 64}-s{i % 64}", 96) for i in range(n * 64)]
    sim.clients = sim._make_clients(sim.samples, num_clients=n)
    sim.target = StubTarget()
    sim.target.prompt_lens = spec.prompt_lens
    sim.records, sim.detailed_log, sim.total_rounds = [], False, 0
    sim.wall_clock_start = 0.0
    sim.virtual_system_time_ms = sim.virtual_target_available_at_ms = 0.0
    sim.draft_execution_mode = "isolated_profile"
    sim.sampling_temperature = 0.0
    sim.measurement_start_ms = spec.warmup_s * 1000.0
    sim.measurement_end_ms = (spec.warmup_s + spec.window_s) * 1000.0
    sim.early_stop_margin_ms = spec.margin_s * 1000.0
    sim.batch_wait_ms = spec.batch_wait_ms
    sim.causal_virtual_time = spec.causal
    sim.coordinator = VirtualCoordinator(sim.batch_wait_ms, EventLedger()) if spec.causal else None
    sim.failure_reports, sim.memory_events = [], {}
    sim.controller = Ablated(
        gamma_choices=[0, 1, 2, 3, 4], V=spec.V, min_tps=spec.x_req, server_price_scale=0.02, policy=spec.policy, fixed_gamma=spec.fixed_gamma,
        seed=spec.seed, slot_ms=100.0, min_exploration_rounds=0, exploration_epsilon=0.0, proposal_vocab_size=151936,
        proposal_distribution_payload="delta_proposal", ablate=spec.ablate, theta_override=spec.theta_f, rescale=spec.rescale)
    target = PhysicalTarget(alphas, spec.seed, prompt_lens=spec.prompt_lens)
    sim.verifier = VerificationBatcher(
        target, max_batch_size=spec.max_batch, batch_wait_ms=sim.batch_wait_ms, verify_token_budget=spec.token_budget,
        scheduler_name=spec.scheduler, service_time_model="batch_wall", coordinator=sim.coordinator, causal=spec.causal,
        on_batch_complete=sim._on_batch_complete if spec.causal else None)
    sim.batch_log = []
    if spec.log_batches:
        _instrument_queue(sim, spec)
    if spec.causal:
        for c in sim.clients:
            sim.coordinator.register_client(c.client_id, c.rtt_ms)
    sim._write_results = lambda: None
    sim.spec = spec
    return sim


def _instrument_queue(sim, spec):
    q = sim.verifier.server_queue
    orig = q.pop_batch
    alts = ("fcfs", "capacity_knapsack", "max_throughput")

    def wrapped(**kw):
        eligible = q.eligible_requests(kw["batch_wait_ms"])
        rec = {"n_eligible": len(eligible), "eligible_cost": sum(r.verifier_token_cost for r in eligible),
               "cap": kw["max_batch_size"], "budget": kw["verify_token_budget"], "eligible_ids": [r.client_id for r in eligible],
               "z_by_client": {r.client_id: r.weight for r in eligible}}
        for name in alts:
            alt = VerificationQueue()
            alt.waiting = [dataclasses.replace(r) for r in eligible]
            kw2 = dict(kw, scheduler_name=name)
            rec[name] = [r.client_id for r in alt.pop_batch(**kw2)]
        batch = orig(**kw)
        rec["chosen_scheduler"] = kw["scheduler_name"]
        rec["chosen"] = [r.client_id for r in batch]
        sim.batch_log.append(rec)
        return batch

    q.pop_batch = wrapped


def run_sim(spec: Spec):
    sim = build(spec)
    asyncio.run(asyncio.wait_for(sim.run(), 300))
    return sim


def summarize(sim) -> dict:
    import pandas as pd
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "scripts" / "c1"))
    import analyze_rq1_pilot as an

    df = pd.DataFrame(sim.records)
    n = sim.spec.n
    m = an.run_metrics(df, n)
    start, end = sim.measurement_start_ms, sim.measurement_end_ms
    win = df[(df["client_receive_ms"] >= start) & (df["client_receive_ms"] <= end)]
    g = win["gamma"].value_counts(normalize=True).sort_index().to_dict()
    return {"x_min": m["x_min"], "x_mean": m["x_mean"], "Y": m["Y"], "fair": m["x_min_over_mean"], "util": m["verifier_utilisation"],
            "batch_size_mean": m["batch_size_mean"], "gamma_mean": float(win["gamma"].mean()), "gamma_dist": {int(k): float(v) for k, v in g.items()},
            "z_mean": float(win["z_queue"].mean()), "z_max": float(win["z_queue"].max()), "qi_mean": float(win["device_queue"].mean()),
            "qi_max": float(win["device_queue"].max()), "qs_mean": float(win["server_queue_before"].mean()), "qs_max": float(win["server_queue_before"].max()),
            "upload_ms_mean": float(win["upload_ms"].mean()), "upload_ms_max": float(win["upload_ms"].max()),
            "round_ms_mean": m["round_ms_mean"], "queue_wait_ms_mean": m["queue_wait_ms_mean"],
            "ledger_violations": (len(sim.coordinator.ledger.check_causality()) if sim.coordinator is not None else None), "per_client_rates": m["rates"], "invalid": m["invalid"]}
