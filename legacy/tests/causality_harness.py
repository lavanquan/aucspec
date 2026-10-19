"""CPU harness for the virtual-time causality tests (no torch, no vLLM).

Clients behave like `EdgeSpecSimulator._run_round`: decide at virtual time t, "draft" for a VIRTUAL
`draft_ms` while really sleeping `real_delay_s` (this is what the tests permute), submit with a virtual
arrival, wait for the batched result and continue at the virtual receive time.  The same coordinator,
batcher and controller classes as the simulator are used.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from edge_specsim.controller import OnlineController
from edge_specsim.models import DraftResult, VerificationResult, stamp_batch_timing
from edge_specsim.verification_batcher import VerificationBatcher
from edge_specsim.virtual_causality import EventLedger, VirtualCoordinator
from edge_specsim.verification_queue import VerificationQueue, VerificationRequest


@dataclass
class Spec:
    cid: int
    rtt: float
    draft_ms: list[float]
    real_delay_s: list[float] = field(default_factory=list)


class MockTarget:
    VERIFIER_SEMANTICS = "V1"

    def __init__(self, wall_fn):
        self.wall_fn = wall_fn
        self.calls = []

    def verify_batch_sync(self, payload):
        self.calls.append(len(payload))
        results = [VerificationResult(0, [1], "x", 0.0) for _ in payload]
        stamp_batch_timing(results, self.wall_fn(len(payload)))
        return results


def _draft(cid, gamma=1):
    return DraftResult(cid, list(range(gamma)), [None] * gamma, "", 1.0, 0)


def default_wall(b):
    return 20.0 + 5.0 * b


async def run_scenario(specs, *, causal, wait_ms=50.0, max_batch=8, wall_fn=default_wall, rounds=None,
                       pre_decision_real_delay_s=None):
    ledger = EventLedger()
    coord = VirtualCoordinator(wait_ms, ledger)
    controller = OnlineController(gamma_choices=[0, 1, 2], V=50.0, min_tps=3.0, server_price_scale=1.0,
                                  policy="fixed_gamma", fixed_gamma=1)
    target = MockTarget(wall_fn)
    batcher = VerificationBatcher(
        target, max_batch_size=max_batch, batch_wait_ms=wait_ms, verify_token_budget=10_000,
        service_time_model="batch_wall", coordinator=coord, causal=causal,
        on_batch_complete=(
            (lambda m: controller.observe_server_batch(m.batch_id, m.verify_finish_ms, m.theta_f_ms_per_token,
                                                        m.batch_token_cost)) if causal else None),
    )
    for sp in specs:
        coord.register_client(sp.cid, sp.rtt)
    await batcher.start()
    batches: dict[int, dict] = {}
    receives: dict[int, list[float]] = {sp.cid: [] for sp in specs}
    decisions: list[dict] = []

    async def client(sp: Spec):
        t = 0.0
        n_rounds = rounds if rounds is not None else len(sp.draft_ms)
        try:
            for k in range(n_rounds):
                if causal:
                    await coord.wait_decision(sp.cid, t)
                    visible_ts, version = controller.load_server_state_view(t)
                else:
                    coord.begin_round(sp.cid, t)
                    if pre_decision_real_delay_s:
                        await asyncio.sleep(pre_decision_real_delay_s.get(sp.cid, 0.0))
                    visible_ts, version = controller.server_state_max_finish_ms, coord.state_version
                q_seen = controller.server_queue
                coord.record_decision(sp.cid, t, visible_ts, version, gamma=1)
                decisions.append({"cid": sp.cid, "round": k, "t": t, "visible_ts": visible_ts, "queue": q_seen})
                if sp.real_delay_s:
                    await asyncio.sleep(sp.real_delay_s[k % len(sp.real_delay_s)])
                arrival = t + sp.draft_ms[k] + sp.rtt / 2.0
                meta, _ = await batcher.submit(
                    client_id=sp.cid, context_token_ids=[0] * 4, draft=_draft(sp.cid), remaining_tokens=50,
                    server_arrival_ms=arrival)
                if not causal:
                    controller.observe_server_batch(meta.batch_id, meta.verify_finish_ms,
                                                    meta.theta_f_ms_per_token, meta.batch_token_cost)
                batches.setdefault(meta.batch_id, {"start": meta.batch_start_ms, "finish": meta.verify_finish_ms,
                                                   "members": []})["members"].append((sp.cid, k))
                t = meta.verify_finish_ms + sp.rtt / 2.0
                receives[sp.cid].append(t)
        finally:
            coord.retire_client(sp.cid)

    try:
        await asyncio.wait_for(asyncio.gather(*(client(sp) for sp in specs)), timeout=30)
    finally:
        await batcher.close()
    for b in batches.values():
        b["members"] = sorted(b["members"])
    return {
        "batches": [batches[i] for i in sorted(batches)],
        "receives": receives,
        "violations": ledger.check_causality(),
        "ledger": ledger,
        "decisions": decisions,
        "controller": controller,
        "target_calls": target.calls,
    }


def reference_batches(specs, *, wait_ms=50.0, max_batch=8, wall_fn=default_wall, rounds=None):
    """Sequential virtual-time reference (no asyncio): the answer causal mode must reproduce."""
    t = {sp.cid: 0.0 for sp in specs}
    k = {sp.cid: 0 for sp in specs}
    n_rounds = {sp.cid: (rounds if rounds is not None else len(sp.draft_ms)) for sp in specs}
    by = {sp.cid: sp for sp in specs}
    pending = {sp.cid: sp.draft_ms[0] + sp.rtt / 2.0 for sp in specs if n_rounds[sp.cid] > 0}
    avail = 0.0
    out = []
    while pending:
        order = sorted(pending.items(), key=lambda kv: (kv[1], kv[0]))
        head_arrival = order[0][1]
        cutoff = head_arrival + wait_ms
        eligible = [(cid, a) for cid, a in order if a <= cutoff][:max_batch]
        start = max(cutoff, avail)
        finish = start + wall_fn(len(eligible))
        avail = finish
        out.append({"start": start, "finish": finish, "members": sorted((cid, k[cid]) for cid, _ in eligible)})
        for cid, _ in eligible:
            del pending[cid]
            t[cid] = finish + by[cid].rtt / 2.0
            k[cid] += 1
            if k[cid] < n_rounds[cid]:
                pending[cid] = t[cid] + by[cid].draft_ms[k[cid]] + by[cid].rtt / 2.0
    return out
