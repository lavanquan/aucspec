"""Causal batching: a request whose VIRTUAL arrival falls inside the batch
window must join the batch even if it reaches the batcher later in wall time
(its client was still running a real draft step)."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types

import pytest

if importlib.util.find_spec("torch") is None and "edge_specsim.target_client" not in sys.modules:
    _stub = types.ModuleType("edge_specsim.target_client")
    _stub.VLLMCandidateVerifier = object
    sys.modules["edge_specsim.target_client"] = _stub

from edge_specsim.models import DraftResult, VerificationResult  # noqa: E402
from edge_specsim.verification_batcher import VerificationBatcher  # noqa: E402


class _Target:
    def verify_batch_sync(self, payload):
        return [
            VerificationResult(accepted_length=0, committed_token_ids=[1], committed_text="",
                               target_latency_ms=10.0 / len(payload))
            for _ in payload
        ]


def _draft(cid):
    return DraftResult(client_id=cid, token_ids=[5, 6], draft_logprobs=[], text="",
                       latency_ms=0.0, worker_id=0)


def _run(causal: bool, late_arrival_ms: float):
    async def go():
        batcher = VerificationBatcher(_Target(), max_batch_size=8, batch_wait_ms=150.0,
                                      verify_token_budget=100, service_time_model="batch_wall")
        # client 1 is "drafting" from virtual t=50 until it submits
        state = {"drafting": True}
        if causal:
            batcher.arrival_lower_bound = lambda: 50.0 if state["drafting"] else float("inf")
        await batcher.start()

        async def early():
            return await batcher.submit(0, [1, 2], _draft(0), 10, server_arrival_ms=0.0)

        async def late():
            await asyncio.sleep(0.05)  # wall-clock draft time
            state["drafting"] = False
            return await batcher.submit(1, [1, 2], _draft(1), 10,
                                        server_arrival_ms=late_arrival_ms)

        (m0, _), (m1, _) = await asyncio.gather(early(), late())
        await batcher.close()
        return m0, m1

    return asyncio.run(go())


def test_wall_clock_batching_misses_virtually_earlier_request():
    m0, m1 = _run(causal=False, late_arrival_ms=60.0)
    assert m0.batch_id != m1.batch_id


def test_causal_batching_includes_it():
    m0, m1 = _run(causal=True, late_arrival_ms=60.0)
    assert m0.batch_id == m1.batch_id and m0.batch_size == 2


def test_causal_batching_still_splits_outside_window():
    m0, m1 = _run(causal=True, late_arrival_ms=400.0)
    assert m0.batch_id != m1.batch_id
    assert m1.batch_start_ms >= 400.0


def test_client_exit_wakes_batcher():
    async def go():
        batcher = VerificationBatcher(_Target(), max_batch_size=8, batch_wait_ms=150.0,
                                      verify_token_budget=100, service_time_model="batch_wall")
        alive = {"other": True}
        batcher.arrival_lower_bound = lambda: 50.0 if alive["other"] else float("inf")
        await batcher.start()

        async def leave():
            await asyncio.sleep(0.05)
            alive["other"] = False
            batcher.notify()

        (meta, _), _ = await asyncio.gather(
            batcher.submit(0, [1, 2], _draft(0), 10, server_arrival_ms=0.0), leave()
        )
        await batcher.close()
        return meta

    assert asyncio.run(go()).batch_size == 1


def _single(early_close: bool):
    async def go():
        batcher = VerificationBatcher(_Target(), max_batch_size=8, batch_wait_ms=150.0,
                                      verify_token_budget=100, service_time_model="batch_wall")
        batcher.arrival_lower_bound = lambda: float("inf")
        batcher.early_close = early_close
        batcher.all_clients_waiting = lambda: True
        await batcher.start()
        m, _ = await batcher.submit(0, [1, 2], _draft(0), 10, server_arrival_ms=20.0)
        await batcher.close()
        return m

    return asyncio.run(go())


def test_window_waits_full_batch_wait_by_default():
    assert _single(False).batch_start_ms == pytest.approx(170.0)


def test_early_close_when_all_clients_waiting():
    assert _single(True).batch_start_ms == pytest.approx(20.0)


def test_barrier_waits_for_every_client_without_time_cap():
    async def go():
        batcher = VerificationBatcher(_Target(), max_batch_size=8, batch_wait_ms=float("inf"),
                                      verify_token_budget=100, service_time_model="batch_wall")
        state = {"drafting": True}
        batcher.arrival_lower_bound = lambda: 50.0 if state["drafting"] else float("inf")
        await batcher.start()

        async def early():
            return await batcher.submit(0, [1, 2], _draft(0), 10, server_arrival_ms=0.0)

        async def late():
            await asyncio.sleep(0.05)
            state["drafting"] = False
            return await batcher.submit(1, [1, 2], _draft(1), 10, server_arrival_ms=900.0)

        (m0, _), (m1, _) = await asyncio.gather(early(), late())
        await batcher.close()
        return m0, m1

    m0, m1 = asyncio.run(go())
    assert m0.batch_id == m1.batch_id and m0.batch_size == 2
    assert m0.batch_start_ms == pytest.approx(900.0)


def _continuous(server_free_ms: float, arrivals: list[float]):
    async def go():
        batcher = VerificationBatcher(_Target(), max_batch_size=8, batch_wait_ms=150.0,
                                      verify_token_budget=100, service_time_model="batch_wall")
        batcher.dispatch_mode = "continuous"
        batcher.server_available_at_ms = server_free_ms
        pending = {"n": len(arrivals)}
        batcher.arrival_lower_bound = lambda: (
            min(arrivals[len(arrivals) - pending["n"]:]) if pending["n"] else float("inf")
        )
        await batcher.start()

        async def one(cid, t):
            await asyncio.sleep(0.01 * cid)
            pending["n"] -= 1
            return await batcher.submit(cid, [1, 2], _draft(cid), 10, server_arrival_ms=t)

        out = await asyncio.gather(*(one(i, t) for i, t in enumerate(arrivals)))
        await batcher.close()
        return [m for m, _ in out]

    return asyncio.run(go())


def test_continuous_idle_server_starts_immediately():
    (m,) = _continuous(0.0, [20.0])
    assert m.batch_start_ms == pytest.approx(20.0)


def test_continuous_busy_server_takes_all_arrived_by_free_time():
    m0, m1, m2 = _continuous(500.0, [0.0, 300.0, 700.0])
    assert m0.batch_id == m1.batch_id != m2.batch_id
    assert m0.batch_start_ms == pytest.approx(500.0)
    # the simulator asserts batch_ready >= every member's server arrival
    assert m0.batch_ready_ms >= 300.0 - 1e-6
    assert m2.batch_ready_ms >= 700.0 - 1e-6
