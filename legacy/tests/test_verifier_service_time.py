"""CPU-only tests for verifier service-time semantics (V1 vs V1-batchwall).

A mock target stamps a known batch wall time via stamp_batch_timing, exactly as
VLLMCandidateVerifier.verify_batch_sync does, so no torch/vLLM is needed.
"""

from __future__ import annotations

import asyncio

import pytest

from edge_specsim.models import DraftResult, VerificationResult, stamp_batch_timing
from edge_specsim.verification_batcher import SERVICE_TIME_MODELS, VerificationBatcher


class MockTarget:
    VERIFIER_SEMANTICS = "V1"

    def __init__(self, wall_ms_by_size):
        self.wall_ms_by_size = wall_ms_by_size
        self.calls = []

    def verify_batch_sync(self, payload):
        self.calls.append(len(payload))
        results = [
            VerificationResult(
                accepted_length=0,
                committed_token_ids=[1],
                committed_text="x",
                target_latency_ms=0.0,
            )
            for _ in payload
        ]
        stamp_batch_timing(results, self.wall_ms_by_size[len(payload)])
        return results


def _draft(client_id, gamma):
    return DraftResult(
        client_id=client_id,
        token_ids=list(range(gamma)),
        draft_logprobs=[None] * gamma,
        text="",
        latency_ms=1.0,
        worker_id=0,
    )


async def _run_groups(model, sizes, wall, arrivals, gamma=3, ctx=10):
    target = MockTarget(wall)
    batcher = VerificationBatcher(
        target,
        max_batch_size=max(sizes),
        batch_wait_ms=3.0,
        verify_token_budget=1000,
        service_time_model=model,
    )
    await batcher.start()
    out = []
    for group, (size, arrival) in enumerate(zip(sizes, arrivals)):
        reqs = [
            batcher.submit(
                client_id=group * 100 + i,
                context_token_ids=list(range(ctx)),
                draft=_draft(i, gamma),
                remaining_tokens=100,
                server_arrival_ms=arrival,
            )
            for i in range(size)
        ]
        out.append(await asyncio.gather(*reqs))
    await batcher.close()
    return target, out


def run(*args, **kw):
    return asyncio.run(_run_groups(*args, **kw))


def test_models_listed():
    assert SERVICE_TIME_MODELS == ("amortized_per_request", "batch_wall")
    with pytest.raises(ValueError):
        VerificationBatcher(MockTarget({}), service_time_model="bogus")


def test_stamp_batch_timing():
    results = [VerificationResult(0, [1], "x", 0.0) for _ in range(4)]
    stamp_batch_timing(results, 200.0)
    assert {r.target_latency_ms for r in results} == {50.0}
    assert {r.target_batch_wall_ms for r in results} == {200.0}


@pytest.mark.parametrize("model", SERVICE_TIME_MODELS)
def test_batch_size_one_identical(model):
    _, out = run(model, [1], {1: 174.0}, [0.0])
    meta, _ = out[0][0]
    assert meta.modeled_batch_service_ms == pytest.approx(174.0)
    assert meta.verify_finish_ms == pytest.approx(meta.batch_start_ms + 174.0)


def test_batchwall_occupies_full_wall():
    _, out = run("batch_wall", [4], {4: 200.0}, [0.0])
    meta, _ = out[0][0]
    assert meta.modeled_batch_service_ms == pytest.approx(200.0)
    assert meta.verify_finish_ms - meta.batch_start_ms == pytest.approx(200.0)
    assert meta.batch_wall_ms == pytest.approx(200.0)
    assert meta.amortized_request_ms == pytest.approx(50.0)


def test_v1_amortized_is_wall_over_b():
    _, out = run("amortized_per_request", [4], {4: 200.0}, [0.0])
    meta, _ = out[0][0]
    assert meta.modeled_batch_service_ms == pytest.approx(50.0)
    assert meta.batch_wall_ms == pytest.approx(200.0)  # diagnostic still full wall
    assert meta.service_time_model == "amortized_per_request"


def test_same_finish_for_all_requests_in_batch():
    _, out = run("batch_wall", [5], {5: 300.0}, [0.0])
    metas = [m for m, _ in out[0]]
    assert len({m.batch_id for m in metas}) == 1
    assert len({m.verify_finish_ms for m in metas}) == 1


def test_next_batch_cannot_start_before_previous_finishes():
    # second group arrives (virtual) long before the first one finishes
    _, out = run("batch_wall", [3, 3], {3: 400.0}, [0.0, 1.0])
    first, second = out[0][0][0], out[1][0][0]
    assert first.verify_finish_ms == pytest.approx(first.batch_start_ms + 400.0)
    assert second.batch_start_ms == pytest.approx(first.verify_finish_ms)
    assert second.verify_finish_ms == pytest.approx(first.verify_finish_ms + 400.0)


def test_v1_second_batch_advances_by_amortized_only():
    _, out = run("amortized_per_request", [3, 3], {3: 300.0}, [0.0, 1.0])
    first, second = out[0][0][0], out[1][0][0]
    assert second.batch_start_ms == pytest.approx(first.verify_finish_ms)
    assert first.verify_finish_ms - first.batch_start_ms == pytest.approx(100.0)


def test_idle_gap_not_charged():
    _, out = run("batch_wall", [2, 2], {2: 100.0}, [0.0, 5000.0])
    first, second = out[0][0][0], out[1][0][0]
    assert second.batch_start_ms == pytest.approx(5003.0)  # arrival + batch_wait
    assert second.verify_finish_ms == pytest.approx(5103.0)
    assert second.batch_start_ms > first.verify_finish_ms


def test_token_accounting_unchanged_and_new_counts():
    _, out = run("batch_wall", [4], {4: 200.0}, [0.0], gamma=3, ctx=10)
    meta, _ = out[0][0]
    assert meta.batch_size == 4
    assert meta.batch_token_cost == 4 * (3 + 1)  # logical verifier tokens, gamma+1
    assert meta.batch_context_tokens_sum == 40
    assert meta.batch_context_tokens_max == 10
    assert meta.batch_processed_tokens == 40 + 4 * 3  # context + draft tokens


def test_batch_size_constraint_still_respected():
    target = MockTarget({1: 10.0, 2: 20.0})

    async def go():
        b = VerificationBatcher(
            target, max_batch_size=2, batch_wait_ms=3.0, verify_token_budget=1000,
            service_time_model="batch_wall",
        )
        await b.start()
        res = await asyncio.gather(*[
            b.submit(i, [1, 2], _draft(i, 2), 50, 0.0) for i in range(4)
        ])
        await b.close()
        return res

    asyncio.run(go())
    assert max(target.calls) <= 2 and sum(target.calls) == 4


def test_batch_wall_fallback_when_target_does_not_stamp():
    class Bare(MockTarget):
        def verify_batch_sync(self, payload):
            rs = super().verify_batch_sync(payload)
            for r in rs:
                r.target_batch_wall_ms = 0.0
            return rs

    async def go():
        b = VerificationBatcher(Bare({1: 5.0}), service_time_model="batch_wall")
        await b.start()
        r = await b.submit(0, [1], _draft(0, 1), 10, 0.0)
        await b.close()
        return r

    meta, _ = asyncio.run(go())
    assert meta.batch_wall_ms > 0.0 and meta.modeled_batch_service_ms == meta.batch_wall_ms
