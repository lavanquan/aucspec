"""CPU-only tests for the virtual verifier occupancy of one batch
(AUC_SPEC eq. 35: a batch holds the verifier for its whole service time).
target_client imports torch/transformers; when they are missing (CPU CI), it
is stubbed before the batcher import.
"""

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
from edge_specsim.verification_batcher import (  # noqa: E402
    VerificationBatcher,
    batch_service_time_ms,
)


def test_batch_wall_is_b_times_per_request_latency():
    assert batch_service_time_ms([40.0, 40.0, 40.0], "batch_wall") == pytest.approx(120.0)
    assert batch_service_time_ms([40.0, 40.0, 40.0], "per_request_average") == pytest.approx(40.0)
    assert batch_service_time_ms([], "batch_wall") == 0.0


def test_unknown_service_time_model_is_rejected():
    with pytest.raises(ValueError):
        VerificationBatcher(target=None, service_time_model="nope")


class _FakeTarget:
    """Mimics VLLMCandidateVerifier: every result carries wall_ms / B."""

    def __init__(self, wall_ms: float) -> None:
        self.wall_ms = wall_ms

    def verify_batch_sync(self, payload):
        per_request = self.wall_ms / len(payload)
        return [
            VerificationResult(accepted_length=0, committed_token_ids=[1], committed_text="",
                               target_latency_ms=per_request)
            for _ in payload
        ]


def _run_batch(service_time_model: str, batch_size: int, wall_ms: float):
    async def go():
        batcher = VerificationBatcher(
            _FakeTarget(wall_ms), max_batch_size=batch_size, batch_wait_ms=0.0,
            verify_token_budget=100, service_time_model=service_time_model,
        )
        loop = asyncio.get_running_loop()
        draft = DraftResult(client_id=0, token_ids=[5, 6], draft_logprobs=[], text="",
                            latency_ms=0.0, worker_id=0)
        from edge_specsim.verification_queue import VerificationRequest

        futures = []
        for cid in range(batch_size):
            fut = loop.create_future()
            batcher.server_queue.push(VerificationRequest(
                client_id=cid, arrival_time=0.0, gamma=2, candidate_token_ids=[5, 6], weight=1.0,
                deadline=None, expected_acceptance_rate=0.5, expected_acceptance_profile=[],
                context_token_ids=[1, 2, 3], draft=draft, remaining_tokens=10, future=fut,
            ))
            futures.append(fut)
        batch = batcher.server_queue.pop_batch(
            scheduler_name="fcfs", max_batch_size=batch_size, batch_wait_ms=0.0,
            verify_token_budget=100, utility_lambda=0.0,
        )
        await batcher._process(batch, 100)
        return [f.result()[0] for f in futures], batcher.server_available_at_ms

    return asyncio.run(go())


def test_batch_wall_occupies_verifier_for_whole_batch():
    metas, available_at = _run_batch("batch_wall", batch_size=4, wall_ms=200.0)
    assert len({m.batch_id for m in metas}) == 1
    m = metas[0]
    assert m.batch_size == 4
    assert m.verify_finish_ms - m.batch_start_ms == pytest.approx(200.0)
    assert available_at == pytest.approx(m.verify_finish_ms)
    # theta_f signal is the per-verifier-token price of the whole batch
    assert m.theta_f_ms_per_token == pytest.approx(200.0 / (4 * 3))


def test_per_request_average_reproduces_v1_semantics():
    metas, _ = _run_batch("per_request_average", batch_size=4, wall_ms=200.0)
    m = metas[0]
    assert m.verify_finish_ms - m.batch_start_ms == pytest.approx(50.0)
