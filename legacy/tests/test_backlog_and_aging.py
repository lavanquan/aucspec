"""server_queue_work=backlog needs batch metadata to carry the waiting
tokens through to the client; age_weights must re-rank waiting requests."""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import types

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


def _draft(cid, n=4):
    return DraftResult(client_id=cid, token_ids=[5] * n, draft_logprobs=[], text="",
                       latency_ms=0.0, worker_id=0)


def test_backlog_reaches_the_client_metadata():
    async def go():
        # budget 5 tokens = one request per batch; 3 simultaneous requests
        batcher = VerificationBatcher(_Target(), max_batch_size=8, batch_wait_ms=0.0,
                                      verify_token_budget=5, service_time_model="batch_wall")
        await batcher.start()
        outs = await asyncio.gather(*(
            batcher.submit(c, [1, 2], _draft(c), 10, server_arrival_ms=0.0) for c in range(3)
        ))
        await batcher.close()
        return sorted(m.backlog_token_cost for m, _ in outs)

    backlogs = asyncio.run(go())
    assert max(backlogs) == 10  # the first batch left two 5-token requests waiting
    assert min(backlogs) == 0


def test_weight_refresh_is_applied_before_pop():
    async def go():
        batcher = VerificationBatcher(_Target(), max_batch_size=8, batch_wait_ms=0.0,
                                      verify_token_budget=100, service_time_model="batch_wall")
        seen = []

        def refresh(request, now_ms):
            seen.append(request.client_id)
            return 123.0

        batcher.weight_refresh = refresh
        await batcher.start()
        await batcher.submit(0, [1, 2], _draft(0), 10, server_arrival_ms=0.0)
        await batcher.close()
        return seen

    assert 0 in asyncio.run(go())
