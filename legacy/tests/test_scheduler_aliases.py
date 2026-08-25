from __future__ import annotations

import asyncio

from edge_specsim.models import DraftResult
from edge_specsim.verification_queue import VerificationQueue, VerificationRequest


def _make_request(client_id: int, gamma: int, expected_acceptance_rate: float) -> VerificationRequest:
    loop = asyncio.new_event_loop()
    future = loop.create_future()
    return VerificationRequest(
        client_id=client_id,
        arrival_time=0.0,
        gamma=gamma,
        candidate_token_ids=[1] * gamma,
        weight=1.0,
        deadline=None,
        expected_acceptance_rate=expected_acceptance_rate,
        expected_acceptance_profile=[],
        context_token_ids=[],
        draft=DraftResult(client_id=client_id, token_ids=[1] * gamma, draft_logprobs=[], text="", latency_ms=0.0, worker_id=-1),
        remaining_tokens=32,
        future=future,
    )


def test_equal_tokens_prefers_smaller_verifier_costs() -> None:
    queue = VerificationQueue()
    queue.extend(
        [
            _make_request(client_id=0, gamma=8, expected_acceptance_rate=0.9),
            _make_request(client_id=1, gamma=1, expected_acceptance_rate=0.2),
            _make_request(client_id=2, gamma=2, expected_acceptance_rate=0.3),
        ]
    )

    selected = queue.pop_batch(
        scheduler_name="equal_tokens",
        max_batch_size=3,
        batch_wait_ms=1.0,
        verify_token_budget=5,
        utility_lambda=0.0,
    )

    assert [request.client_id for request in selected] == [1, 2]


def test_max_expected_accepted_tokens_prefers_higher_expected_accepts() -> None:
    queue = VerificationQueue()
    queue.extend(
        [
            _make_request(client_id=0, gamma=4, expected_acceptance_rate=0.2),
            _make_request(client_id=1, gamma=2, expected_acceptance_rate=0.9),
        ]
    )

    selected = queue.pop_batch(
        scheduler_name="max_expected_accepted_tokens",
        max_batch_size=1,
        batch_wait_ms=1.0,
        verify_token_budget=8,
        utility_lambda=0.0,
    )

    assert [request.client_id for request in selected] == [1]


def test_weighted_utility_prefers_higher_paper_utility_density() -> None:
    queue = VerificationQueue()
    high_density = _make_request(client_id=0, gamma=1, expected_acceptance_rate=0.9)
    high_density.weight = 10.0
    low_density = _make_request(client_id=1, gamma=4, expected_acceptance_rate=0.9)
    low_density.weight = 10.0
    queue.extend([low_density, high_density])

    selected = queue.pop_batch(
        scheduler_name="weighted_utility",
        max_batch_size=1,
        batch_wait_ms=1.0,
        verify_token_budget=8,
        utility_lambda=1000.0,
    )

    assert [request.client_id for request in selected] == [0]
