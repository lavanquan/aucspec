from __future__ import annotations

import torch

from edge_specsim.models import DraftResult


def test_delta_proposal_supports_lossless_rejection_sampling() -> None:
    proposal = DraftResult(
        client_id=0,
        token_ids=[5, 6],
        draft_logprobs=[None, None],
        text="",
        latency_ms=0.0,
        worker_id=0,
        distribution_payload="delta_proposal",
    )

    assert proposal.supports_lossless_rejection_sampling(vocab_size=1024)
    assert proposal.log_q_proposed == [0.0, 0.0]
    assert proposal.distribution_dict(0, vocab_size=1024) == {5: 0.0}


def test_sampling_requires_full_vocab_payload_for_lossless_rejection_sampling() -> None:
    proposal = DraftResult(
        client_id=0,
        token_ids=[5],
        draft_logprobs=[torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)],
        text="",
        latency_ms=0.0,
        worker_id=0,
        distribution_payload="proposed_token_logprob",
    )

    assert not proposal.supports_lossless_rejection_sampling(vocab_size=3)


def test_full_vocab_payload_exposes_dense_distribution_support() -> None:
    proposal = DraftResult(
        client_id=0,
        token_ids=[2],
        draft_logprobs=[
            torch.tensor([float("-inf"), -1.5, -0.2, float("-inf")], dtype=torch.float32)
        ],
        text="",
        latency_ms=0.0,
        worker_id=0,
        distribution_payload="full_vocab_logprobs",
    )

    assert proposal.supports_lossless_rejection_sampling(vocab_size=4)
    assert proposal.distribution_dict(0, vocab_size=4) == {1: -1.5, 2: -0.2}
