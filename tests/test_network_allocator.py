from __future__ import annotations

import torch

from edge_specsim.models import DraftResult
from edge_specsim.network import paper_square_root_allocation_weight
from edge_specsim.network import (
    estimate_uplink_bits_per_token,
    estimate_uplink_bytes,
    estimate_uplink_bytes_from_gamma,
)


def test_paper_square_root_weight_matches_closed_form() -> None:
    weight = paper_square_root_allocation_weight(
        queue_price=9.0,
        payload_bits_per_token=4.0,
        gamma=4,
        spectral_efficiency=1.0,
    )

    assert weight == 12.0


def test_paper_square_root_weight_is_zero_without_price_or_gamma() -> None:
    assert paper_square_root_allocation_weight(0.0, 4.0, 4, 2.0) == 0.0
    assert paper_square_root_allocation_weight(3.0, 4.0, 0, 2.0) == 0.0
    assert paper_square_root_allocation_weight(3.0, 0.0, 4, 2.0) == 0.0


def test_paper_square_root_weight_decreases_with_higher_spectral_efficiency() -> None:
    low_efficiency = paper_square_root_allocation_weight(
        queue_price=9.0,
        payload_bits_per_token=4.0,
        gamma=4,
        spectral_efficiency=1.0,
    )
    high_efficiency = paper_square_root_allocation_weight(
        queue_price=9.0,
        payload_bits_per_token=4.0,
        gamma=4,
        spectral_efficiency=4.0,
    )

    assert high_efficiency < low_efficiency


def test_delta_proposal_uplink_payload_is_compact() -> None:
    proposal = DraftResult(
        client_id=0,
        token_ids=[11, 12, 13],
        draft_logprobs=[None, None, None],
        text="",
        latency_ms=0.0,
        worker_id=0,
        distribution_payload="delta_proposal",
    )

    assert estimate_uplink_bytes(proposal, vocab_size=1024) == estimate_uplink_bytes_from_gamma(
        3,
        distribution_payload="delta_proposal",
    )


def test_uplink_kappa_bits_per_token_matches_one_token_payload_delta() -> None:
    expected_bits = 8.0 * (
        estimate_uplink_bytes_from_gamma(1, distribution_payload="delta_proposal")
        - estimate_uplink_bytes_from_gamma(0, distribution_payload="delta_proposal")
    )

    assert estimate_uplink_bits_per_token(distribution_payload="delta_proposal") == expected_bits


def test_full_vocab_uplink_payload_scales_with_vocab_size() -> None:
    proposal = DraftResult(
        client_id=0,
        token_ids=[7, 8],
        draft_logprobs=[
            torch.zeros(16, dtype=torch.float32),
            torch.zeros(16, dtype=torch.float32),
        ],
        text="",
        latency_ms=0.0,
        worker_id=0,
        distribution_payload="full_vocab_logprobs",
    )

    compact_bytes = estimate_uplink_bytes_from_gamma(2, distribution_payload="delta_proposal")
    full_bytes = estimate_uplink_bytes(proposal, vocab_size=16)

    assert full_bytes > compact_bytes
    assert full_bytes == estimate_uplink_bytes_from_gamma(
        2,
        distribution_payload="full_vocab_logprobs",
        vocab_size=16,
    )
