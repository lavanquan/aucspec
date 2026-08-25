from __future__ import annotations

import math

from .models import DraftProposal, VerificationResult


def estimate_uplink_bytes_from_gamma(
    gamma: int,
    *,
    distribution_payload: str = "proposed_token_logprob",
    vocab_size: int | None = None,
    logprob_dtype_bytes: int = 4,
    metadata_bytes: int = 64,
    request_id_bytes: int = 16,
    client_id_bytes: int = 8,
) -> int:
    candidate_bytes = gamma * 4
    if distribution_payload == "delta_proposal":
        logprob_bytes = 0
    elif distribution_payload == "proposed_token_logprob":
        logprob_bytes = gamma * logprob_dtype_bytes
    elif distribution_payload == "full_vocab_logprobs":
        if vocab_size is None:
            raise ValueError("vocab_size is required for full_vocab_logprobs uplink estimation")
        logprob_bytes = gamma * int(vocab_size) * logprob_dtype_bytes
    else:
        raise ValueError(f"Unsupported distribution_payload={distribution_payload!r}")
    return candidate_bytes + logprob_bytes + metadata_bytes + request_id_bytes + client_id_bytes


def estimate_uplink_bits_per_token(
    *,
    distribution_payload: str = "proposed_token_logprob",
    vocab_size: int | None = None,
    logprob_dtype_bytes: int = 4,
) -> float:
    marginal_payload_bytes = max(
        0,
        estimate_uplink_bytes_from_gamma(
            1,
            distribution_payload=distribution_payload,
            vocab_size=vocab_size,
            logprob_dtype_bytes=logprob_dtype_bytes,
            metadata_bytes=64,
            request_id_bytes=16,
            client_id_bytes=8,
        )
        - estimate_uplink_bytes_from_gamma(
            0,
            distribution_payload=distribution_payload,
            vocab_size=vocab_size,
            logprob_dtype_bytes=logprob_dtype_bytes,
            metadata_bytes=64,
            request_id_bytes=16,
            client_id_bytes=8,
        ),
    )
    return 8.0 * float(marginal_payload_bytes)


def estimate_uplink_bytes(proposal: DraftProposal, vocab_size: int | None = None) -> int:
    if proposal.distribution_payload == "full_vocab_logprobs":
        if vocab_size is None:
            raise ValueError(
                "vocab_size is required to estimate uplink bytes for full_vocab_logprobs payloads"
            )
        return proposal.estimate_uplink_bytes(vocab_size)
    if vocab_size is not None:
        return proposal.estimate_uplink_bytes(vocab_size)
    return estimate_uplink_bytes_from_gamma(
        len(proposal.token_ids),
        distribution_payload=proposal.distribution_payload,
        logprob_dtype_bytes=proposal.logprob_dtype_bytes,
        metadata_bytes=proposal.metadata_bytes,
        request_id_bytes=proposal.request_id_bytes,
        client_id_bytes=proposal.client_id_bytes,
    )


def estimate_downlink_bytes_from_token_count(token_count: int, accepted_length: int | None = None) -> int:
    accepted_length_bytes = 4
    committed_token_bytes = token_count * 4
    acceptance_feedback_bytes = (accepted_length if accepted_length is not None else token_count) * 4
    metadata_bytes = 32
    return (
        accepted_length_bytes
        + committed_token_bytes
        + acceptance_feedback_bytes
        + metadata_bytes
    )


def estimate_downlink_bytes(verification: VerificationResult) -> int:
    return estimate_downlink_bytes_from_token_count(
        len(verification.committed_token_ids),
        accepted_length=len(verification.acceptance_probabilities),
    )


def paper_square_root_allocation_weight(
    queue_price: float,
    payload_bits_per_token: float,
    gamma: int,
    spectral_efficiency: float,
) -> float:
    effective_gamma = max(0, int(gamma))
    effective_price = max(0.0, float(queue_price))
    effective_payload_bits_per_token = max(0.0, float(payload_bits_per_token))
    effective_spectral_efficiency = max(1e-9, float(spectral_efficiency))
    if (
        effective_gamma <= 0
        or effective_price <= 0.0
        or effective_payload_bits_per_token <= 0.0
    ):
        return 0.0
    return math.sqrt(
        effective_price
        * effective_payload_bits_per_token
        * effective_gamma
        / effective_spectral_efficiency
    )
