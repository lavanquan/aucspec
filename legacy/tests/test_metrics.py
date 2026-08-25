from __future__ import annotations

from edge_specsim.models import VerificationResult


def test_useful_tokens_count_matches_committed_tokens_with_correction() -> None:
    verification = VerificationResult(
        accepted_length=2,
        committed_token_ids=[11, 12, 99],
        committed_text="",
        target_latency_ms=0.0,
        correction_token_id=99,
    )

    assert verification.useful_token_count == 3
    assert verification.accepted_length == 2
    assert verification.correction_token_id == 99
    assert verification.bonus_token_id is None


def test_useful_tokens_count_matches_committed_tokens_with_bonus() -> None:
    verification = VerificationResult(
        accepted_length=3,
        committed_token_ids=[21, 22, 23, 24],
        committed_text="",
        target_latency_ms=0.0,
        bonus_token_id=24,
    )

    assert verification.useful_token_count == 4
    assert verification.accepted_length == 3
    assert verification.bonus_token_id == 24
    assert verification.correction_token_id is None
