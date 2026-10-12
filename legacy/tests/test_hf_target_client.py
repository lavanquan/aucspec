import torch

from edge_specsim.hf_target_client import (
    HFGreedyCandidateVerifier,
    greedy_verify_choices,
)


def test_greedy_verify_accepts_prefix_then_correction():
    accepted, committed, correction, bonus, eos = greedy_verify_choices(
        [10, 11, 12],
        [10, 11, 99, 77],
        remaining_tokens=8,
        eos_token_id=None,
    )
    assert accepted == 2
    assert committed == [10, 11, 99]
    assert correction == 99
    assert bonus is None
    assert eos is False


def test_greedy_verify_all_accepts_bonus():
    accepted, committed, correction, bonus, eos = greedy_verify_choices(
        [10, 11],
        [10, 11, 55],
        remaining_tokens=8,
        eos_token_id=None,
    )
    assert accepted == 2
    assert committed == [10, 11, 55]
    assert correction is None
    assert bonus == 55
    assert eos is False


def test_greedy_verify_respects_eos_and_budget():
    accepted, committed, correction, bonus, eos = greedy_verify_choices(
        [10, 2, 11],
        [10, 2, 11, 55],
        remaining_tokens=8,
        eos_token_id=2,
    )
    assert accepted == 2
    assert committed == [10, 2]
    assert correction is None
    assert bonus is None
    assert eos is True

    _, committed, _, _, _ = greedy_verify_choices(
        [10, 11],
        [10, 11, 55],
        remaining_tokens=2,
        eos_token_id=None,
    )
    assert committed == [10, 11]


def test_logit_alignment_extracts_candidate_choices_and_bonus():
    logits = torch.zeros((1, 5, 8), dtype=torch.float32)
    logits[0, 2, 4] = 5.0
    logits[0, 3, 6] = 5.0
    logits[0, 4, 7] = 5.0
    choices = HFGreedyCandidateVerifier._greedy_ids_from_logits(
        logits,
        context_length=3,
        candidate_length=2,
    )
    assert choices == [4, 6, 7]


def test_torch_argmax_gives_smallest_id_on_exact_tie():
    logits = torch.zeros((1, 1, 6), dtype=torch.float32)
    logits[0, 0, 2] = 3.0
    logits[0, 0, 4] = 3.0
    choices = HFGreedyCandidateVerifier._greedy_ids_from_logits(
        logits,
        context_length=1,
        candidate_length=0,
    )
    assert choices == [2]
