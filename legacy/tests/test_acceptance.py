from __future__ import annotations

import math

from edge_specsim.acceptance import (
    acceptance_probability,
    greedy_correction_token,
    residual_distribution,
    sample_correction_token,
    should_accept,
)


def test_acceptance_probability_is_one_when_p_equals_q() -> None:
    log_p = math.log(0.25)
    log_q = math.log(0.25)
    assert acceptance_probability(log_p, log_q) == 1.0


def test_acceptance_probability_is_one_when_p_exceeds_q() -> None:
    log_p = math.log(0.40)
    log_q = math.log(0.20)
    assert acceptance_probability(log_p, log_q) == 1.0


def test_acceptance_probability_matches_p_over_q_when_p_is_smaller() -> None:
    log_p = math.log(0.20)
    log_q = math.log(0.50)
    expected = 0.20 / 0.50
    assert acceptance_probability(log_p, log_q) == expected
    assert should_accept(0.39, expected)
    assert not should_accept(0.41, expected)


def test_greedy_correction_token_comes_from_target_not_draft_bias() -> None:
    target_logprobs = {
        7: math.log(0.60),
        3: math.log(0.25),
        9: math.log(0.15),
    }
    draft_preferred_token = 3
    assert draft_preferred_token != greedy_correction_token(target_logprobs)
    assert greedy_correction_token(target_logprobs) == 7


def test_residual_distribution_excludes_negative_residual_mass() -> None:
    target_logprobs = {
        0: math.log(0.50),
        1: math.log(0.30),
        2: math.log(0.20),
    }
    draft_logprobs = {
        0: math.log(0.10),
        1: math.log(0.80),
        2: math.log(0.10),
    }

    residual = residual_distribution(target_logprobs, draft_logprobs)

    assert 1 not in residual
    assert residual[0] > residual[2]
    assert math.isclose(sum(residual.values()), 1.0)


def test_sample_correction_token_comes_from_residual_distribution() -> None:
    target_logprobs = {
        0: math.log(0.50),
        1: math.log(0.30),
        2: math.log(0.20),
    }
    draft_logprobs = {
        0: math.log(0.10),
        1: math.log(0.80),
        2: math.log(0.10),
    }

    assert sample_correction_token(target_logprobs, draft_logprobs, 0.10) == 0
    assert sample_correction_token(target_logprobs, draft_logprobs, 0.95) == 2
