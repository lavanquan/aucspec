from __future__ import annotations


def expected_useful_tokens(
    gamma: int,
    alpha: float,
) -> float:
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    clipped_alpha = max(0.0, min(1.0, float(alpha)))
    if gamma == 0:
        return 1.0
    if clipped_alpha == 1.0:
        return float(gamma + 1)
    return (1.0 - clipped_alpha ** (gamma + 1)) / (1.0 - clipped_alpha)


def expected_useful_tokens_from_profile(
    acceptance_profile: list[float],
) -> float:
    if not acceptance_profile:
        return 1.0
    expected = 1.0
    prefix_success_probability = 1.0
    for alpha in acceptance_profile:
        clipped_alpha = max(0.0, min(1.0, float(alpha)))
        prefix_success_probability *= clipped_alpha
        expected += prefix_success_probability
    return expected
