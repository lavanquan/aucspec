from __future__ import annotations

import math
from collections.abc import Mapping


def acceptance_probability(log_p: float, log_q: float) -> float:
    if math.isfinite(log_p) and math.isfinite(log_q):
        return 1.0 if log_p >= log_q else math.exp(log_p - log_q)
    if math.isfinite(log_p) and not math.isfinite(log_q):
        return 1.0
    return 0.0


def should_accept(random_draw: float, probability: float) -> bool:
    return random_draw <= probability


def greedy_correction_token(target_logprobs: dict[int, float]) -> int:
    if not target_logprobs:
        raise ValueError("target_logprobs must not be empty")
    return max(target_logprobs.items(), key=lambda item: item[1])[0]


def residual_distribution(
    target_logprobs: Mapping[int, float],
    draft_logprobs: Mapping[int, float],
) -> dict[int, float]:
    if not target_logprobs:
        raise ValueError("target_logprobs must not be empty")
    support = set(int(token_id) for token_id in target_logprobs) | set(
        int(token_id) for token_id in draft_logprobs
    )
    residual: dict[int, float] = {}
    total_mass = 0.0
    for token_id in support:
        log_p = float(target_logprobs.get(token_id, float("-inf")))
        log_q = float(draft_logprobs.get(token_id, float("-inf")))
        p_mass = 0.0 if not math.isfinite(log_p) else math.exp(log_p)
        q_mass = 0.0 if not math.isfinite(log_q) else math.exp(log_q)
        mass = max(0.0, p_mass - q_mass)
        if mass <= 0.0:
            continue
        residual[token_id] = mass
        total_mass += mass
    if total_mass <= 0.0:
        best_token = greedy_correction_token(dict(target_logprobs))
        return {best_token: 1.0}
    return {
        token_id: mass / total_mass
        for token_id, mass in residual.items()
    }


def sample_correction_token(
    target_logprobs: Mapping[int, float],
    draft_logprobs: Mapping[int, float],
    random_draw: float,
) -> int:
    distribution = residual_distribution(target_logprobs, draft_logprobs)
    clamped_draw = min(1.0, max(0.0, float(random_draw)))
    cumulative = 0.0
    selected_token = next(iter(distribution))
    for token_id, probability in sorted(
        distribution.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        cumulative += probability
        selected_token = token_id
        if clamped_draw <= cumulative:
            return token_id
    return selected_token
