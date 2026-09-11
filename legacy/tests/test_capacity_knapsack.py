"""CPU-only tests for the capacity_dpp batch value and the
`capacity_knapsack` scheduler (CAPACITY_AWARE_FRAMEWORK_CODEX_IMPLEMENTATION.md
Section 6, Section 15 "Batch control").
"""

from __future__ import annotations

import itertools

import pytest

from edge_specsim.verification_queue import VerificationQueue, VerificationRequest


def _req(
    client_id: int,
    gamma: int,
    weight: float,
    *,
    acceptance: float = 0.9,
    remaining: int = 100,
    arrival_time: float = 0.0,
    server_queue_price: float = 0.0,
    theta_f_ms_per_token: float = 0.0,
) -> VerificationRequest:
    return VerificationRequest(
        client_id=client_id,
        arrival_time=arrival_time,
        gamma=gamma,
        candidate_token_ids=list(range(gamma)),
        weight=weight,
        deadline=None,
        expected_acceptance_rate=acceptance,
        expected_acceptance_profile=[],
        context_token_ids=[],
        draft=None,  # not touched by scheduling
        remaining_tokens=remaining,
        future=object(),  # not touched by scheduling
        server_queue_price=server_queue_price,
        theta_f_ms_per_token=theta_f_ms_per_token,
    )


# --------------------------------------------------------------- value formula
def test_capacity_dpp_value_formula():
    r = _req(0, gamma=3, weight=10.0, acceptance=0.8, server_queue_price=2.0, theta_f_ms_per_token=0.5)
    # phi(3, 0.8) = 1 + 0.8 + 0.64 + 0.512 = 2.952 (i.i.d scalar, no profile)
    expected_phi = 1 + 0.8 + 0.8**2 + 0.8**3
    expected = 10.0 * expected_phi - 2.0 * 0.5 * 4  # verifier_token_cost = gamma+1 = 4
    assert r.capacity_dpp_value == pytest.approx(expected)


def test_capacity_dpp_value_can_be_negative():
    r = _req(0, gamma=4, weight=1.0, acceptance=0.5, server_queue_price=100.0, theta_f_ms_per_token=1.0)
    assert r.capacity_dpp_value < 0.0


# --------------------------------------------------------------- exact vs brute force
def _brute_force_best(reqs, max_batch_size, budget, value_fn):
    best_value, best_subset = 0.0, ()  # empty selection is always a valid option
    n = len(reqs)
    for size in range(1, min(max_batch_size, n) + 1):
        for combo in itertools.combinations(range(n), size):
            cost = sum(reqs[i].verifier_token_cost for i in combo)
            if cost > budget:
                continue
            value = sum(value_fn(reqs[i]) for i in combo)
            if value > best_value:
                best_value, best_subset = value, combo
    return best_value, best_subset


def test_capacity_knapsack_matches_brute_force_small_instance():
    reqs = [
        _req(0, gamma=1, weight=5.0, acceptance=0.9, server_queue_price=1.0, theta_f_ms_per_token=0.3),
        _req(1, gamma=4, weight=8.0, acceptance=0.7, server_queue_price=1.0, theta_f_ms_per_token=0.3),
        _req(2, gamma=2, weight=3.0, acceptance=0.95, server_queue_price=1.0, theta_f_ms_per_token=0.3),
        _req(3, gamma=0, weight=6.0, acceptance=0.5, server_queue_price=1.0, theta_f_ms_per_token=0.3),
        _req(4, gamma=3, weight=2.0, acceptance=0.6, server_queue_price=1.0, theta_f_ms_per_token=0.3),
    ]
    value_fn = lambda r: r.capacity_dpp_value
    for budget in (3, 5, 8, 12):
        q = VerificationQueue()
        q.extend(list(reqs))
        selected = q.pop_batch("capacity_knapsack", max_batch_size=5, batch_wait_ms=1e9,
                                verify_token_budget=budget, utility_lambda=0.0)
        got_value = sum(value_fn(r) for r in selected) if not q.last_batch_forced_service else None
        best_value, _ = _brute_force_best(reqs, 5, budget, value_fn)
        if q.last_batch_forced_service:
            # DP's honest answer was the empty batch (best_value <= 0);
            # forced-service picked one request anyway for liveness.
            assert best_value <= 1e-9
        else:
            assert got_value == pytest.approx(best_value)
            assert sum(r.verifier_token_cost for r in selected) <= budget
            assert len(selected) <= 5


def test_capacity_knapsack_allows_empty_batch_when_all_values_nonpositive():
    reqs = [_req(i, gamma=4, weight=1.0, acceptance=0.5, server_queue_price=1000.0,
                  theta_f_ms_per_token=1.0) for i in range(3)]
    q = VerificationQueue()
    q.extend(reqs)
    q.pop_batch("capacity_knapsack", max_batch_size=5, batch_wait_ms=1e9,
                verify_token_budget=20, utility_lambda=0.0)
    assert q.last_batch_forced_service is True  # DP chose empty -> liveness fallback kicked in


def test_server_queue_penalty_changes_selected_batch():
    # a big-gamma request is worth including at low Q_s but not at high Q_s
    def make(server_queue_price):
        return [
            _req(0, gamma=4, weight=10.0, acceptance=0.9, server_queue_price=server_queue_price, theta_f_ms_per_token=1.0),
            _req(1, gamma=1, weight=10.0, acceptance=0.9, server_queue_price=server_queue_price, theta_f_ms_per_token=1.0),
        ]

    q_low = VerificationQueue()
    q_low.extend(make(server_queue_price=0.0))
    selected_low = q_low.pop_batch("capacity_knapsack", max_batch_size=1, batch_wait_ms=1e9,
                                    verify_token_budget=5, utility_lambda=0.0)

    # value0 = 10*phi(4,0.9) - Qs*theta_f*5 = 40.951 - 5*Qs*theta_f
    # value1 = 10*phi(1,0.9) - Qs*theta_f*2 = 19.0   - 2*Qs*theta_f
    # at Qs*theta_f=9: value0 = -4.05 (negative), value1 = +1.0 (still positive)
    q_high = VerificationQueue()
    q_high.extend(make(server_queue_price=9.0))
    selected_high = q_high.pop_batch("capacity_knapsack", max_batch_size=1, batch_wait_ms=1e9,
                                      verify_token_budget=5, utility_lambda=0.0)

    assert selected_low[0].client_id == 0    # low Q_s: prefer the big-gamma request
    assert not q_high.last_batch_forced_service
    assert selected_high[0].client_id == 1   # high Q_s: its verifier cost dominates, avoid it


# --------------------------------------------------------------- legacy unaffected
def test_legacy_knapsack_scheduler_unchanged_by_capacity_value_fields():
    reqs = [
        _req(0, gamma=1, weight=5.0, acceptance=0.9),
        _req(1, gamma=4, weight=8.0, acceptance=0.7),
        _req(2, gamma=2, weight=3.0, acceptance=0.95),
    ]
    q = VerificationQueue()
    q.extend(list(reqs))
    selected = q.pop_batch("knapsack", max_batch_size=3, batch_wait_ms=1e9,
                            verify_token_budget=6, utility_lambda=0.0)
    value_fn = lambda r: r.batching_utility
    best_value, _ = _brute_force_best(reqs, 3, 6, value_fn)
    got_value = sum(value_fn(r) for r in selected)
    assert got_value == pytest.approx(best_value)
    assert sum(r.verifier_token_cost for r in selected) <= 6


def test_cardinality_and_budget_constraints_hold():
    reqs = [_req(i, gamma=4, weight=1.0 + i, acceptance=0.9) for i in range(8)]
    q = VerificationQueue()
    q.extend(list(reqs))
    selected = q.pop_batch("capacity_knapsack", max_batch_size=3, batch_wait_ms=1e9,
                            verify_token_budget=10, utility_lambda=0.0)
    assert len(selected) <= 3
    assert sum(r.verifier_token_cost for r in selected) <= 10
