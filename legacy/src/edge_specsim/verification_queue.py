from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from math import inf

from .expectations import expected_useful_tokens, expected_useful_tokens_from_profile
from .models import DraftResult, VerificationResult


@dataclass
class VerificationRequest:
    client_id: int
    arrival_time: float
    gamma: int
    candidate_token_ids: list[int]
    weight: float
    deadline: float | None
    expected_acceptance_rate: float
    expected_acceptance_profile: list[float]
    context_token_ids: list[int]
    draft: DraftResult
    remaining_tokens: int
    future: asyncio.Future[tuple["VerificationBatchMetadata", VerificationResult]]
    # CAPACITY_AWARE_FRAMEWORK_CODEX_IMPLEMENTATION.md Section 6.2: fields
    # for the capacity_dpp-specific batch value, used ONLY by the
    # `capacity_knapsack` scheduler. Default 0.0 so legacy schedulers
    # (fcfs, weighted_utility, knapsack, ...) are byte-for-byte unchanged.
    server_queue_price: float = 0.0       # Q_s at submission time
    theta_f_ms_per_token: float = 0.0     # DPP compute-work price (ms/verifier-token)

    @property
    def verifier_token_cost(self) -> int:
        return max(1, self.gamma + 1)

    @property
    def service_weight(self) -> float:
        """Alias for `weight` (= V + Z_i for capacity_dpp) -- Section 6.1's
        v_i formula names it service_weight; kept as a read-only alias so
        existing code that sets `weight` needs no change."""
        return self.weight

    @property
    def expected_accepted_tokens(self) -> float:
        return max(0.0, min(1.0, self.expected_acceptance_rate)) * max(0, self.gamma)

    @property
    def expected_useful_tokens(self) -> float:
        if self.remaining_tokens <= 0:
            return 0.0
        if self.expected_acceptance_profile:
            expected = expected_useful_tokens_from_profile(self.expected_acceptance_profile)
        else:
            expected = expected_useful_tokens(self.gamma, self.expected_acceptance_rate)
        return min(
            float(self.remaining_tokens),
            expected,
        )

    @property
    def batching_utility(self) -> float:
        return max(0.0, float(self.weight)) * self.expected_useful_tokens

    @property
    def capacity_dpp_value(self) -> float:
        """Section 6.1: v_i = (V+Z_i) phi_hat_i(gamma_i) - Q_s theta_f c_i^v.
        Unlike `batching_utility`, this MAY be negative -- the exact DP
        knapsack must allow an empty batch when every value is
        non-positive (Section 6.2)."""
        return (
            self.service_weight * self.expected_useful_tokens
            - self.server_queue_price * self.theta_f_ms_per_token * self.verifier_token_cost
        )


@dataclass(frozen=True)
class VerificationBatchMetadata:
    batch_id: int
    batch_size: int
    batch_token_cost: int
    batch_token_budget: int
    scheduler_name: str
    measured_batch_service_ms: float
    modeled_batch_service_ms: float
    compute_bound_batch_service_ms: float
    theta_f_ms_per_token: float
    batch_ready_ms: float
    batch_start_ms: float
    verify_finish_ms: float
    forced_service: bool = False
    sum_capacity_value: float = 0.0


@dataclass
class VerificationQueue:
    waiting: list[VerificationRequest] = field(default_factory=list)
    random_seed: int = 0
    _rng: random.Random = field(init=False, repr=False)
    last_batch_forced_service: bool = False

    def __post_init__(self) -> None:
        self._rng = random.Random(self.random_seed)

    def push(self, request: VerificationRequest) -> None:
        self.waiting.append(request)
        self.waiting.sort(key=lambda item: item.arrival_time)

    def extend(self, requests: list[VerificationRequest]) -> None:
        if not requests:
            return
        self.waiting.extend(requests)
        self.waiting.sort(key=lambda item: item.arrival_time)

    def pop_batch(
        self,
        scheduler_name: str,
        max_batch_size: int,
        batch_wait_ms: float,
        verify_token_budget: int,
        utility_lambda: float,
    ) -> list[VerificationRequest]:
        if not self.waiting:
            return []
        cutoff_ms = self.waiting[0].arrival_time + batch_wait_ms
        eligible = [request for request in self.waiting if request.arrival_time <= cutoff_ms]
        if not eligible:
            return []
        if scheduler_name == "fcfs":
            selected = self._select_fcfs(eligible, max_batch_size, verify_token_budget)
        elif scheduler_name == "random":
            selected = self._select_random(eligible, max_batch_size, verify_token_budget)
        elif scheduler_name == "equal_tokens":
            selected = self._select_ranked(
                eligible,
                max_batch_size,
                verify_token_budget,
                key_fn=lambda request: (
                    -request.verifier_token_cost,
                    -request.expected_useful_tokens,
                    -request.arrival_time,
                ),
            )
        elif scheduler_name == "max_expected_accepted_tokens":
            selected = self._select_ranked(
                eligible,
                max_batch_size,
                verify_token_budget,
                key_fn=lambda request: (
                    request.expected_accepted_tokens,
                    request.expected_useful_tokens,
                    -request.arrival_time,
                ),
            )
        elif scheduler_name == "max_throughput":
            selected = self._select_ranked(
                eligible,
                max_batch_size,
                verify_token_budget,
                key_fn=lambda request: (
                    request.expected_useful_tokens / request.verifier_token_cost,
                    request.expected_useful_tokens,
                    -request.arrival_time,
                ),
            )
        elif scheduler_name == "max_expected_accepted_per_cost":
            selected = self._select_ranked(
                eligible,
                max_batch_size,
                verify_token_budget,
                key_fn=lambda request: (
                    request.expected_accepted_tokens / request.verifier_token_cost,
                    request.expected_accepted_tokens,
                    -request.arrival_time,
                ),
            )
        elif scheduler_name == "weighted_utility":
            selected = self._select_ranked(
                eligible,
                max_batch_size,
                verify_token_budget,
                key_fn=lambda request: (
                    self._weighted_utility_density(request, utility_lambda),
                    self._weighted_utility_value(request, utility_lambda),
                    -request.arrival_time,
                ),
            )
        elif scheduler_name == "knapsack":
            selected = self._select_knapsack(
                eligible,
                max_batch_size,
                verify_token_budget,
                value_fn=lambda request: self._weighted_utility_value(request, utility_lambda),
            )
        elif scheduler_name == "capacity_knapsack":
            # CAPACITY_AWARE_FRAMEWORK_CODEX_IMPLEMENTATION.md Section 6:
            # capacity_dpp-specific value v_i = (V+Z_i) phi_i - Q_s theta_f
            # c_i^v, which MAY be negative -- the exact DP already prefers
            # the empty selection (value 0.0) over any negative-value pick,
            # so "allow an empty batch when every value is non-positive"
            # falls out of the existing DP structure with no extra code.
            selected = self._select_knapsack(
                eligible,
                max_batch_size,
                verify_token_budget,
                value_fn=lambda request: request.capacity_dpp_value,
            )
        else:
            raise ValueError(
                "verification_batching.scheduler must be one of: "
                "fcfs, random, equal_tokens, max_expected_accepted_tokens, "
                "max_throughput, max_expected_accepted_per_cost, weighted_utility, "
                "knapsack, capacity_knapsack"
            )

        self.last_batch_forced_service = not selected
        if not selected:
            # Section 6.2: liveness escape hatch so a request is never
            # deadlocked forever when the optimizer's honest choice was
            # "serve nobody this cycle" (e.g. every capacity_dpp_value is
            # negative). Logged via last_batch_forced_service -- this is
            # NOT the DPP optimum and must never be presented as one.
            selected = [eligible[0]]

        selected_ids = {id(request) for request in selected}
        self.waiting = [
            request for request in self.waiting if id(request) not in selected_ids
        ]
        self.waiting.sort(key=lambda item: item.arrival_time)
        return selected

    def has_request_past_cutoff(self, cutoff_ms: float) -> bool:
        return any(request.arrival_time > cutoff_ms for request in self.waiting)

    def eligible_requests(self, batch_wait_ms: float) -> list[VerificationRequest]:
        if not self.waiting:
            return []
        cutoff_ms = self.waiting[0].arrival_time + batch_wait_ms
        return [request for request in self.waiting if request.arrival_time <= cutoff_ms]

    def __bool__(self) -> bool:
        return bool(self.waiting)

    @staticmethod
    def _weighted_utility_value(
        request: VerificationRequest,
        utility_lambda: float,
    ) -> float:
        del utility_lambda
        return request.batching_utility

    @classmethod
    def _weighted_utility_density(
        cls,
        request: VerificationRequest,
        utility_lambda: float,
    ) -> float:
        del utility_lambda
        return cls._weighted_utility_value(request, 0.0) / request.verifier_token_cost

    @staticmethod
    def _select_fcfs(
        eligible: list[VerificationRequest],
        max_batch_size: int,
        verify_token_budget: int,
    ) -> list[VerificationRequest]:
        selected: list[VerificationRequest] = []
        total_cost = 0
        for request in eligible:
            request_cost = request.verifier_token_cost
            fits_budget = total_cost + request_cost <= verify_token_budget
            if len(selected) < max_batch_size and (fits_budget or not selected):
                selected.append(request)
                total_cost += request_cost
        return selected

    @staticmethod
    def _select_ranked(
        eligible: list[VerificationRequest],
        max_batch_size: int,
        verify_token_budget: int,
        key_fn,
    ) -> list[VerificationRequest]:
        ranked = sorted(
            eligible,
            key=lambda request: (
                key_fn(request),
                -request.arrival_time,
            ),
            reverse=True,
        )
        selected: list[VerificationRequest] = []
        total_cost = 0
        for request in ranked:
            request_cost = request.verifier_token_cost
            if len(selected) >= max_batch_size:
                break
            if total_cost + request_cost <= verify_token_budget:
                selected.append(request)
                total_cost += request_cost
        return selected

    def _select_random(
        self,
        eligible: list[VerificationRequest],
        max_batch_size: int,
        verify_token_budget: int,
    ) -> list[VerificationRequest]:
        ranked = list(eligible)
        self._rng.shuffle(ranked)
        selected: list[VerificationRequest] = []
        total_cost = 0
        for request in ranked:
            request_cost = request.verifier_token_cost
            if len(selected) >= max_batch_size:
                break
            if total_cost + request_cost <= verify_token_budget:
                selected.append(request)
                total_cost += request_cost
        return selected

    @classmethod
    def _select_knapsack(
        cls,
        eligible: list[VerificationRequest],
        max_batch_size: int,
        verify_token_budget: int,
        value_fn,
    ) -> list[VerificationRequest]:
        """Exact 0/1 knapsack DP. `value_fn(request) -> float` may return
        negative values (Section 6.2); dp[0][0] = (0.0, []) means the
        EMPTY selection always competes on equal footing, so the DP
        naturally returns an empty batch when every value is non-positive
        -- no special-casing needed."""
        capped_budget = max(1, verify_token_budget)
        max_items = max(1, max_batch_size)
        dp: list[list[tuple[float, list[int]]]] = [
            [(-inf, []) for _ in range(capped_budget + 1)]
            for _ in range(max_items + 1)
        ]
        dp[0][0] = (0.0, [])
        for index, request in enumerate(eligible):
            value = value_fn(request)
            cost = request.verifier_token_cost
            next_dp = [row[:] for row in dp]
            for item_count in range(max_items - 1, -1, -1):
                for used_budget in range(capped_budget - cost + 1):
                    current_value, current_items = dp[item_count][used_budget]
                    if current_value == -inf:
                        continue
                    new_value = current_value + value
                    new_budget = used_budget + cost
                    existing_value, existing_items = next_dp[item_count + 1][new_budget]
                    candidate_items = current_items + [index]
                    if (
                        new_value > existing_value
                        or (
                            new_value == existing_value
                            and cls._tie_break_indices(candidate_items, existing_items)
                        )
                    ):
                        next_dp[item_count + 1][new_budget] = (new_value, candidate_items)
            dp = next_dp

        best_value = -inf
        best_indices: list[int] = []
        for item_count in range(max_items + 1):
            for used_budget in range(capped_budget + 1):
                value, indices = dp[item_count][used_budget]
                if value == -inf:
                    continue
                if (
                    value > best_value
                    or (
                        value == best_value
                        and cls._tie_break_indices(indices, best_indices)
                    )
                ):
                    best_value = value
                    best_indices = indices
        return [eligible[index] for index in best_indices]

    @staticmethod
    def _tie_break_indices(candidate: list[int], incumbent: list[int]) -> bool:
        if not incumbent:
            return True
        return tuple(candidate) < tuple(incumbent)
