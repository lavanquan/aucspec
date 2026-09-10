from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Literal


@dataclass(frozen=True)
class PromptSample:
    sample_id: str
    dataset_name: str
    prompt: str
    max_new_tokens: int
    reference: str | None = None


@dataclass
class ClientProfile:
    client_id: int
    prompt_queue: Deque[PromptSample]
    draft_model_id: str
    draft_device_class: str
    draft_tokens_per_second: float
    draft_fixed_latency_ms: float
    rtt_ms: float
    uplink_mbps: float
    downlink_mbps: float
    uplink_base_snr_db: float
    downlink_base_snr_db: float
    uplink_snr_jitter_db: float
    downlink_snr_jitter_db: float
    snr_period_ms: float
    snr_phase_rad: float
    wireless_weight: float
    packet_loss: float
    channel_model: str = "iid_block_fading"
    snr_block_duration_ms: float = 100.0
    channel_seed: int = 0
    gamma: int = 1
    z_queue: float = 0.0
    device_queue: float = 0.0
    generated_tokens: int = 0
    # AUC_ACHIEVABLE_REGION_DIAGNOSTIC.md Section 4: diagnostic-only class
    # priority multiplier m_{c(i)} applied to the server-side verification
    # weight w_i = m_{c(i)} * (V + Z_i). Must not affect acceptance, draft
    # latency, network latency, dataset sampling, reward accounting,
    # x_requirement, or virtual queue update equations -- only which
    # requests the batcher prioritizes.
    diagnostic_priority_multiplier: float = 1.0
    alpha_prior_success: float = 1.0
    alpha_prior_failure: float = 1.0
    alpha_ucb_scale: float = 1.0
    alpha_discount: float = 1.0
    alpha_learning_mode: str = "bernstein_censored"
    alpha_profile_mode: str = "position"
    alpha_window_size: int = 0
    alpha_confidence_delta: float = 0.05
    alpha_min_value: float = 1e-6
    alpha_max_value: float = 1.0 - 1e-6
    alpha_successes: float = 0.0
    alpha_failures: float = 0.0
    alpha_position_successes: dict[int, float] = field(default_factory=dict)
    alpha_position_failures: dict[int, float] = field(default_factory=dict)
    alpha_position_hat: dict[int, float] = field(default_factory=dict)
    alpha_position_ucb: dict[int, float] = field(default_factory=dict)
    alpha_observed_tokens: int = 0
    alpha_censored_rounds: int = 0
    alpha_failure_events: int = 0
    alpha_hat: float = 0.5
    alpha_ucb: float = 0.5
    speculative_rounds: int = 0
    last_selected_gamma: int = 0
    last_forced_exploration: bool = False
    last_exploration_reason: str = "policy"
    accepted_tokens: int = 0
    useful_tokens: int = 0
    rounds_completed: int = 0
    samples_completed: int = 0
    virtual_start_time_ms: float = 0.0
    virtual_time_ms: float = 0.0
    sample_start_time_ms: float = 0.0
    current_sample: PromptSample | None = None
    history_token_ids: list[int] = field(default_factory=list)
    generated_for_sample: int = 0
    draft_speed_multiplier: float = 1.0
    alpha_recent_events: Deque[tuple[int, int, bool]] = field(default_factory=deque)

    def _normalized_learning_mode(self) -> str:
        mode = str(self.alpha_learning_mode).strip().lower()
        if mode == "discounted_ucb":
            return "bernstein_censored_discounted"
        return mode

    def _alpha_prior_mean(self) -> float:
        return self.alpha_prior_success / max(
            1e-9,
            self.alpha_prior_success + self.alpha_prior_failure,
        )

    def _clip_alpha(self, value: float) -> float:
        return min(
            float(self.alpha_max_value),
            max(float(self.alpha_min_value), float(value)),
        )

    def _bernstein_ucb(
        self,
        successes: float,
        failures: float,
        total_rounds: int,
    ) -> tuple[float, float]:
        observed = max(0.0, float(successes) + float(failures))
        if observed <= 1e-9:
            prior_mean = self._clip_alpha(self._alpha_prior_mean())
            return prior_mean, prior_mean

        empirical_mean = float(successes) / observed
        empirical_mean = self._clip_alpha(empirical_mean)
        empirical_variance = empirical_mean * (1.0 - empirical_mean)
        if self.alpha_confidence_delta > 0.0:
            delta = min(0.5, max(1e-12, float(self.alpha_confidence_delta)))
        else:
            delta = 1.0 / float(max(2, total_rounds + 1) ** 2)
        log_term = math.log(3.0 / delta)
        radius = math.sqrt(2.0 * empirical_variance * log_term / observed) + (
            3.0 * log_term / observed
        )
        radius *= max(0.0, float(self.alpha_ucb_scale))
        upper = self._clip_alpha(empirical_mean + radius)
        return empirical_mean, upper

    def refresh_learning_stats(self, total_rounds: int) -> None:
        self.alpha_hat, self.alpha_ucb = self._bernstein_ucb(
            self.alpha_successes,
            self.alpha_failures,
            total_rounds,
        )

        all_positions = set(self.alpha_position_successes) | set(self.alpha_position_failures)
        for position in all_positions:
            successes = self.alpha_position_successes.get(position, 0)
            failures = self.alpha_position_failures.get(position, 0)
            position_hat, position_ucb = self._bernstein_ucb(
                successes,
                failures,
                total_rounds,
            )
            self.alpha_position_hat[position] = position_hat
            self.alpha_position_ucb[position] = position_ucb

    def _apply_discount(self) -> None:
        discount = max(0.0, min(1.0, self.alpha_discount))
        self.alpha_successes *= discount
        self.alpha_failures *= discount
        for position in list(self.alpha_position_successes):
            discounted = self.alpha_position_successes[position] * discount
            if discounted <= 1e-9:
                self.alpha_position_successes.pop(position, None)
            else:
                self.alpha_position_successes[position] = discounted
        for position in list(self.alpha_position_failures):
            discounted = self.alpha_position_failures[position] * discount
            if discounted <= 1e-9:
                self.alpha_position_failures.pop(position, None)
            else:
                self.alpha_position_failures[position] = discounted

    def _reset_alpha_counts(self) -> None:
        self.alpha_successes = 0.0
        self.alpha_failures = 0.0
        self.alpha_position_successes.clear()
        self.alpha_position_failures.clear()

    def _accumulate_observation(
        self,
        accepted: int,
        inspected: int,
    ) -> None:
        self.alpha_successes += accepted
        for position in range(1, accepted + 1):
            self.alpha_position_successes[position] = (
                self.alpha_position_successes.get(position, 0.0) + 1.0
            )
        if accepted < inspected:
            self.alpha_failures += 1.0
            rejected_position = accepted + 1
            self.alpha_position_failures[rejected_position] = (
                self.alpha_position_failures.get(rejected_position, 0.0) + 1.0
            )

    def _recompute_window_counts(self) -> None:
        self._reset_alpha_counts()
        for accepted, inspected, _ in self.alpha_recent_events:
            self._accumulate_observation(accepted=accepted, inspected=inspected)

    def positional_alpha(
        self,
        position: int,
        use_ucb: bool,
    ) -> float:
        if position <= 0:
            raise ValueError("position must be positive")
        prior_mean = self._clip_alpha(self._alpha_prior_mean())
        table = self.alpha_position_ucb if use_ucb else self.alpha_position_hat
        if position in table:
            return table[position]
        return prior_mean

    def positional_acceptance_profile(
        self,
        gamma: int,
        use_ucb: bool,
    ) -> list[float]:
        if gamma <= 0:
            return []
        if self.alpha_profile_mode == "prefix":
            scalar_alpha = self.alpha_ucb if use_ucb else self.alpha_hat
            return [scalar_alpha for _ in range(gamma)]
        return [self.positional_alpha(position, use_ucb) for position in range(1, gamma + 1)]

    def observe_acceptance_feedback(
        self,
        proposed_tokens: int,
        accepted_length: int,
        inspected_tokens: int | None = None,
    ) -> None:
        if proposed_tokens <= 0:
            return
        inspected = proposed_tokens if inspected_tokens is None else int(inspected_tokens)
        inspected = max(0, min(inspected, proposed_tokens))
        accepted = max(0, min(int(accepted_length), inspected))

        self.alpha_observed_tokens += inspected
        rejected_within_observed = accepted < inspected
        if rejected_within_observed:
            self.alpha_failure_events += 1

        # All inspected proposal tokens succeeded, but no failure is observed after them.
        # This is a right-censored round rather than evidence about token gamma+1.
        if inspected == proposed_tokens:
            self.alpha_censored_rounds += 1

        if self._normalized_learning_mode() == "bernstein_censored_discounted":
            self._apply_discount()
            self._accumulate_observation(accepted=accepted, inspected=inspected)
            return

        if self._normalized_learning_mode() == "sliding_window":
            self.alpha_recent_events.append((accepted, inspected, rejected_within_observed))
            while self.alpha_window_size > 0 and len(self.alpha_recent_events) > self.alpha_window_size:
                self.alpha_recent_events.popleft()
            self._recompute_window_counts()
            return

        self._accumulate_observation(accepted=accepted, inspected=inspected)

    def start_next_sample(self) -> bool:
        if not self.prompt_queue:
            self.current_sample = None
            return False
        self.current_sample = self.prompt_queue.popleft()
        self.history_token_ids.clear()
        self.generated_for_sample = 0
        self.virtual_start_time_ms = self.virtual_time_ms
        self.sample_start_time_ms = self.virtual_time_ms
        return True

    @property
    def remaining_tokens(self) -> int:
        if self.current_sample is None:
            return 0
        return max(0, self.current_sample.max_new_tokens - self.generated_for_sample)


@dataclass
class DraftResult:
    client_id: int
    token_ids: list[int]
    draft_logprobs: list[Any]
    text: str
    latency_ms: float
    worker_id: int
    distribution_payload: Literal[
        "proposed_token_logprob",
        "delta_proposal",
        "full_vocab_logprobs",
    ] = "proposed_token_logprob"
    logprob_dtype_bytes: int = 4
    metadata_bytes: int = 64
    request_id_bytes: int = 16
    client_id_bytes: int = 8

    @property
    def log_q_proposed(self) -> list[float]:
        log_q: list[float] = []
        for step_index, (token_id, logprobs) in enumerate(zip(self.token_ids, self.draft_logprobs)):
            if self.distribution_payload == "delta_proposal":
                log_q.append(0.0)
                continue
            if self.distribution_payload == "proposed_token_logprob":
                value = logprobs
            else:
                value = logprobs[token_id]
            if hasattr(value, "logprob"):
                log_q.append(float(value.logprob))
            else:
                log_q.append(float(value))
        return log_q

    def distribution_dict(self, step_index: int, vocab_size: int) -> dict[int, float]:
        if step_index < 0 or step_index >= len(self.token_ids):
            raise IndexError(f"step_index {step_index} is out of range for gamma={len(self.token_ids)}")
        token_id = int(self.token_ids[step_index])
        if self.distribution_payload == "delta_proposal":
            return {token_id: 0.0}
        if step_index >= len(self.draft_logprobs):
            raise ValueError(
                "Draft proposal is missing per-step q payload required by the declared "
                f"distribution_payload={self.distribution_payload!r}."
            )
        step_payload = self.draft_logprobs[step_index]
        if self.distribution_payload == "proposed_token_logprob":
            if hasattr(step_payload, "logprob"):
                return {token_id: float(step_payload.logprob)}
            return {token_id: float(step_payload)}
        distribution: dict[int, float] = {}
        for support_token_id, value in enumerate(step_payload.tolist()):
            token_index = int(support_token_id)
            if token_index >= int(vocab_size):
                break
            float_value = float(value)
            if float_value != float("-inf"):
                distribution[token_index] = float_value
        return distribution

    def supports_lossless_rejection_sampling(self, vocab_size: int) -> bool:
        if len(self.token_ids) != len(self.draft_logprobs) and self.distribution_payload != "delta_proposal":
            return False
        if self.distribution_payload == "delta_proposal":
            return True
        if self.distribution_payload != "full_vocab_logprobs":
            return False
        required_vocab_size = int(vocab_size)
        for step_payload in self.draft_logprobs:
            shape = getattr(step_payload, "shape", None)
            if shape is None or not shape:
                return False
            if int(shape[-1]) < required_vocab_size:
                return False
        return True

    def estimate_uplink_bytes(self, vocab_size: int) -> int:
        candidate_bytes = len(self.token_ids) * 4
        if self.distribution_payload == "delta_proposal":
            logprob_bytes = 0
        elif self.distribution_payload == "proposed_token_logprob":
            logprob_bytes = len(self.token_ids) * self.logprob_dtype_bytes
        elif self.distribution_payload == "full_vocab_logprobs":
            logprob_bytes = len(self.token_ids) * int(vocab_size) * self.logprob_dtype_bytes
        else:
            raise ValueError(f"Unsupported distribution_payload={self.distribution_payload!r}")
        return (
            candidate_bytes
            + logprob_bytes
            + self.metadata_bytes
            + self.request_id_bytes
            + self.client_id_bytes
        )


DraftProposal = DraftResult


@dataclass
class VerificationResult:
    accepted_length: int
    committed_token_ids: list[int]
    committed_text: str
    target_latency_ms: float
    target_logprobs: list[Any] = field(default_factory=list)
    acceptance_probabilities: list[float] = field(default_factory=list)
    target_greedy_token_ids: list[int] = field(default_factory=list)
    reached_eos: bool = False
    finish_reason: str | None = None
    correction_token_id: int | None = None
    bonus_token_id: int | None = None

    @property
    def useful_token_count(self) -> int:
        """Committed tokens are the only useful tokens for system-level metrics."""
        return len(self.committed_token_ids)
