from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque


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
    rtt_ms: float
    uplink_mbps: float
    draft_speed_multiplier: float
    gamma: int = 1
    z_queue: float = 0.0
    device_queue: float = 0.0
    alpha_successes: int = 1
    alpha_failures: int = 1
    accepted_tokens: int = 0
    useful_tokens: int = 0
    rounds_completed: int = 0
    samples_completed: int = 0
    current_sample: PromptSample | None = None
    history_token_ids: list[int] = field(default_factory=list)
    generated_for_sample: int = 0

    @property
    def alpha_hat(self) -> float:
        return self.alpha_successes / (self.alpha_successes + self.alpha_failures)

    def alpha_ucb(self, total_rounds: int) -> float:
        import math

        n = max(1, self.alpha_successes + self.alpha_failures)
        bonus = math.sqrt(2.0 * math.log(max(2, total_rounds + 1)) / n)
        return min(0.999, self.alpha_hat + bonus)

    def start_next_sample(self) -> bool:
        if not self.prompt_queue:
            self.current_sample = None
            return False
        self.current_sample = self.prompt_queue.popleft()
        self.history_token_ids.clear()
        self.generated_for_sample = 0
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

    @property
    def log_q_proposed(self) -> list[float]:
        log_q: list[float] = []
        for token_id, logprobs in zip(self.token_ids, self.draft_logprobs):
            value = logprobs[token_id]
            if hasattr(value, "logprob"):
                log_q.append(float(value.logprob))
            else:
                log_q.append(float(value))
        return log_q


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
