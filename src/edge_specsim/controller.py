from __future__ import annotations

import math

from .models import ClientProfile


class OnlineController:
    def __init__(self, gamma_max: int, v: float, min_tps: float, server_price_scale: float) -> None:
        self.gamma_max = gamma_max
        self.v = v
        self.min_tps = min_tps
        self.server_price_scale = server_price_scale
        self.server_queue = 0.0

    def choose_gamma(self, client: ClientProfile, total_rounds: int) -> int:
        alpha = max(1e-3, min(0.999, client.alpha_ucb(total_rounds)))
        benefit_weight = self.v + client.z_queue
        cost = 1.0 + self.server_price_scale * self.server_queue + 0.01 * client.device_queue

        gamma = 0
        for candidate in range(1, self.gamma_max + 1):
            marginal_benefit = benefit_weight * (alpha ** candidate)
            if marginal_benefit >= cost:
                gamma = candidate
            else:
                break
        return gamma

    def update(
        self,
        client: ClientProfile,
        useful_tokens: int,
        round_latency_ms: float,
        draft_latency_ms: float,
        target_latency_ms: float,
    ) -> None:
        seconds = max(1e-6, round_latency_ms / 1000.0)
        observed_tps = useful_tokens / seconds
        client.z_queue = max(0.0, client.z_queue + self.min_tps - observed_tps)
        client.device_queue = max(0.0, client.device_queue + draft_latency_ms - round_latency_ms)
        self.server_queue = max(0.0, self.server_queue + target_latency_ms - round_latency_ms)
