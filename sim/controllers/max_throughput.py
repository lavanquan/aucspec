"""Max-throughput baseline: largest batch, small fixed gamma (datacenter-style).

See TASKS.md T3.1. A single fixed operating point on the frontier
(Proposition 1's degenerate w = delta_{x0} case): no per-stream pricing, no
interactivity target -- just admit the largest batch the token budget allows
at a small, fixed speculation length, mirroring datacenter SD serving that
maximizes aggregate throughput without per-user fairness.
"""

from __future__ import annotations

from typing import Sequence

from .auc_controller import SpeculationDecision, StreamLinkState


class MaxThroughputController:
    """Fixed small gamma for every stream; batch admission is greedy by
    stream index up to the token budget (largest batch, no per-stream
    pricing); bandwidth is split evenly among admitted streams (throughput
    only -- no fairness or latency objective to waterfill against)."""

    def __init__(
        self,
        n_streams: int,
        fixed_gamma: int,
        gamma_budget: float,
        total_bandwidth_hz: float,
    ) -> None:
        if fixed_gamma < 0:
            raise ValueError("fixed_gamma must be non-negative")
        if n_streams <= 0:
            raise ValueError("n_streams must be positive")
        self.n_streams = n_streams
        self.fixed_gamma = fixed_gamma
        self.gamma_budget = gamma_budget
        self.total_bandwidth_hz = total_bandwidth_hz

    def decide(self, t: int, links: Sequence[StreamLinkState]) -> SpeculationDecision:
        if len(links) != self.n_streams:
            raise ValueError("links must have one entry per stream")

        gammas = [self.fixed_gamma] * self.n_streams
        selected: list[int] = []
        used = 0.0
        for i in range(self.n_streams):
            cost = gammas[i] + 1
            if used + cost <= self.gamma_budget:
                selected.append(i)
                used += cost

        bandwidth = [0.0] * self.n_streams
        if selected:
            share = self.total_bandwidth_hz / len(selected)
            for i in selected:
                bandwidth[i] = share

        return SpeculationDecision(
            gammas=gammas,
            selected_batch=selected,
            bandwidth_allocation_hz=bandwidth,
            lambda_price=0.0,
            mu_prices=[0.0] * self.n_streams,
        )
