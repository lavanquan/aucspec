"""Fixed multi-SLO baseline (simplified AdaServe-style, EuroSys'26).

Each request has its own latency target; gamma chosen to meet that target.
Not a full hardware-aware speculation-tree construction. See TASKS.md T3.4.

Per TASKS.md T3.4: "khong can implement day du hardware-aware tree
construction ... chi can dung y tuong 'moi request co latency target rieng,
chon gamma theo target do'". This controller does exactly that and nothing
more: for each stream i, pick the largest gamma_i in {0,...,gamma_max} whose
expected round cycle time (eq. 4's denominator, reusing
core.device.round_cycle_seconds) still meets that stream's own SLO
slo_seconds[i], independently of every other stream. There is no
speculation-tree, no shared pricing, and no interactivity or energy
objective -- purely a per-stream deadline check.
"""

from __future__ import annotations

from typing import Sequence

from .auc_controller import SpeculationDecision, StreamLinkState
from ..core.device import round_cycle_seconds
from ..core.server import RooflineParams, verification_latency


class FixedSLOController:
    """Per-stream latency SLOs slo_seconds[i]: choose the largest gamma_i
    whose expected round cycle time still meets slo_seconds[i]."""

    def __init__(
        self,
        n_streams: int,
        gamma_max: int,
        slo_seconds: Sequence[float],
        kappa_bits_per_token: float,
        delta_seconds: float,
        roofline: RooflineParams,
        gamma_budget: float,
        total_bandwidth_hz: float,
        context_length: float = 0.0,
    ) -> None:
        if len(slo_seconds) != n_streams:
            raise ValueError("slo_seconds must have one entry per stream")
        if n_streams <= 0:
            raise ValueError("n_streams must be positive")
        self.n_streams = n_streams
        self.gamma_max = gamma_max
        self.slo_seconds = list(slo_seconds)
        self.kappa_bits_per_token = kappa_bits_per_token
        self.delta_seconds = delta_seconds
        self.roofline = roofline
        self.gamma_budget = gamma_budget
        self.total_bandwidth_hz = total_bandwidth_hz
        self.context_length = context_length

    def _largest_gamma_meeting_slo(self, link: StreamLinkState, slo: float) -> int:
        for gamma in range(self.gamma_max, -1, -1):
            t_v = verification_latency(self.roofline, [gamma], [self.context_length])
            cycle = round_cycle_seconds(
                gamma,
                link.tau_d_seconds,
                self.kappa_bits_per_token,
                link.uplink_rate_bps,
                queueing_delay_seconds=0.0,
                verification_latency_seconds=t_v,
                delta_seconds=self.delta_seconds,
            )
            if cycle <= slo:
                return gamma
        return 0  # not even gamma=0 meets the SLO; still serve, just as fast as possible

    def decide(self, t: int, links: Sequence[StreamLinkState]) -> SpeculationDecision:
        if len(links) != self.n_streams:
            raise ValueError("links must have one entry per stream")
        gammas = [
            self._largest_gamma_meeting_slo(link, self.slo_seconds[i]) for i, link in enumerate(links)
        ]

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
