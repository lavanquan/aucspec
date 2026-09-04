"""auc_controller.py with UCB/Bernstein learning replaced by a static offline
acceptance-rate estimate alpha_hat_i, profiled once before the run.

Not a direct Px solver -- same controller architecture as auc_controller.py,
just alpha_hat_i plugged in as a constant instead of learned online. Profile
alpha_hat_i from the first half of the trace only and evaluate on the second
half to avoid data leakage. See TASKS.md T2.3.

Implementation note: TASKS.md T2.3 says to "copy nguyên auc_controller.py"
(copy it outright) and swap the estimator. We instead reuse AUCController
directly and only swap in StaticAlphaEstimator below -- functionally
identical (same decide()/update_queues() code path, same eq. 12-13 logic
runs either way) but without maintaining two literal copies of the
controller that could silently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from .auc_controller import AUCController


@dataclass
class StaticAlphaEstimator:
    """alpha_hat_i profiled once offline (T2.3), constant over t -- Theorem
    4's UCB/Bernstein learning is disabled entirely."""

    alpha_hat: list[float]

    def estimate(self, stream_index: int, t: int) -> float:
        return self.alpha_hat[stream_index]


def profile_alpha_from_trace(
    accept_flags_by_stream: list[list[bool]], split_fraction: float = 0.5
) -> list[float]:
    """Estimate alpha_hat_i from the FIRST split_fraction of each stream's
    trace only (configs/base.yaml's oracle.profile_split_fraction); the
    remaining rounds are left for evaluation to avoid data leakage.
    """
    if not (0.0 < split_fraction < 1.0):
        raise ValueError("split_fraction must be in (0, 1)")
    estimates = []
    for flags in accept_flags_by_stream:
        cutoff = max(1, int(len(flags) * split_fraction))
        profile_window = flags[:cutoff]
        if not profile_window:
            raise ValueError("profile window is empty")
        estimates.append(sum(1 for f in profile_window if f) / len(profile_window))
    return estimates


def build_oracle_controller(alpha_hat: list[float], **controller_kwargs) -> AUCController:
    """Construct an AUCController with StaticAlphaEstimator(alpha_hat) in
    place of UCBBernsteinEstimator -- same controller, offline-profiled
    alpha instead of learned alpha."""
    return AUCController(alpha_estimator=StaticAlphaEstimator(alpha_hat), **controller_kwargs)
