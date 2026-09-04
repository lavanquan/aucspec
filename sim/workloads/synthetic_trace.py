"""Synthetic accept/reject traces for exercising trace_loader.py end-to-end
before a real Qwen2.5 trace exists (docs/model_choices.md; TASKS.md T4.2's
real data collection needs GPU hardware this repo does not have wired up
yet). NOT real data -- every value here is drawn from a fixed random model,
labeled as such, and must never be reported as a measured acceptance rate.

Two generators are provided as positive/negative controls for the
misspecification check (EXPERIMENTS.md Exp misspecification):
  - generate_stationary_trace: true i.i.d. Bernoulli(alpha) per Assumption 1
    -- lag-1 autocorrelation should be ~0.
  - generate_drifting_trace: alpha(t) walks over time (a stand-in for
    real chat-vs-code non-stationarity) -- lag-1 autocorrelation should be
    detectably nonzero.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..core.acceptance import sample_round
from .trace_loader import TraceRecord


def generate_stationary_trace(
    stream_id: str,
    domain: str,
    alpha: float,
    num_rounds: int,
    gamma: int,
    seed: int,
) -> list[TraceRecord]:
    """i.i.d. Bernoulli(alpha) every round -- Assumption 1 holds by
    construction. Positive control: a misspecification check run on this
    trace should find near-zero autocorrelation."""
    rng = random.Random(seed)
    records = []
    for round_index in range(num_rounds):
        tokens, all_accepted = sample_round(gamma, alpha, rng)
        records.append(
            TraceRecord(
                stream_id=stream_id,
                domain=domain,
                round_index=round_index,
                gamma=gamma,
                accepted=tokens - 1,
                all_accepted=all_accepted,
            )
        )
    return records


def generate_drifting_trace(
    stream_id: str,
    domain: str,
    alpha_start: float,
    alpha_end: float,
    num_rounds: int,
    gamma: int,
    seed: int,
) -> list[TraceRecord]:
    """alpha(t) drifts linearly from alpha_start to alpha_end over the
    trace -- Assumption 1 (i.i.d. over time) is violated by construction.
    Negative control: a misspecification check run on this trace should
    detect the violation (nonzero autocorrelation, or the sliding-window
    alpha(t) plot visibly trending)."""
    rng = random.Random(seed)
    records = []
    for round_index in range(num_rounds):
        progress = round_index / max(1, num_rounds - 1)
        alpha_t = alpha_start + progress * (alpha_end - alpha_start)
        tokens, all_accepted = sample_round(gamma, alpha_t, rng)
        records.append(
            TraceRecord(
                stream_id=stream_id,
                domain=domain,
                round_index=round_index,
                gamma=gamma,
                accepted=tokens - 1,
                all_accepted=all_accepted,
            )
        )
    return records


def write_trace_jsonl(records: list[TraceRecord], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    {
                        "stream_id": record.stream_id,
                        "domain": record.domain,
                        "round_index": record.round_index,
                        "gamma": record.gamma,
                        "accepted": record.accepted,
                        "all_accepted": record.all_accepted,
                    }
                )
                + "\n"
            )
