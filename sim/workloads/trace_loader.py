"""Load chat/code mix traffic traces; measure alpha_i(t) over a sliding window.

Backs the heterogeneous-case experiments (Exp 2b) and the misspecification
check (Assumption 1 robustness). See TASKS.md T4.1-T4.3.

Trace file format (JSON Lines, one line per round per stream):
    {"stream_id": "chat_0", "domain": "chat", "round_index": 0,
     "gamma": 4, "accepted": 3, "all_accepted": false}

`accepted` is the truncated-geometric accepted-prefix count (Assumption 1,
Theorem 4's censored observation: `accepted` Bernoulli successes, plus one
observed failure unless `all_accepted` is true). This is exactly what
core.acceptance.sample_round returns, so a real accept/reject logger only
has to write out its (tokens_delivered - 1, all_accepted) pair per round.

As of this commit, no real accept/reject trace has been collected yet --
that requires running Qwen2.5 draft/target models on GPU hardware this
session does not have (see docs/model_choices.md's ROCm/CUDA note). This
module and its tests are validated against synthetic traces
(sim/workloads/synthetic_trace.py), clearly labeled as such; swap in a real
JSONL file (same schema) once T4.2's data collection actually runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..core.acceptance import TraceAcceptance, sliding_window_alpha


@dataclass(frozen=True)
class TraceRecord:
    stream_id: str
    domain: str
    round_index: int
    gamma: int
    accepted: int
    all_accepted: bool


def load_trace_jsonl(path: str | Path) -> list[TraceRecord]:
    """Read a trace file in the JSON Lines schema documented above."""
    records: list[TraceRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                TraceRecord(
                    stream_id=str(obj["stream_id"]),
                    domain=str(obj["domain"]),
                    round_index=int(obj["round_index"]),
                    gamma=int(obj["gamma"]),
                    accepted=int(obj["accepted"]),
                    all_accepted=bool(obj["all_accepted"]),
                )
            )
    return records


def group_by_stream(records: list[TraceRecord]) -> dict[str, list[TraceRecord]]:
    """Split records by stream_id, each sorted by round_index ascending."""
    grouped: dict[str, list[TraceRecord]] = {}
    for record in records:
        grouped.setdefault(record.stream_id, []).append(record)
    for stream_records in grouped.values():
        stream_records.sort(key=lambda r: r.round_index)
    return grouped


def token_accept_flags(records: list[TraceRecord]) -> list[bool]:
    """Expand a stream's per-round (accepted, all_accepted) records into the
    token-level accept/reject sequence Theorem 4's censored observation
    model describes: `accepted` True values, plus one False unless the round
    was fully accepted (right-censored)."""
    flags: list[bool] = []
    for record in records:
        if record.accepted < 0:
            raise ValueError("accepted must be non-negative")
        flags.extend([True] * record.accepted)
        if not record.all_accepted:
            flags.append(False)
    return flags


def token_level_alpha_series(records: list[TraceRecord], window: int) -> list[float]:
    """alpha_i(t) measured over a trailing sliding window of `window`
    token-level accept/reject observations (EXPERIMENTS.md Hinh 5a)."""
    return sliding_window_alpha(token_accept_flags(records), window)


def round_level_alpha_series(records: list[TraceRecord], window_rounds: int) -> list[float]:
    """alpha_i(t) measured over a trailing window of `window_rounds` ROUNDS
    (not tokens): for round k, pool every token-level observation from
    rounds [k-window_rounds+1, k] and take the empirical mean. Coarser than
    token_level_alpha_series but round-indexed, which is what
    core.acceptance.TraceAcceptance needs.
    """
    if window_rounds <= 0:
        raise ValueError("window_rounds must be positive")
    series: list[float] = []
    for k in range(len(records)):
        start = max(0, k + 1 - window_rounds)
        window_flags = token_accept_flags(records[start : k + 1])
        series.append(sum(1 for f in window_flags if f) / len(window_flags) if window_flags else 0.5)
    return series


def build_trace_acceptance(records: list[TraceRecord], window_rounds: int, seed: int = 0) -> TraceAcceptance:
    """Turn one stream's records into a core.acceptance.TraceAcceptance
    (trace-driven alpha(t) mode, sim/core/acceptance.py)."""
    alpha_by_round = round_level_alpha_series(records, window_rounds)
    return TraceAcceptance(alpha_by_round=alpha_by_round, seed=seed, window=window_rounds)


def lag1_autocorrelation(flags: list[bool]) -> float:
    """Pearson autocorrelation at lag 1 of the 0/1 accept/reject sequence --
    evidence for or against Assumption 1's i.i.d. hypothesis
    (EXPERIMENTS.md Exp misspecification, Buoc 1). A value close to 0
    supports i.i.d.; a materially positive or negative value is evidence of
    violation (e.g. bursty acceptance/rejection, or an alternating pattern).
    Returns 0.0 if there are fewer than 2 observations or the sequence is
    constant (zero variance -- autocorrelation is undefined, not evidence
    either way).
    """
    if len(flags) < 2:
        return 0.0
    values = [1.0 if f else 0.0 for f in flags]
    x = values[:-1]
    y = values[1:]
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    denom = (var_x * var_y) ** 0.5
    if denom == 0.0:
        return 0.0
    return cov / denom


def domain_alpha_summary(records: list[TraceRecord]) -> dict[str, float]:
    """Overall (non-windowed) empirical acceptance rate per domain, pooling
    every stream that shares that domain -- a quick chat-vs-code comparison
    ahead of the full sliding-window plot."""
    flags_by_domain: dict[str, list[bool]] = {}
    for record in records:
        flags_by_domain.setdefault(record.domain, []).extend(token_accept_flags([record]))
    return {
        domain: (sum(1 for f in flags if f) / len(flags) if flags else 0.0)
        for domain, flags in flags_by_domain.items()
    }
