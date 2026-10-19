"""Per-client workload lifecycle and measurement-window exhaustion (audit items M1/M2).

Why this exists
---------------
The previous guard (`run_capacity_frontier.measure_candidate`) declared a client exhausted when
`max(receive) - min(receive)` over its IN-WINDOW rounds was shorter than 0.85 * T_m.  That quantity is
edge slack (about two round times), not workload exhaustion: it fires for every short window and misses
a client that runs out at >= 85 % of the window.

Definition used here (virtual time)
-----------------------------------
A client's lifecycle is the sequence pending -> active -> (draft -> uplink -> verify queue -> verify ->
downlink) per round.  Each round record carries

    client_samples_assigned     number of prompts assigned to the client at start-up
    sample_finished             this round completed the active sample (EOS, no progress, or budget reached)
    client_samples_pending      prompts not yet started AFTER this round
    client_work_complete        sample_finished and no pending prompt, i.e. nothing left to do

`completion_ms` is the virtual `client_receive_ms` of the round with `client_work_complete`.  A client is

    EXHAUSTED   iff its work completed strictly before `measurement_end_ms` (so it had no useful work, and
                no request in flight, during the tail of the window);
    ACTIVE      otherwise.  This includes (i) clients with sparse rounds, (ii) a round that is in flight at
                `measurement_end_ms` (draft_start <= end < receive), (iii) completion exactly at the
                boundary (receive == end is counted INSIDE the window by the simulator), and (iv) clients
                with zero in-window tokens that still have work.

A run with any EXHAUSTED client is INVALID for capacity measurement: its fixed-denominator rate is an
under-estimate of the sustained rate that the client could have delivered, which would be a false
INFEASIBLE.  INVALID is a separate outcome from INFEASIBLE and must be propagated, never classified.

Legacy round logs (before this change) lack the lifecycle columns; they are reported as UNKNOWN unless the
caller supplies `expected_samples_per_client`, in which case a clearly labelled inference is made.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

LIFECYCLE_COLUMNS = (
    "client_samples_assigned",
    "sample_finished",
    "client_samples_pending",
    "client_work_complete",
)

ACTIVE = "active"
EXHAUSTED = "exhausted"
NO_WORK = "no_work"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClientLifecycle:
    client_id: int
    status: str
    samples_assigned: int | None
    rounds_total: int
    completion_ms: float | None          # virtual time at which the last assigned sample finished
    last_event_ms: float | None          # last client_receive_ms of any round
    in_flight_at_end: bool               # a round straddles measurement_end_ms
    in_window_rounds: int
    in_window_tokens: int
    basis: str                           # "lifecycle_columns" | "inferred_from_rows" | "none"
    note: str = ""


@dataclass
class WorkloadAssessment:
    measurement_end_ms: float | None
    clients: list[ClientLifecycle] = field(default_factory=list)

    @property
    def exhausted_clients(self) -> list[int]:
        return [c.client_id for c in self.clients if c.status in (EXHAUSTED, NO_WORK)]

    @property
    def unknown_clients(self) -> list[int]:
        return [c.client_id for c in self.clients if c.status == UNKNOWN]

    @property
    def invalid(self) -> bool:
        """INVALID iff any client was exhausted (or had no work at all)."""
        return bool(self.exhausted_clients)

    @property
    def verifiable(self) -> bool:
        return not self.unknown_clients

    def to_dict(self) -> dict:
        return {
            "measurement_end_ms": self.measurement_end_ms,
            "invalid": self.invalid,
            "verifiable": self.verifiable,
            "exhausted_clients": self.exhausted_clients,
            "unknown_clients": self.unknown_clients,
            "clients": [asdict(c) for c in self.clients],
        }


def classify_client(
    *,
    client_id: int,
    rows: Sequence[dict],
    measurement_start_ms: float,
    measurement_end_ms: float,
    expected_samples: int | None = None,
) -> ClientLifecycle:
    """`rows`: the client's round records (any order) with at least client_receive_ms, draft_start_ms and
    committed_token_count; lifecycle columns when available."""
    ordered = sorted(rows, key=lambda r: float(r["client_receive_ms"]))
    in_window = [
        r for r in ordered
        if measurement_start_ms <= float(r["client_receive_ms"]) <= measurement_end_ms
    ]
    tokens = int(sum(int(r.get("committed_token_count", 0) or 0) for r in in_window))
    last_event = float(ordered[-1]["client_receive_ms"]) if ordered else None
    straddling = any(
        float(r["draft_start_ms"]) <= measurement_end_ms < float(r["client_receive_ms"])
        for r in ordered
    )

    have_cols = bool(ordered) and all(c in ordered[0] for c in LIFECYCLE_COLUMNS)
    assigned: int | None = None
    completion: float | None = None
    basis = "none"
    if have_cols:
        basis = "lifecycle_columns"
        assigned = int(ordered[0]["client_samples_assigned"])
        done = [r for r in ordered if bool(r["client_work_complete"])]
        completion = float(done[0]["client_receive_ms"]) if done else None
    elif ordered and expected_samples is not None:
        basis = "inferred_from_rows"
        assigned = int(expected_samples)
        finished = [r for r in ordered if _row_finished_sample(r)]
        if len(finished) >= assigned:
            completion = float(finished[assigned - 1]["client_receive_ms"])
    elif not ordered and expected_samples is not None:
        assigned = int(expected_samples)
        basis = "inferred_from_rows"

    if assigned is not None and assigned <= 0:
        return ClientLifecycle(client_id, NO_WORK, assigned, len(ordered), None, last_event, False,
                               len(in_window), tokens, basis, "client was assigned no samples")
    if basis == "none":
        return ClientLifecycle(client_id, UNKNOWN, None, len(ordered), None, last_event, straddling,
                               len(in_window), tokens, basis,
                               "legacy rows without lifecycle columns and no expected_samples given")
    if completion is not None and completion < measurement_end_ms:
        status, note = EXHAUSTED, "all assigned work completed before measurement_end"
    else:
        status = ACTIVE
        note = ("round in flight at measurement_end" if straddling
                else "work remains (or completes at/after measurement_end)")
    return ClientLifecycle(client_id, status, assigned, len(ordered), completion, last_event, straddling,
                           len(in_window), tokens, basis, note)


def _row_finished_sample(r: dict) -> bool:
    if bool(r.get("reached_eos", False)):
        return True
    gen, cap = r.get("generated_for_sample"), r.get("max_new_tokens")
    return gen is not None and cap is not None and int(gen) >= int(cap)


def assess_workload(
    rows_by_client: dict[int, Sequence[dict]],
    *,
    admitted_client_ids: Iterable[int],
    measurement_start_ms: float,
    measurement_end_ms: float | None,
    expected_samples_per_client: int | None = None,
) -> WorkloadAssessment:
    """Assess every ADMITTED client (a client with no rows is still assessed)."""
    if measurement_end_ms is None:
        return WorkloadAssessment(None, [])
    out = WorkloadAssessment(float(measurement_end_ms))
    for cid in admitted_client_ids:
        out.clients.append(
            classify_client(
                client_id=int(cid),
                rows=list(rows_by_client.get(int(cid), [])),
                measurement_start_ms=float(measurement_start_ms),
                measurement_end_ms=float(measurement_end_ms),
                expected_samples=expected_samples_per_client,
            )
        )
    return out


def assess_from_dataframe(df, *, admitted_client_ids: Iterable[int], measurement_start_ms: float,
                          measurement_end_ms: float | None,
                          expected_samples_per_client: int | None = None) -> WorkloadAssessment:
    """Convenience wrapper over a per-round DataFrame (ALL rows, not only in-window ones)."""
    rows_by_client: dict[int, list[dict]] = {}
    for rec in df.to_dict("records"):
        rows_by_client.setdefault(int(rec["client_id"]), []).append(rec)
    return assess_workload(
        rows_by_client,
        admitted_client_ids=admitted_client_ids,
        measurement_start_ms=measurement_start_ms,
        measurement_end_ms=measurement_end_ms,
        expected_samples_per_client=expected_samples_per_client,
    )
