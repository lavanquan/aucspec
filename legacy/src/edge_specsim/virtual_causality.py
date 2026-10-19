"""Conservative virtual-event coordination for the hybrid real/virtual simulator (audit items D1/D2).

See docs/VIRTUAL_TIME_EVENT_GRAPH.md for the event graph and the argument.  Summary:

* every client carries a LOWER BOUND on the virtual timestamp of its next request arrival;
* a verification batch with cutoff C is formed only when every client that has no request queued has a lower
  bound strictly greater than C (so no request with arrival <= C can still appear);
* a depth decision (or request stamp) at virtual time t may proceed only when the verifier horizon H > t, i.e.
  no batch that finishes at or before t is still unprocessed; it then reads the server-state snapshot of the
  latest batch with finish <= t.

No sleeps and no real-time delays are used: waiters are released by state changes only.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Callable

EPS = 1e-9


class CausalityViolation(RuntimeError):
    """A producer violated its declared lookahead (an event earlier than a promised lower bound)."""


class CausalityDeadlock(RuntimeError):
    """Every active participant is blocked on the coordinator; no event can ever advance the horizon."""


@dataclass
class LedgerEvent:
    seq: int
    virtual_ts: float
    type: str                     # arrival | batch_formed | batch_complete | decision | retire
    producer: str                 # client:<i> | verifier
    decision_ts: float | None     # timestamp at which a decision/formation was taken
    visible_update_ts: float | None  # latest server-state update visible to that decision
    queue_state_version: int      # number of completed batches visible (virtual queue state version)
    detail: dict = field(default_factory=dict)


class EventLedger:
    def __init__(self) -> None:
        self.events: list[LedgerEvent] = []

    def add(self, virtual_ts, type_, producer, decision_ts=None, visible_update_ts=None, version=0, **detail) -> None:
        self.events.append(
            LedgerEvent(len(self.events), float(virtual_ts), type_, producer,
                        None if decision_ts is None else float(decision_ts),
                        None if visible_update_ts is None else float(visible_update_ts),
                        int(version), dict(detail))
        )

    def to_records(self) -> list[dict]:
        return [dict(seq=e.seq, virtual_ts=e.virtual_ts, type=e.type, producer=e.producer,
                     decision_ts=e.decision_ts, visible_update_ts=e.visible_update_ts,
                     queue_state_version=e.queue_state_version, **e.detail) for e in self.events]

    def check_causality(self) -> list[str]:
        """Return the violated invariants (empty list = causally consistent).

        D1: a batch_formed event with cutoff C must not be followed (in real/recording order) by an arrival
            whose virtual timestamp is <= C -- that request had already arrived in virtual time.
        D2: a decision at t must not have seen a server-state update with timestamp > t.
        Also: verifier service intervals do not overlap; no request is served twice or before it arrived.
        """
        bad: list[str] = []
        arrival_ts: dict[tuple, float] = {}
        arrival_seq: dict[tuple, int] = {}
        served: set[tuple] = set()
        last_finish = -math.inf
        formed = [e for e in self.events if e.type == "batch_formed"]
        for e in self.events:
            if e.type == "arrival":
                key = (e.detail["client"], e.detail["request"])
                if key in arrival_ts:
                    bad.append(f"duplicate arrival {key}")
                arrival_ts[key], arrival_seq[key] = e.virtual_ts, e.seq
                for f in formed:
                    if f.seq < e.seq and e.virtual_ts <= f.detail["cutoff_ms"] + EPS:
                        bad.append(f"D1 seq {f.seq}: batch cutoff {f.detail['cutoff_ms']:.3f} formed before "
                                   f"arrival {key} at {e.virtual_ts:.3f} was recorded (seq {e.seq})")
            elif e.type == "decision":
                if e.visible_update_ts is not None and e.visible_update_ts > e.decision_ts + EPS:
                    bad.append(f"D2 seq {e.seq}: decision at {e.decision_ts:.3f} saw state update at "
                               f"{e.visible_update_ts:.3f}")
            elif e.type == "batch_formed":
                for m in e.detail["members"]:
                    key = tuple(m)
                    if key in served:
                        bad.append(f"seq {e.seq}: request {key} served twice")
                    served.add(key)
                    if key not in arrival_ts:
                        bad.append(f"seq {e.seq}: request {key} served before its arrival was recorded")
                    elif arrival_ts[key] > e.detail["cutoff_ms"] + EPS:
                        bad.append(f"seq {e.seq}: request {key} arrives after the cutoff of its batch")
            elif e.type == "batch_complete":
                start, finish = e.detail["start_ms"], e.detail["finish_ms"]
                if start + EPS < last_finish:
                    bad.append(f"seq {e.seq}: verifier service [{start},{finish}] overlaps previous finish {last_finish}")
                last_finish = max(last_finish, finish)
        dropped = set(arrival_ts) - served
        # a request still queued when the run ends is a dropped request
        if dropped:
            bad.append(f"requests never served: {sorted(dropped)}")
        return bad


@dataclass
class _ClientState:
    rtt_ms: float
    lb: float
    status: str = "idle"   # idle | queued | service | retired


class VirtualCoordinator:
    def __init__(self, batch_wait_ms: float, ledger: EventLedger | None = None) -> None:
        self.batch_wait_ms = float(batch_wait_ms)
        self.ledger = ledger if ledger is not None else EventLedger()
        self.clients: dict[int, _ClientState] = {}
        self.queued: dict[int, float] = {}
        self.inflight_start: float | None = None
        self.server_available_at = 0.0
        self.state_version = 0
        self.last_update_ts = 0.0
        self._waiters: list[asyncio.Future] = []
        self._decision_waiting: set[int] = set()
        self._batcher_waiting = False
        self._head_cutoff_fn: Callable[[], float | None] | None = None

    # ------------------------------------------------------------------ registry
    def register_client(self, client_id: int, rtt_ms: float, start_ms: float = 0.0) -> None:
        if rtt_ms <= 0.0 and self.batch_wait_ms <= 0.0:
            raise ValueError("causal virtual time needs lookahead: rtt_ms > 0 or batch_wait_ms > 0")
        self.clients[int(client_id)] = _ClientState(float(rtt_ms), float(start_ms) + float(rtt_ms) / 2.0)
        self._poke()

    def retire_client(self, client_id: int) -> None:
        st = self.clients[int(client_id)]
        st.status, st.lb = "retired", math.inf
        self.ledger.add(self.last_update_ts, "retire", f"client:{client_id}")
        self._poke()

    # ------------------------------------------------------------------ producer side
    def begin_round(self, client_id: int, t: float) -> None:
        """Client resumes at virtual time t: its next arrival cannot precede t + RTT/2."""
        st = self.clients[int(client_id)]
        st.lb = max(st.lb, float(t) + st.rtt_ms / 2.0)
        self._poke()

    def note_submit(self, client_id: int, request_id, arrival_ms: float) -> None:
        st = self.clients[int(client_id)]
        if st.status != "idle":
            raise CausalityViolation(f"client {client_id} submitted while {st.status}")
        if arrival_ms + EPS < st.lb:
            raise CausalityViolation(
                f"client {client_id} arrival {arrival_ms} earlier than its declared lower bound {st.lb}")
        st.status = "queued"
        self.queued[int(client_id)] = float(arrival_ms)
        self.ledger.add(arrival_ms, "arrival", f"client:{client_id}", client=int(client_id), request=request_id)
        self._poke()

    # ------------------------------------------------------------------ verifier side
    def cutoff_satisfied(self, cutoff_ms: float) -> bool:
        return all(st.lb > cutoff_ms for st in self.clients.values() if st.status == "idle")

    def min_idle_lb(self) -> float:
        return min((st.lb for st in self.clients.values() if st.status == "idle"), default=math.inf)

    def horizon(self) -> float:
        if self.inflight_start is not None:
            return self.inflight_start
        cands = []
        if self.queued:
            cands.append(min(self.queued.values()))
        idle = self.min_idle_lb()
        if idle < math.inf:
            cands.append(idle)
        if not cands:
            return math.inf
        return max(self.server_available_at, min(cands) + self.batch_wait_ms)

    def batch_formed(self, batch_id: int, cutoff_ms: float, members: list[tuple], start_ms: float) -> None:
        for cid, _ in members:
            st = self.clients[int(cid)]
            st.status = "service"
            self.queued.pop(int(cid), None)
        self.inflight_start = float(start_ms)
        self.ledger.add(start_ms, "batch_formed", "verifier", decision_ts=cutoff_ms,
                        visible_update_ts=self.last_update_ts, version=self.state_version,
                        batch=batch_id, cutoff_ms=float(cutoff_ms), members=[list(m) for m in members],
                        min_idle_lb=self.min_idle_lb())
        self._poke()

    def batch_done(self, batch_id: int, members: list[tuple], start_ms: float, finish_ms: float) -> None:
        for cid, _ in members:
            st = self.clients[int(cid)]
            st.status = "idle"
            st.lb = max(st.lb, float(finish_ms) + st.rtt_ms)
        self.inflight_start = None
        self.server_available_at = float(finish_ms)
        self.state_version += 1
        self.last_update_ts = float(finish_ms)
        self.ledger.add(finish_ms, "batch_complete", "verifier", version=self.state_version,
                        batch=batch_id, start_ms=float(start_ms), finish_ms=float(finish_ms),
                        members=[list(m) for m in members])
        self._poke()

    # ------------------------------------------------------------------ waiting
    def _poke(self) -> None:
        waiters, self._waiters = self._waiters, []
        for fut in waiters:
            if not fut.done():
                fut.set_result(None)

    async def _wait_until(self, predicate: Callable[[], bool]) -> None:
        loop = asyncio.get_running_loop()
        while not predicate():
            self._check_deadlock()
            fut = loop.create_future()
            self._waiters.append(fut)
            await fut

    async def wait_for_cutoff(self, ready: Callable[[], bool]) -> None:
        """Batcher side: `ready()` re-reads the queue head and checks `cutoff_satisfied`."""
        self._batcher_waiting = True
        try:
            await self._wait_until(ready)
        finally:
            self._batcher_waiting = False

    async def wait_decision(self, client_id: int, t: float) -> None:
        """Client side: block until no batch finishing at or before t can still be unprocessed (H > t)."""
        self.begin_round(client_id, t)
        self._decision_waiting.add(int(client_id))
        try:
            await self._wait_until(lambda: self.horizon() > t)
        finally:
            self._decision_waiting.discard(int(client_id))

    def record_decision(self, client_id: int, t: float, visible_update_ts: float, version: int, **detail) -> None:
        self.ledger.add(t, "decision", f"client:{client_id}", decision_ts=t,
                        visible_update_ts=visible_update_ts, version=version, client=int(client_id), **detail)

    def _check_deadlock(self) -> None:
        if self.inflight_start is not None:
            return
        for cid, st in self.clients.items():
            if st.status == "retired":
                continue
            if st.status == "idle" and cid not in self._decision_waiting:
                return  # somebody is still computing in real time -> progress is possible
        # nobody can make progress except through the coordinator
        if self.queued and self._head_cutoff_fn is not None:
            cutoff = self._head_cutoff_fn()
            if cutoff is not None and self.cutoff_satisfied(cutoff):
                return  # batcher can form a batch
        if not self._decision_waiting and not self._batcher_waiting:
            return
        raise CausalityDeadlock(
            f"all active clients blocked: decision_waiting={sorted(self._decision_waiting)} "
            f"queued={dict(self.queued)} horizon={self.horizon()} min_idle_lb={self.min_idle_lb()}")
