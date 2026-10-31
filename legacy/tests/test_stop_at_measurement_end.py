"""A client stops once its virtual clock passes the measurement window, so a
capacity candidate costs warmup + measurement instead of every client's whole
questions_per_client workload."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest

for _name, _attrs in (
    ("edge_specsim.target_client", {"VLLMCandidateVerifier": object}),
    ("edge_specsim.draft_worker", {"DraftRequest": object, "DraftWorker": object}),
):
    if importlib.util.find_spec("torch") is None and _name not in sys.modules:
        _stub = types.ModuleType(_name)
        for _k, _v in _attrs.items():
            setattr(_stub, _k, _v)
        sys.modules[_name] = _stub

from edge_specsim.simulator import EdgeSpecSimulator  # noqa: E402


class _Client:
    def __init__(self, samples: int, tokens: int) -> None:
        self.client_id = 0
        self.virtual_time_ms = 0.0
        self.queue = [SimpleNamespace(sample_id=i) for i in range(samples)]
        self.current_sample = None
        self.tokens = tokens
        self.generated_for_sample = 0
        self.samples_completed = 0

    def start_next_sample(self) -> bool:
        if not self.queue:
            return False
        self.current_sample = self.queue.pop(0)
        self.generated_for_sample = 0
        return True

    @property
    def remaining_tokens(self) -> int:
        return self.tokens - self.generated_for_sample


class _Worker:
    async def reset_client(self, client_id: int) -> None:
        pass


def _sim(stop: bool, end_ms: float | None):
    sim = EdgeSpecSimulator.__new__(EdgeSpecSimulator)
    sim.stop_at_measurement_end = stop
    sim.measurement_end_ms = end_ms
    sim.workers = [_Worker()]
    sim._done_clients = set()
    sim.verifier = SimpleNamespace(notify=lambda: None)
    sim.target = SimpleNamespace(set_confirmed_prefix=lambda *a: None)
    starts: list[float] = []

    async def _run_round(client, round_id):
        starts.append(client.virtual_time_ms)
        client.virtual_time_ms += 100.0
        client.generated_for_sample += 1
        return False

    sim._run_round = _run_round
    return sim, starts


@pytest.mark.parametrize("stop", [False, True])
def test_client_stops_after_window(stop):
    sim, starts = _sim(stop, end_ms=1000.0)
    client = _Client(samples=5, tokens=4)  # 20 rounds of 100 ms
    asyncio.run(sim._run_client(client))
    if stop:
        # every started round began before the window closed; none after
        assert len(starts) == 10 and max(starts) < 1000.0
    else:
        assert len(starts) == 20


def test_no_window_means_no_stop():
    sim, starts = _sim(True, end_ms=None)
    asyncio.run(sim._run_client(_Client(samples=2, tokens=3)))
    assert len(starts) == 6
