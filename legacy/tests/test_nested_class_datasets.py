"""simulation.nested_class_datasets: per-class dataset pools in nested mode."""
from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

for _name, _attrs in (
    ("edge_specsim.target_client", {"VLLMCandidateVerifier": object}),
    ("edge_specsim.draft_worker", {"DraftRequest": object, "DraftWorker": object}),
):
    if importlib.util.find_spec("torch") is None and _name not in sys.modules:
        _stub = types.ModuleType(_name)
        for _k, _v in _attrs.items():
            setattr(_stub, _k, _v)
        sys.modules[_name] = _stub

import edge_specsim.simulator as simmod  # noqa: E402
from edge_specsim.simulator import EdgeSpecSimulator  # noqa: E402

_CFG = Path(__file__).resolve().parents[2] / "configs" / "capacity_frontier_v2_screen.yaml"
QPC, NMAX = 2, 6
CLASSES = ["a", "b", "g", "a", "b", "g"]  # g = no dataset override (global pool)


def _S(sid: str):
    return SimpleNamespace(sample_id=sid)


def _ids(samples):
    return [s.sample_id for s in samples]


def _sim(monkeypatch, flag: bool, loads: list):
    cfg = copy.deepcopy(yaml.safe_load(_CFG.read_text()))
    cfg["simulation"]["nested_class_datasets"] = flag
    cfg["clients"]["client_classes"] = {
        "a": {"probability": 1 / 3, "dataset": {"name": "ds_a"}},
        "b": {"probability": 1 / 3, "dataset": {"name": "ds_b"}},
        "g": {"probability": 1 / 3},
    }

    def fake_load(dcfg):
        loads.append(dict(dcfg))
        return [_S(f"{dcfg['name']}:{k}") for k in range(int(dcfg["num_questions"]))]

    monkeypatch.setattr(simmod, "load_samples", fake_load)
    sim = EdgeSpecSimulator.__new__(EdgeSpecSimulator)
    sim.cfg = cfg
    sim.learning_cfg = {}
    sim.workers = [SimpleNamespace(device="cpu")]
    sim._nested_population = True
    sim._nested_qpc = QPC
    sim._nested_n_max = NMAX
    names = iter(CLASSES)

    def pick(_c):
        n = next(names)
        return n, cfg["clients"]["client_classes"][n]

    sim._sample_client_class = pick
    glob = [_S(f"global:{k}") for k in range(NMAX * QPC)]
    sim.samples = list(glob)
    return sim, glob


def _prompts(clients):
    return [[s.sample_id for s in c.prompt_queue] for c in clients]


def test_flag_off_keeps_global_blocks(monkeypatch):
    loads: list = []
    sim, glob = _sim(monkeypatch, False, loads)
    p = _prompts(sim._make_clients(glob, len(CLASSES)))
    assert loads == []
    assert p == [_ids(glob[i * QPC:(i + 1) * QPC]) for i in range(len(CLASSES))]


def test_flag_on_uses_class_pools_by_rank(monkeypatch):
    loads: list = []
    sim, glob = _sim(monkeypatch, True, loads)
    p = _prompts(sim._make_clients(glob, len(CLASSES)))
    assert sorted(d["name"] for d in loads) == ["ds_a", "ds_b"]
    assert all(d["num_questions"] == NMAX * QPC for d in loads)
    assert p[0] == ["ds_a:0", "ds_a:1"] and p[3] == ["ds_a:2", "ds_a:3"]
    assert p[1] == ["ds_b:0", "ds_b:1"] and p[4] == ["ds_b:2", "ds_b:3"]
    assert p[2] == _ids(glob[2 * QPC:3 * QPC]) and p[5] == _ids(glob[5 * QPC:6 * QPC])


def test_flag_on_prefix_stable(monkeypatch):
    sim6, glob = _sim(monkeypatch, True, [])
    p6 = _prompts(sim6._make_clients(glob, 6))
    sim4, glob = _sim(monkeypatch, True, [])
    p4 = _prompts(sim4._make_clients(glob, 4))
    assert p4 == p6[:4]
