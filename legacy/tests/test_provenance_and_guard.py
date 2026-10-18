"""CPU-only tests: timeline has no double-counting; provenance; guards."""

from __future__ import annotations

import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest

from edge_specsim.provenance import (
    config_sha256,
    git_provenance,
    node_provenance,
    verifier_semantics_record,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def simulator_module(monkeypatch):
    """Import the simulator with the torch-dependent modules stubbed."""
    dw = types.ModuleType("edge_specsim.draft_worker")
    dw.DraftRequest = dw.DraftWorker = dw.DraftKVState = object
    tc = types.ModuleType("edge_specsim.target_client")
    tc.VLLMCandidateVerifier = object
    monkeypatch.setitem(sys.modules, "edge_specsim.draft_worker", dw)
    monkeypatch.setitem(sys.modules, "edge_specsim.target_client", tc)
    monkeypatch.delitem(sys.modules, "edge_specsim.simulator", raising=False)
    mod = importlib.import_module("edge_specsim.simulator")
    yield mod
    sys.modules.pop("edge_specsim.simulator", None)


def test_verifier_time_counted_once_in_round_latency(simulator_module):
    sim = types.SimpleNamespace(virtual_system_time_ms=0.0, virtual_target_available_at_ms=0.0)
    client = types.SimpleNamespace(virtual_time_ms=1000.0, virtual_start_time_ms=0.0, rtt_ms=20.0)
    advance = simulator_module.EdgeSpecSimulator._advance_virtual_timeline
    # draft 50, upload 5, rtt/2 10 -> arrival 1065; batch starts 1065, wall 200
    out = advance(sim, client, 50.0, 5.0, 4.0, 1065.0, 1065.0, 1065.0, 1265.0)
    assert out["client_receive_ms"] == pytest.approx(1265.0 + 10.0 + 4.0)
    # verifier occupancy appears exactly once: round = draft + up + wait(0) + verify + down
    assert out["virtual_round_latency_ms"] == pytest.approx(50 + 10 + 5 + 200 + 10 + 4)
    assert sim.virtual_target_available_at_ms == 1265.0


def test_target_latency_not_used_for_timeline(simulator_module):
    src = Path(simulator_module.__file__).read_text()
    # target_latency_ms (amortized) may only be logged, never fed into the timeline call
    start = src.index("self._advance_virtual_timeline(")
    timeline_call = src[start : start + 700]
    assert "verify_finish_ms" in timeline_call  # guard against a vacuous slice
    assert "target_latency_ms" not in timeline_call


# ---- provenance ---------------------------------------------------------
def _git(cwd, *a):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": __import__("os").environ["PATH"],
                        "HOME": str(cwd)})


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "a.py").write_text("x=1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_git_clean(repo):
    p = git_provenance(repo)
    assert p["git_dirty"] is False and len(p["git_sha"]) == 40 and p["git_diff_sha256"] is None


def test_git_dirty_tracked(repo):
    (repo / "scripts" / "a.py").write_text("x=2\n")
    p = git_provenance(repo)
    assert p["git_dirty"] is True and p["git_dirty_tracked_files"] == ["scripts/a.py"]
    assert p["git_diff_sha256"]


def test_git_untracked_source_is_dirty_but_results_are_not(repo):
    (repo / "results").mkdir()
    (repo / "results" / "r.csv").write_text("1\n")
    assert git_provenance(repo)["git_dirty"] is False
    (repo / "scripts" / "new.py").write_text("y=1\n")
    p = git_provenance(repo)
    assert p["git_dirty"] is True and p["git_untracked_source_files"] == ["scripts/new.py"]


def test_git_error_is_recorded_not_raised(tmp_path):
    p = git_provenance(tmp_path / "nonexistent")
    assert p["git_sha"] is None and p["errors"]


def test_node_provenance_reads_slurm_env(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURMD_NODENAME", "nid001")
    n = node_provenance()
    assert n["slurm"]["SLURM_JOB_ID"] == "123" and n["slurm"]["SLURMD_NODENAME"] == "nid001"
    assert "hostname" in n


def test_config_hash_stable_and_sensitive():
    assert config_sha256({"a": 1, "b": 2}) == config_sha256({"b": 2, "a": 1})
    assert config_sha256({"a": 1}) != config_sha256({"a": 2})


def test_verifier_semantics_record():
    t = types.SimpleNamespace(VERIFIER_SEMANTICS="V1", TIE_BREAK_RULE="r", effective_max_logprobs=1, greedy_only=True)
    r = verifier_semantics_record(t, {"service_time_model": "batch_wall", "verify_token_budget": 100})
    assert r["verifier_semantics"] == "V1" and r["service_time_model"] == "batch_wall"
    assert r["effective_max_logprobs"] == 1 and r["verify_token_budget"] == 100
    assert verifier_semantics_record(object(), {})["service_time_model"] == "amortized_per_request"


# ---- guards -------------------------------------------------------------
@pytest.fixture
def guard():
    sys.path.insert(0, str(REPO / "scripts"))
    import canonical_guard
    yield canonical_guard
    sys.path.remove(str(REPO / "scripts"))


GOOD = {"verification_batching": {"service_time_model": "batch_wall", "scheduler": "capacity_knapsack",
                                  "budget_mode": "fixed", "verify_token_budget": 100,
                                  "max_batch_size": 32, "batch_wait_ms": 3.0}}


def test_guard_clean_and_dirty(guard, repo):
    assert guard.assert_clean_source(repo)["git_dirty"] is False
    (repo / "scripts" / "a.py").write_text("x=3\n")
    with pytest.raises(guard.GuardError):
        guard.assert_clean_source(repo)


def test_guard_head_mismatch(guard, repo):
    with pytest.raises(guard.GuardError):
        guard.assert_head_matches("0" * 40, repo)


def test_guard_config(guard):
    guard.validate_canonical_config(GOOD)
    bad = {"verification_batching": {**GOOD["verification_batching"], "service_time_model": "amortized_per_request"}}
    with pytest.raises(guard.GuardError):
        guard.validate_canonical_config(bad)
    missing = {"verification_batching": {"scheduler": "x"}}
    with pytest.raises(guard.GuardError):
        guard.validate_canonical_config(missing)


SAMPLE = """\
/software   1 TiB ... 3.2 (project)   250378   100.2 (user)
/scratch    10 TiB   30.0 (project)   25000    2.5 (user)
/home       1 GiB    100.0 (user)     400      4.0 (user)
"""


def test_guard_quota_parse_and_headroom(guard):
    st = guard.parse_pawsey_storage(SAMPLE)
    assert st["/software"]["files"] == 250378 and st["/software"]["file_limit_est"] == 250000
    with pytest.raises(guard.GuardError):
        guard.check_headroom(st, 100, 100)
    ok = {"/software": {"files": 100000, "file_limit_est": 250000, "space_pct": 1, "space_scope": "user"},
          "/scratch": {"files": 1000, "file_limit_est": 1000000, "space_pct": 1, "space_scope": "user"},
          "/home": {"files": 1, "file_limit_est": 10000, "space_pct": 100.0, "space_scope": "user"}}
    warnings = guard.check_headroom(ok, 1000, 5000)
    assert warnings and "/home" in warnings[0]
    with pytest.raises(guard.GuardError):
        guard.check_headroom(ok, 200000, 1)
