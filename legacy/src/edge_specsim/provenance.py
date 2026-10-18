"""Behavior-neutral run provenance: source state, compute node, verifier semantics.

Everything here is best-effort and read-only. It never changes simulation
behavior; failures are recorded in an `errors` list instead of raised, so a
missing tool can never abort a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any

# Untracked files inside these directories change what runs, so they make a
# worktree "dirty" for reproducibility purposes. Results/logs/caches do not.
SOURCE_DIRS = ("legacy/src", "legacy/scripts", "scripts", "configs")

SLURM_ENV_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "SLURMD_NODENAME",
    "SLURM_CLUSTER_NAME",
    "SLURM_GPUS_ON_NODE",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "LOADEDMODULES",
    "ROCM_PATH",
)


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def git_provenance(repo_root: Path | str) -> dict[str, Any]:
    """Commit, dirty state and a hash of any tracked modification."""
    root = Path(repo_root)
    out: dict[str, Any] = {
        "git_sha": None,
        "git_dirty": None,
        "git_dirty_tracked_files": [],
        "git_untracked_source_files": [],
        "git_diff_sha256": None,
        "errors": [],
    }
    try:
        out["git_sha"] = _git(root, "rev-parse", "HEAD").strip()
        tracked = [
            line[3:]
            for line in _git(root, "status", "--porcelain", "--untracked-files=no").splitlines()
        ]
        out["git_dirty_tracked_files"] = sorted(tracked)
        untracked = [
            line[3:]
            for line in _git(root, "status", "--porcelain", "--untracked-files=all", "--", *SOURCE_DIRS).splitlines()
            if line.startswith("??")
        ]
        out["git_untracked_source_files"] = sorted(untracked)
        diff = _git(root, "diff", "HEAD")
        out["git_diff_sha256"] = hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None
        out["git_dirty"] = bool(tracked or untracked)
    except Exception as exc:  # noqa: BLE001 - provenance must never abort a run
        out["errors"].append(f"git: {exc!r}")
    return out


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except Exception:  # noqa: BLE001
        return None


def node_provenance() -> dict[str, Any]:
    """Compute-node facts, collected from the job process itself."""
    out: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "slurm": {key: os.environ.get(key) for key in SLURM_ENV_KEYS},
        "rocm_version_file": _read_text("/opt/rocm/.info/version"),
        "amdgpu_driver_version": _read_text("/sys/module/amdgpu/version"),
        "gpu": {"device_count": None, "device_names": [], "hip_version": None, "cuda_version": None},
        "errors": [],
    }
    try:
        import torch  # type: ignore

        out["gpu"]["hip_version"] = getattr(torch.version, "hip", None)
        out["gpu"]["cuda_version"] = getattr(torch.version, "cuda", None)
        count = int(torch.cuda.device_count())
        out["gpu"]["device_count"] = count
        out["gpu"]["device_names"] = [torch.cuda.get_device_name(i) for i in range(count)]
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(f"torch: {exc!r}")
    return out


def config_sha256(cfg: dict[str, Any]) -> str:
    """Hash of the resolved per-run config (paths and seed included, so it
    identifies this exact run; the campaign-level hash is separate)."""
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect_provenance(repo_root: Path | str, cfg: dict[str, Any]) -> dict[str, Any]:
    campaign = cfg.get("provenance", {}) or {}
    return {
        **git_provenance(repo_root),
        "node": node_provenance(),
        "resolved_config_sha256": config_sha256({k: v for k, v in cfg.items() if k != "provenance"}),
        "campaign_config_hash": campaign.get("campaign_config_hash"),
        "population_fingerprint": campaign.get("population_fingerprint"),
    }


def verifier_semantics_record(
    target: Any,
    batching_cfg: dict[str, Any],
    controller_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """What the verifier actually does, as exposed by the running objects."""
    return {
        "verifier_semantics": getattr(target, "VERIFIER_SEMANTICS", "unknown"),
        "tie_break_rule": getattr(target, "TIE_BREAK_RULE", "unknown"),
        "effective_max_logprobs": getattr(target, "effective_max_logprobs", None),
        "greedy_only": getattr(target, "greedy_only", None),
        "service_time_model": str(batching_cfg.get("service_time_model", "amortized_per_request")),
        # Only "zero" exists today: the controller is handed theta_f = 0 in fixed-budget mode.
        "verifier_price_mode": str((controller_cfg or {}).get("verifier_price_mode", "zero")),
        "budget_mode": str(batching_cfg.get("budget_mode", "fixed")),
        "verify_token_budget": batching_cfg.get("verify_token_budget"),
        "max_batch_size": batching_cfg.get("max_batch_size"),
    }
