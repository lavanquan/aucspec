"""Clean-source and storage-quota guards for canonical runs.

Pure checks: no GPU, no job submission, no writes. The functions raise
GuardError (or return warnings) so a launcher can refuse to start.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "legacy" / "src"))

from edge_specsim.provenance import git_provenance  # noqa: E402
from edge_specsim.verification_batcher import SERVICE_TIME_MODELS  # noqa: E402

REQUIRED_BATCHING_KEYS = (
    "service_time_model",
    "scheduler",
    "budget_mode",
    "verify_token_budget",
    "max_batch_size",
    "batch_wait_ms",
)


class GuardError(RuntimeError):
    pass


# ------------------------------------------------------------------ source
def assert_clean_source(repo_root: Path | str = REPO_ROOT) -> dict:
    """Refuse a dirty tree: any tracked modification, or an untracked file
    inside a source directory (legacy/src, legacy/scripts, scripts, configs)."""
    prov = git_provenance(repo_root)
    if prov["git_sha"] is None:
        raise GuardError(f"cannot read git state: {prov['errors']}")
    if prov["git_dirty"]:
        raise GuardError(
            "source tree is not clean; refusing a canonical run. "
            f"tracked changes: {prov['git_dirty_tracked_files']}; "
            f"untracked source files: {prov['git_untracked_source_files']}"
        )
    return prov


def assert_head_matches(expected_sha: str, repo_root: Path | str = REPO_ROOT) -> None:
    prov = git_provenance(repo_root)
    if prov["git_sha"] != expected_sha:
        raise GuardError(
            f"code revision changed since the run was initialised: HEAD={prov['git_sha']} run={expected_sha}"
        )


# ------------------------------------------------------------------ config
def validate_canonical_config(cfg: dict, semantics: str = "batch_wall") -> None:
    """The verifier clock semantics must be chosen explicitly, never inherited
    from a default, so a canonical run cannot silently use legacy V1."""
    batching = cfg.get("verification_batching") or {}
    missing = [key for key in REQUIRED_BATCHING_KEYS if key not in batching]
    if missing:
        raise GuardError(f"verification_batching is missing required keys: {missing}")
    model = batching["service_time_model"]
    if model not in SERVICE_TIME_MODELS:
        raise GuardError(f"unknown service_time_model {model!r}; expected one of {SERVICE_TIME_MODELS}")
    if model != semantics:
        raise GuardError(f"service_time_model is {model!r} but this launch requires {semantics!r}")
    if batching["budget_mode"] != "fixed":
        raise GuardError("canonical contract uses budget_mode: fixed")
    mode = (cfg.get("controller") or {}).get("verifier_price_mode")
    if mode != "zero":
        raise GuardError(f"controller.verifier_price_mode must be explicitly 'zero' (got {mode!r}); calibrated pricing is not enabled")


# ------------------------------------------------------------------- quota
_LINE = re.compile(r"^\s*(?P<fs>/\w+)\s+(?P<rest>.*?)\s+(?P<files>\d+)\s+(?P<files_pct>[\d.]+)\s+\(user\)\s*$")
_SPACE = re.compile(r"(?P<pct>[\d.]+)\s+\((?P<scope>project|user)\)")


def parse_pawsey_storage(text: str) -> dict[str, dict]:
    """Parse `pawseyAccountBalance -p <project> -storage` output.

    The tool prints the user file count and its percentage of the (unprinted)
    limit; the limit is recovered as files / pct and rounded to 1000, which
    matches the documented 250,000 (/software), 1,000,000 (/scratch) and
    10,000 (/home) limits.
    """
    out: dict[str, dict] = {}
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        files = int(m["files"])
        pct = float(m["files_pct"])
        space = _SPACE.search(m["rest"])
        out[m["fs"]] = {
            "files": files,
            "files_pct": pct,
            "file_limit_est": int(round(files * 100.0 / pct, -3)) if pct > 0 else None,
            "space_pct": float(space["pct"]) if space else None,
            "space_scope": space["scope"] if space else None,
        }
    return out


def read_pawsey_storage(project: str = "pawsey1257", timeout_s: int = 90) -> dict[str, dict]:
    result = subprocess.run(
        ["pawseyAccountBalance", "-p", project, "-storage"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    parsed = parse_pawsey_storage(result.stdout)
    if "/software" not in parsed:
        raise GuardError("could not read /software file-count quota from pawseyAccountBalance")
    return parsed


def check_headroom(
    storage: dict[str, dict],
    planned_software_files: int,
    planned_scratch_files: int,
    margin: int = 2000,
) -> list[str]:
    """Return warnings; raise GuardError if a file-count budget would be exceeded."""
    warnings: list[str] = []
    for fs, planned in (("/software", planned_software_files), ("/scratch", planned_scratch_files)):
        info = storage.get(fs)
        if info is None:
            raise GuardError(f"no quota information for {fs}")
        headroom = info["file_limit_est"] - info["files"]
        if headroom < planned + margin:
            raise GuardError(
                f"{fs}: file headroom {headroom} < planned {planned} + margin {margin} "
                f"({info['files']}/{info['file_limit_est']} files used)"
            )
    home = storage.get("/home")
    if home is not None and home["space_pct"] is not None and home["space_pct"] >= 99.0:
        warnings.append(
            f"/home is {home['space_pct']}% full: any tool writing under $HOME may fail; "
            "keep caches and temp files under /scratch"
        )
    return warnings


def preflight(
    planned_software_files: int,
    planned_scratch_files: int,
    margin: int = 2000,
    project: str = "pawsey1257",
) -> dict:
    storage = read_pawsey_storage(project)
    warnings = check_headroom(storage, planned_software_files, planned_scratch_files, margin)
    return {"storage": storage, "warnings": warnings}
