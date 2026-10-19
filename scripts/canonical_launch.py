"""Storage-safe canonical launcher (V1-batchwall).

Raw traces and Slurm logs live on /scratch; the run directory under
results/capacity_frontier/<tag>/ holds only compact files (config, hashes,
provenance, per-run summaries) and symlinks `raw` and `logs` into /scratch,
so /software file counts stay small. The simulator, controller, exact-SLO
classifier and run_one are reused unchanged.

  init     guards (clean tree, config semantics, quota) + immutable run dir
  run      guards (clean tree, HEAD unchanged) + one (N, seed) candidate
  summary  compact per-run verifier summary from the raw CSV (no GPU)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "legacy" / "src"))

import canonical_guard as guard  # noqa: E402

SCRATCH_ROOT = Path(os.environ.get("CANONICAL_SCRATCH", "/scratch/pawsey1257/quanla/canonical_raw"))
# Files one candidate run adds: per-run csv + yaml + cache part + compact summary.
SOFTWARE_FILES_PER_RUN = 4
SCRATCH_FILES_PER_RUN = 6


def _run_dir(tag: str) -> Path:
    import run_capacity_frontier as rcf

    return rcf.RESULTS_ROOT / tag


def cmd_init(a) -> int:
    import run_capacity_frontier as rcf

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    prov = guard.assert_clean_source()
    guard.validate_canonical_config(cfg, semantics=a.semantics)
    n_runs = int(a.planned_runs)
    pre = guard.preflight(n_runs * SOFTWARE_FILES_PER_RUN + 50, n_runs * SCRATCH_FILES_PER_RUN + 50)
    for w in pre["warnings"]:
        print(f"[warn] {w}", file=sys.stderr)

    run_dir = _run_dir(a.run_tag)
    scratch = SCRATCH_ROOT / a.run_tag
    if run_dir.exists() or scratch.exists():
        print(f"refusing to overwrite existing immutable run {run_dir} / {scratch}", file=sys.stderr)
        return 2
    for sub in ("raw", "logs"):
        (scratch / sub).mkdir(parents=True)
    run_dir.mkdir(parents=True)
    for sub in ("raw", "logs"):
        (run_dir / sub).symlink_to(scratch / sub, target_is_directory=True)
    ctx = rcf.RunContext.create(cfg, run_tag=a.run_tag)
    (ctx.run_dir / "cache_parts").mkdir(exist_ok=True)
    (ctx.run_dir / "compact").mkdir(exist_ok=True)
    (ctx.run_dir / "provenance_init.json").write_text(
        json.dumps(
            {"git": prov, "storage": pre["storage"], "scratch_dir": str(scratch), "semantics": a.semantics},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[init] run_dir={ctx.run_dir} raw/logs -> {scratch} config_hash={ctx.config_hash} git={ctx.git_sha}")
    return 0


def cmd_run(a) -> int:
    import certify_x3_run as cx
    import run_capacity_frontier as rcf

    ctx = rcf.RunContext.resume(a.run_tag)
    guard.assert_clean_source()
    guard.assert_runtime_env()
    guard.assert_head_matches(ctx.git_sha)
    init_info = json.loads((ctx.run_dir / "provenance_init.json").read_text(encoding="utf-8"))
    guard.validate_canonical_config(ctx.base_cfg, semantics=init_info["semantics"])
    for w in guard.preflight(SOFTWARE_FILES_PER_RUN + 50, SCRATCH_FILES_PER_RUN + 50, required=False)["warnings"]:
        print(f"[warn] {w}", file=sys.stderr)
    rc = cx.cmd_run(argparse.Namespace(run_tag=a.run_tag, n=a.n, seed=a.seed))
    cmd_summary(argparse.Namespace(run_tag=a.run_tag, n=a.n, seed=a.seed))
    return rc


def _hist(series, bins):
    import numpy as np

    counts, edges = np.histogram(series, bins=list(bins))
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def summarize_csv(csv_path: Path) -> dict:
    """Per-batch verifier summary (one row per batch id), compact and JSON-able."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    b = df.drop_duplicates("verification_batch_id")
    wall = b["verification_batch_wall_ms"]
    modeled = b["verification_batch_service_ms"]
    out = {
        "n_batches": int(len(b)),
        "service_time_model": sorted(set(b["verification_service_time_model"].astype(str))),
        "batch_size": {
            "mean": float(b["verification_batch_size"].mean()),
            "max": int(b["verification_batch_size"].max()),
            "hist": _hist(b["verification_batch_size"], range(1, 34)),
        },
        "batch_wall_ms": {
            "mean": float(wall.mean()),
            "p50": float(wall.median()),
            "p95": float(wall.quantile(0.95)),
            "max": float(wall.max()),
        },
        "modeled_service_ms": {"mean": float(modeled.mean()), "p95": float(modeled.quantile(0.95))},
        "amortized_request_ms_mean": float(b["verification_amortized_request_ms"].mean()),
        "wall_over_modeled_mean": float((wall / modeled.where(modeled > 0)).mean()),
        "logical_verifier_tokens": {
            "mean": float(b["verification_batch_token_cost"].mean()),
            "max": int(b["verification_batch_token_cost"].max()),
        },
        "processed_tokens": {
            "mean": float(b["verification_batch_processed_tokens"].mean()),
            "max": int(b["verification_batch_processed_tokens"].max()),
        },
        "context_tokens_sum_mean": float(b["verification_batch_context_tokens_sum"].mean()),
        "budget_binding_fraction": float(
            (b["verification_batch_token_cost"] >= b["verification_batch_token_budget"]).mean()
        ),
        "verifier_utilisation": None,
    }
    if "verify_finish_ms" in b and "batch_start_ms" in b:
        span_ms = float(b["verify_finish_ms"].max() - b["batch_start_ms"].min())
        if span_ms > 0:
            out["verifier_utilisation"] = float(modeled.sum() / span_ms)
    return out


def cmd_summary(a) -> int:
    run_dir = _run_dir(a.run_tag)
    csvs = sorted((run_dir / "raw").glob(f"*_n{a.n}_x*_s{a.seed}.csv"))
    if not csvs:
        print(f"no raw csv for n={a.n} seed={a.seed}", file=sys.stderr)
        return 1
    for csv in csvs:
        (run_dir / "compact").mkdir(exist_ok=True)
        dest = run_dir / "compact" / (csv.stem + ".verifier_summary.json")
        dest.write_text(json.dumps(summarize_csv(csv), indent=2), encoding="utf-8")
        print(f"[summary] {dest}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--run-tag", required=True)
    i.add_argument("--config", required=True)
    i.add_argument("--planned-runs", type=int, default=1)
    i.add_argument("--semantics", default="batch_wall", choices=["batch_wall", "amortized_per_request"],
                   help="amortized_per_request is only for a labelled legacy-V1 control run")
    for name in ("run", "summary"):
        s = sub.add_parser(name)
        s.add_argument("--run-tag", required=True)
        s.add_argument("--n", type=int, required=True)
        s.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    try:
        return {"init": cmd_init, "run": cmd_run, "summary": cmd_summary}[a.cmd](a)
    except guard.GuardError as exc:
        print(f"GUARD REFUSED: {exc}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
