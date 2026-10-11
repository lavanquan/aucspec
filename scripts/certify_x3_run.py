"""Single-candidate runner for the budget100 x=3.0 boundary certification.

Reuses run_capacity_frontier.run_one unchanged (same simulator, same
measure_candidate, same exact-window denominator). Each (N, seed) job writes
its own cache part under <run_dir>/cache_parts/ so concurrent jobs never
append to a shared file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_capacity_frontier as rcf  # noqa: E402
from edge_specsim.capacity import CandidateCache  # noqa: E402

POLICY = "capacity_dpp"


def cmd_init(a) -> int:
    run_dir = rcf.RESULTS_ROOT / a.run_tag
    if run_dir.exists():
        print(f"refusing to overwrite existing immutable run {run_dir}", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cs = cfg["capacity_search"]
    assert len(cs["x_grid_final"]) == 1, "certification config must hold exactly one x"
    ctx = rcf.RunContext.create(cfg, run_tag=a.run_tag)
    (ctx.run_dir / "cache_parts").mkdir(exist_ok=True)
    (ctx.run_dir / "logs").mkdir(exist_ok=True)
    print(f"[init] run_dir={ctx.run_dir} config_hash={ctx.config_hash} git={ctx.git_sha}")
    return 0


def cmd_run(a) -> int:
    ctx = rcf.RunContext.resume(a.run_tag)
    head = rcf._git_sha()
    if head != ctx.git_sha:
        print(f"code revision changed since init: HEAD={head} run={ctx.git_sha}", file=sys.stderr)
        return 4
    rcf._ACTIVE_POLICY = POLICY
    cs = ctx.base_cfg["capacity_search"]
    x = float(cs["x_grid_final"][0])
    meas_s = float(cs["measurement_seconds_final"])
    part = ctx.run_dir / "cache_parts" / f"n{a.n}_s{a.seed}.jsonl"
    cache = CandidateCache(part)
    m = rcf.run_one(ctx.base_cfg, a.n, x, a.seed, meas_s, ctx.config_hash, cache, ctx.raw_dir, False)
    exhausted = m.extra.get("exhausted_clients") or []
    print(f"[done] n={a.n} seed={a.seed} min_rate={m.min_rate_tps:.4f} exhausted={len(exhausted)}", flush=True)
    if exhausted:
        print("INVALID: workload exhaustion; rerun after increasing workload supply", file=sys.stderr)
        return 3
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("init")
    pi.add_argument("--config", required=True)
    pi.add_argument("--run-tag", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--run-tag", required=True)
    pr.add_argument("--n", type=int, required=True)
    pr.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    return cmd_init(a) if a.cmd == "init" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
