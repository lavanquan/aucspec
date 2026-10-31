"""V2 smoke test before the frontier campaign: a few short candidates per
policy with the new verifier (prefix_fanout + prefix cache, Triton FA via
scripts/setonix_vllm_env.sh, batch_wall). Not a capacity measurement: it
checks the simulator runs end to end, records the right verifier settings,
and shows where N should be searched (min per-client rate vs x).

  python scripts/v2_smoke.py --config configs/capacity_frontier_budget100.yaml \
      --policies capacity_dpp,turbospec --x 4.5 --n 8,24 --measurement-seconds 60
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rcf", REPO_ROOT / "scripts" / "run_capacity_frontier.py")
rcf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcf)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/capacity_frontier_budget100.yaml")
    p.add_argument("--policies", default="capacity_dpp")
    p.add_argument("--x", type=float, default=4.5)
    p.add_argument("--n", default="8,24")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--measurement-seconds", type=float, default=60.0)
    p.add_argument("--warmup-seconds", type=float, default=15.0)
    p.add_argument("--out-dir", default="results/v2_smoke")
    args = p.parse_args()

    base = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    base["capacity_search"]["warmup_seconds"] = args.warmup_seconds
    rows = []
    for policy in [s.strip() for s in args.policies.split(",")]:
        rcf._ACTIVE_POLICY = policy
        for n in [int(v) for v in args.n.split(",")]:
            raw = REPO_ROOT / args.out_dir / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            label = f"{policy}_n{n}_x{args.x}_s{args.seed}"
            out_csv = raw / f"{label}.csv"
            cfg = rcf.build_run_config(base, n, args.x, args.seed, args.measurement_seconds,
                                       str(out_csv.relative_to(REPO_ROOT)))
            cfg_path = raw / f"{label}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            print(f"=== smoke {label} ===", flush=True)
            r = rcf.subprocess.run([sys.executable, "legacy/scripts/run_simulation.py", "--config",
                                    str(cfg_path), "--detailed-log"], cwd=str(REPO_ROOT))
            row = {"policy": policy, "n": n, "x": args.x, "exit": r.returncode}
            if r.returncode == 0:
                m = rcf.measure_candidate(out_csv, n, args.x, args.seed, args.measurement_seconds)
                row.update(min_rate_tps=round(m.min_rate_tps, 3), goodput_tps=round(m.goodput_tps, 2),
                           mean_gamma=round(m.mean_gamma, 3), mean_batch=round(m.mean_batch_size, 2),
                           feasible_point=m.min_rate_tps >= args.x)
                rc = raw / f"{label}_artifacts" / "run_config.json"
                if rc.exists():
                    j = json.loads(rc.read_text())
                    row.update(verify_mode=j.get("target_verify_mode"),
                               prefix_cache=j.get("target_enable_prefix_caching"),
                               service_time=j.get("verifier_service_time_model"),
                               triton=(j.get("software_versions") or {}).get("triton"))
            rows.append(row)
            print("SMOKE", json.dumps(row), flush=True)
    (REPO_ROOT / args.out_dir / "smoke_summary.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
