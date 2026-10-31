"""V2 smoke test before the frontier campaign: a few short candidates per
policy with the new verifier (prefix_fanout + prefix cache, Triton FA via
scripts/setonix_vllm_env.sh, batch_wall). Not a capacity measurement: it
checks the simulator runs end to end, records the right verifier settings,
and shows where N should be searched (min per-client rate vs x).

  python scripts/v2_smoke.py --config configs/capacity_frontier_budget100.yaml \
      --policies capacity_dpp,turbospec --x 4.5 --n 8,24 --measurement-seconds 60

V sweep on chosen (x, n) cells, several seeds:

  python scripts/v2_smoke.py --config configs/capacity_frontier_v2_screen.yaml \
      --cells 4.5:22,6:5 --V 50,250,1000 --seeds 1,2,3 \
      --measurement-seconds 120 --warmup-seconds 30 --out-dir results/v_sweep
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
    p.add_argument("--seeds", default=None, help="comma list; overrides --seed")
    p.add_argument("--cells", default=None, help="x:n pairs, e.g. 4.5:22,6:5; overrides --x/--n")
    p.add_argument("--V", default=None, help="comma list of controller V values (default: config)")
    p.add_argument("--gamma-choices", default=None, help="comma list, e.g. 0,1,2,3,4,5,6,7,8")
    p.add_argument("--measurement-seconds", type=float, default=60.0)
    p.add_argument("--warmup-seconds", type=float, default=15.0)
    p.add_argument("--out-dir", default="results/v2_smoke")
    args = p.parse_args()

    base = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    base["capacity_search"]["warmup_seconds"] = args.warmup_seconds
    if args.gamma_choices:
        base["controller"]["gamma_choices"] = [int(v) for v in args.gamma_choices.split(",")]
        base["simulation"]["gamma_max"] = max(base["controller"]["gamma_choices"])
    g_tag = f"_g{max(base['controller']['gamma_choices'])}" if args.gamma_choices else ""
    cells = (
        [(float(c.split(":")[0]), int(c.split(":")[1])) for c in args.cells.split(",")]
        if args.cells else [(args.x, int(v)) for v in args.n.split(",")]
    )
    seeds = [int(v) for v in args.seeds.split(",")] if args.seeds else [args.seed]
    v_values = [float(v) for v in args.V.split(",")] if args.V else [None]
    rows = []
    jobs = [(pol.strip(), v, x, n, seed) for pol in args.policies.split(",")
            for v in v_values for x, n in cells for seed in seeds]
    for policy, v_value, x, n, seed in jobs:
            rcf._ACTIVE_POLICY = policy
            if v_value is not None:
                base["controller"]["V"] = v_value
            raw = REPO_ROOT / args.out_dir / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            v_tag = "" if v_value is None else f"_V{v_value:g}"
            label = f"{policy}{v_tag}{g_tag}_n{n}_x{x}_s{seed}"
            out_csv = raw / f"{label}.csv"
            cfg = rcf.build_run_config(base, n, x, seed, args.measurement_seconds,
                                       str(out_csv.relative_to(REPO_ROOT)))
            cfg_path = raw / f"{label}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            print(f"=== smoke {label} ===", flush=True)
            r = rcf.subprocess.run([sys.executable, "legacy/scripts/run_simulation.py", "--config",
                                    str(cfg_path), "--detailed-log"], cwd=str(REPO_ROOT))
            row = {"policy": policy, "V": base["controller"]["V"], "n": n, "x": x, "seed": seed,
                   "exit": r.returncode}
            if r.returncode == 0:
                m = rcf.measure_candidate(out_csv, n, x, seed, args.measurement_seconds)
                row.update(min_rate_tps=round(m.min_rate_tps, 3), goodput_tps=round(m.goodput_tps, 2),
                           mean_gamma=round(m.mean_gamma, 3), mean_batch=round(m.mean_batch_size, 2),
                           feasible_point=m.min_rate_tps >= x)
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
