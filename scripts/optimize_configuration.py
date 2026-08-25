from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_specsim.bilevel import (
    BilevelOuterSolver,
    load_yaml_config,
    make_frontier_inner_runner,
    parse_outer_objective_spec,
)
from sweep_min_interactivity import run_frontier_sweep


def _parse_float_list(value: str | None) -> list[float] | None:
    if not value:
        return None
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


async def _run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    base_cfg = load_yaml_config(config_path)
    interactivity_targets = _parse_float_list(args.interactivity_targets)
    policies = [part.strip() for part in args.policies.split(",") if part.strip()] if args.policies else None
    V_values = _parse_float_list(args.V_values)
    fixed_gamma_values = _parse_int_list(args.fixed_gamma_values)
    seeds = _parse_int_list(args.seeds)

    objective_spec = parse_outer_objective_spec(
        base_cfg,
        interactivity_targets=interactivity_targets,
    )
    inner_runner = make_frontier_inner_runner(
        base_cfg=base_cfg,
        interactivity_targets=interactivity_targets,
        policies=policies,
        V_values=V_values,
        fixed_gamma_values=fixed_gamma_values,
        seeds=seeds,
        detailed_log=args.detailed_log,
        run_frontier_sweep=run_frontier_sweep,
    )
    solver = BilevelOuterSolver(
        base_cfg=base_cfg,
        output_dir=Path(args.output_dir),
        objective_spec=objective_spec,
        run_inner_problem=inner_runner,
        clean_output=args.clean_output,
    )
    result = await solver.solve()
    best = result["best_evaluation"]
    print(f"Wrote bilevel evaluation summary to {result['evaluations_csv']}")
    print(f"Wrote outer problem summary to {result['outer_problem_json']}")
    print(f"Wrote best bilevel solution to {result['best_json']}")
    if best is not None:
        print(
            "best configuration={configuration_id} objective={objective:.6f}".format(
                configuration_id=best.configuration.slug(),
                objective=float(best.objective_value),
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the bilevel outer problem over configuration xi. "
            "The outer level optimizes configuration choices, while the inner level "
            "solves the runtime frontier/AUC problem for each xi."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="results/configuration_search")
    parser.add_argument("--interactivity-targets")
    parser.add_argument("--policies")
    parser.add_argument("--V-values")
    parser.add_argument("--fixed-gamma-values")
    parser.add_argument("--seeds")
    parser.add_argument("--detailed-log", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
