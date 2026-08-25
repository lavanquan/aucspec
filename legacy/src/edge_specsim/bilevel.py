from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import pandas as pd
import yaml

from .configuration import (
    Configuration,
    ConfigurationFeasibility,
    apply_configuration,
    build_configuration_space,
    configuration_feasibility,
)


@dataclass(frozen=True)
class OuterObjectiveSpec:
    metric: str = "auc_trapezoidal"
    auc_normalizer_x: float | None = None


@dataclass(frozen=True)
class InnerProblemSummary:
    auc_trapezoidal: float
    normalized_auc: float
    num_frontier_points: int
    num_hull_points: int
    headline_policy_family: str
    headline_goodput_tps: float
    frontier_csv: str
    paper_table_csv: str
    auc_json: str
    frontier_semantics: str


@dataclass(frozen=True)
class BilevelCandidateEvaluation:
    configuration: Configuration
    feasibility: ConfigurationFeasibility
    objective_value: float
    inner_summary: InnerProblemSummary | None = None

    def to_row(self) -> dict[str, object]:
        target_model_id = self.configuration.target_model_id
        row: dict[str, object] = {
            "configuration_id": self.configuration.slug(),
            "draft_model_id": self.configuration.draft_model_id,
            "target_model_id": target_model_id,
            "draft_placement": self.configuration.draft_placement,
            "draft_pool_size": self.configuration.draft_pool_size,
            "target_replication_factor": self.configuration.target_replication_factor,
            "feasible": self.feasibility.feasible,
            "feasibility_reason": self.feasibility.reason,
            "outer_objective_value": self.objective_value,
        }
        if self.inner_summary is None:
            row.update(
                {
                    "auc_trapezoidal": float("nan"),
                    "normalized_auc": float("nan"),
                    "num_frontier_points": 0,
                    "num_hull_points": 0,
                    "headline_policy_family": "",
                    "headline_goodput_tps": float("nan"),
                    "frontier_csv": "",
                    "paper_table_csv": "",
                    "auc_json": "",
                    "frontier_semantics": "",
                }
            )
            return row
        row.update(asdict(self.inner_summary))
        return row


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml_config(config_path: Path, cfg: dict[str, Any]) -> None:
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def parse_outer_objective_spec(
    cfg: dict[str, Any],
    interactivity_targets: list[float] | None = None,
) -> OuterObjectiveSpec:
    outer_cfg = cfg.get("outer_optimization", {})
    objective_cfg = outer_cfg.get("objective", {})
    metric = str(objective_cfg.get("metric", "auc_trapezoidal")).strip().lower()
    if metric not in {"auc_trapezoidal", "normalized_auc", "headline_goodput_tps"}:
        raise ValueError(
            "outer_optimization.objective.metric must be one of: "
            "auc_trapezoidal, normalized_auc, headline_goodput_tps"
        )
    auc_normalizer_x = objective_cfg.get("auc_normalizer_x")
    if auc_normalizer_x is None and interactivity_targets:
        auc_normalizer_x = max(float(value) for value in interactivity_targets)
    if auc_normalizer_x is not None:
        auc_normalizer_x = max(0.0, float(auc_normalizer_x))
    return OuterObjectiveSpec(metric=metric, auc_normalizer_x=auc_normalizer_x)


def summarize_inner_result(
    result: dict[str, object],
    objective_spec: OuterObjectiveSpec,
) -> tuple[InnerProblemSummary, float]:
    auc_summary = result["auc_summary"]
    paper_table_df = result["paper_table_df"]
    frontier_rows = list(result.get("frontier_rows", []))
    headline_goodput = float("nan")
    headline_policy = ""
    if isinstance(paper_table_df, pd.DataFrame) and not paper_table_df.empty:
        best_family = paper_table_df.sort_values("auc_trapezoidal", ascending=False).iloc[0]
        headline_goodput = float(best_family["headline_goodput_tps"])
        headline_policy = str(best_family["policy_family"])
    x_normalizer = objective_spec.auc_normalizer_x
    auc_value = float(auc_summary["estimated_auc_trapezoidal"])
    if x_normalizer is None or x_normalizer <= 0.0:
        normalized_auc = auc_value
    else:
        normalized_auc = auc_value / x_normalizer
    frontier_semantics = "estimated_best_over_sampled_inner_policies"
    if frontier_rows:
        frontier_semantics = str(frontier_rows[0].get("frontier_semantics", frontier_semantics))
    summary = InnerProblemSummary(
        auc_trapezoidal=auc_value,
        normalized_auc=float(normalized_auc),
        num_frontier_points=int(auc_summary["num_frontier_points"]),
        num_hull_points=int(auc_summary["num_hull_points"]),
        headline_policy_family=headline_policy,
        headline_goodput_tps=headline_goodput,
        frontier_csv=str(result["frontier_csv"]),
        paper_table_csv=str(result["paper_table_csv"]),
        auc_json=str(result["auc_json"]),
        frontier_semantics=frontier_semantics,
    )
    if objective_spec.metric == "auc_trapezoidal":
        objective_value = summary.auc_trapezoidal
    elif objective_spec.metric == "normalized_auc":
        objective_value = summary.normalized_auc
    elif objective_spec.metric == "headline_goodput_tps":
        objective_value = summary.headline_goodput_tps
    else:
        raise AssertionError(f"Unsupported objective metric: {objective_spec.metric}")
    return summary, float(objective_value)


class BilevelOuterSolver:
    def __init__(
        self,
        *,
        base_cfg: dict[str, Any],
        output_dir: Path,
        objective_spec: OuterObjectiveSpec,
        run_inner_problem: Callable[[Configuration, Path], Awaitable[dict[str, object]]],
        clean_output: bool = False,
    ) -> None:
        self.base_cfg = base_cfg
        self.output_dir = Path(output_dir)
        self.objective_spec = objective_spec
        self.run_inner_problem = run_inner_problem
        self.clean_output = bool(clean_output)

    def enumerate_candidates(self) -> list[Configuration]:
        return build_configuration_space(self.base_cfg)

    async def evaluate_configuration(
        self,
        configuration: Configuration,
    ) -> BilevelCandidateEvaluation:
        feasibility = configuration_feasibility(self.base_cfg, configuration)
        if not feasibility.feasible:
            return BilevelCandidateEvaluation(
                configuration=configuration,
                feasibility=feasibility,
                objective_value=float("-inf"),
                inner_summary=None,
            )
        config_output_dir = self.output_dir / configuration.slug()
        result = await self.run_inner_problem(configuration, config_output_dir)
        inner_summary, objective_value = summarize_inner_result(result, self.objective_spec)
        return BilevelCandidateEvaluation(
            configuration=configuration,
            feasibility=feasibility,
            objective_value=objective_value,
            inner_summary=inner_summary,
        )

    async def solve(self) -> dict[str, object]:
        if self.clean_output and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        evaluations: list[BilevelCandidateEvaluation] = []
        best: BilevelCandidateEvaluation | None = None
        for configuration in self.enumerate_candidates():
            evaluation = await self.evaluate_configuration(configuration)
            evaluations.append(evaluation)
            if not evaluation.feasibility.feasible:
                continue
            if best is None or evaluation.objective_value > best.objective_value:
                best = evaluation

        rows = [evaluation.to_row() for evaluation in evaluations]
        summary_df = pd.DataFrame(rows).sort_values(
            by=["feasible", "outer_objective_value"],
            ascending=[False, False],
            na_position="last",
        )
        evaluations_csv = self.output_dir / "bilevel_evaluations.csv"
        summary_df.to_csv(evaluations_csv, index=False)
        legacy_csv = self.output_dir / "configuration_search.csv"
        summary_df.to_csv(legacy_csv, index=False)

        outer_problem_json = self.output_dir / "outer_problem.json"
        outer_payload = {
            "problem_type": "bilevel_outer_inner_optimization",
            "upper_level_variable": "xi",
            "upper_level_objective_metric": self.objective_spec.metric,
            "auc_normalizer_x": self.objective_spec.auc_normalizer_x,
            "num_candidates": len(evaluations),
            "num_feasible_candidates": sum(1 for item in evaluations if item.feasibility.feasible),
        }
        with outer_problem_json.open("w", encoding="utf-8") as handle:
            json.dump(outer_payload, handle, indent=2)

        best_json = self.output_dir / "best_bilevel_solution.json"
        legacy_best_json = self.output_dir / "best_configuration.json"
        best_payload: dict[str, object]
        if best is None:
            best_payload = {}
        else:
            best_payload = best.to_row()
            best_payload["selected_configuration"] = asdict(best.configuration)
            best_payload["objective_metric"] = self.objective_spec.metric
        for path in (best_json, legacy_best_json):
            with path.open("w", encoding="utf-8") as handle:
                json.dump(best_payload, handle, indent=2)

        return {
            "evaluations": evaluations,
            "summary_df": summary_df,
            "evaluations_csv": evaluations_csv,
            "legacy_csv": legacy_csv,
            "outer_problem_json": outer_problem_json,
            "best_json": best_json,
            "legacy_best_json": legacy_best_json,
            "best_evaluation": best,
        }


def make_frontier_inner_runner(
    *,
    base_cfg: dict[str, Any],
    interactivity_targets: list[float] | None,
    policies: list[str] | None,
    V_values: list[float] | None,
    fixed_gamma_values: list[int] | None,
    seeds: list[int] | None,
    detailed_log: bool,
    run_frontier_sweep: Callable[..., Awaitable[dict[str, object]]],
) -> Callable[[Configuration, Path], Awaitable[dict[str, object]]]:
    async def _runner(configuration: Configuration, config_output_dir: Path) -> dict[str, object]:
        with tempfile.TemporaryDirectory(prefix=f"outer-config-{configuration.slug()}-") as tmp_dir:
            temp_config_path = Path(tmp_dir) / "config.yaml"
            configured = apply_configuration(base_cfg, configuration)
            write_yaml_config(temp_config_path, configured)
            return await run_frontier_sweep(
                config_path=temp_config_path,
                output_dir=config_output_dir,
                interactivity_targets=interactivity_targets,
                policies=policies,
                V_values=V_values,
                fixed_gamma_values=fixed_gamma_values,
                seeds=seeds,
                detailed_log=detailed_log,
                clean_output=False,
            )

    return _runner
