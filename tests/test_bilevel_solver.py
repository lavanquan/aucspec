from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from edge_specsim.bilevel import (
    BilevelOuterSolver,
    OuterObjectiveSpec,
    parse_outer_objective_spec,
    summarize_inner_result,
)


def _base_cfg() -> dict:
    return {
        "models": {
            "target": "target-model",
            "draft": "draft-small",
            "draft_devices": ["cuda:1", "cuda:2"],
        },
        "simulation": {
            "num_clients": 4,
        },
        "outer_optimization": {
            "target_model_choices": ["target-model"],
            "draft_model_choices": ["draft-small", "draft-large"],
            "draft_placements": ["edge_device", "shared_gpu_pool"],
            "draft_pool_size_choices": [1],
            "target_replication_choices": [1],
            "model_catalog": {
                "target-model": {"memory_gb": 16.0},
                "draft-small": {"memory_gb": 4.0},
                "draft-large": {"memory_gb": 12.0},
            },
            "budgets": {
                "device_memory_budget_gb": 8.0,
                "server_gpu_memory_budget_gb": 24.0,
                "available_target_gpus": 1,
                "available_draft_gpus": 2,
            },
            "objective": {
                "metric": "normalized_auc",
            },
        },
    }


def _mock_inner_result(auc: float, headline_goodput: float = 10.0) -> dict[str, object]:
    return {
        "auc_summary": {
            "estimated_auc_trapezoidal": auc,
            "auc_trapezoidal": auc,
            "num_frontier_points": 3,
            "num_hull_points": 2,
        },
        "paper_table_df": pd.DataFrame(
            [
                {
                    "policy_family": "adaptive",
                    "auc_trapezoidal": auc,
                    "headline_goodput_tps": headline_goodput,
                }
            ]
        ),
        "frontier_rows": [
            {
                "frontier_semantics": "estimated_best_over_sampled_candidates",
            }
        ],
        "frontier_csv": Path("/tmp/frontier.csv"),
        "paper_table_csv": Path("/tmp/paper_table.csv"),
        "auc_json": Path("/tmp/auc.json"),
    }


def test_parse_outer_objective_uses_interactivity_range_for_normalized_auc() -> None:
    objective = parse_outer_objective_spec(
        _base_cfg(),
        interactivity_targets=[0.0, 2.0, 4.0],
    )

    assert objective.metric == "normalized_auc"
    assert objective.auc_normalizer_x == 4.0


def test_summarize_inner_result_computes_normalized_auc() -> None:
    summary, objective = summarize_inner_result(
        _mock_inner_result(auc=12.0),
        OuterObjectiveSpec(metric="normalized_auc", auc_normalizer_x=4.0),
    )

    assert summary.auc_trapezoidal == 12.0
    assert summary.normalized_auc == 3.0
    assert objective == 3.0


def test_bilevel_solver_selects_best_feasible_configuration(tmp_path: Path) -> None:
    cfg = _base_cfg()
    objective = parse_outer_objective_spec(cfg, interactivity_targets=[0.0, 4.0])

    async def _runner(configuration, _output_dir):
        if configuration.draft_model_id == "draft-small":
            return _mock_inner_result(auc=8.0)
        return _mock_inner_result(auc=20.0)

    solver = BilevelOuterSolver(
        base_cfg=cfg,
        output_dir=tmp_path / "bilevel",
        objective_spec=objective,
        run_inner_problem=_runner,
        clean_output=True,
    )

    result = asyncio.run(solver.solve())
    best = result["best_evaluation"]

    assert best is not None
    assert best.configuration.draft_model_id == "draft-large"
    assert best.inner_summary is not None
    assert best.inner_summary.auc_trapezoidal == 20.0
    assert (tmp_path / "bilevel" / "bilevel_evaluations.csv").exists()
    assert (tmp_path / "bilevel" / "outer_problem.json").exists()
    assert (tmp_path / "bilevel" / "best_bilevel_solution.json").exists()
