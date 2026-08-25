from __future__ import annotations

from edge_specsim.configuration import (
    Configuration,
    build_configuration_space,
    configuration_feasibility,
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
            "draft_pool_size_choices": [1, 2],
            "target_replication_choices": [1, 2],
            "model_catalog": {
                "target-model": {"memory_gb": 16.0},
                "draft-small": {"memory_gb": 4.0},
                "draft-large": {"memory_gb": 12.0},
            },
            "budgets": {
                "device_memory_budget_gb": 8.0,
                "server_gpu_memory_budget_gb": 24.0,
                "available_target_gpus": 2,
                "available_draft_gpus": 2,
            },
        },
    }


def test_build_configuration_space_enumerates_outer_choices() -> None:
    space = build_configuration_space(_base_cfg())

    assert len(space) == 16


def test_edge_device_configuration_enforces_device_memory_budget() -> None:
    cfg = _base_cfg()
    configuration = Configuration(
        draft_model_id="draft-large",
        target_model_id="target-model",
        draft_placement="edge_device",
        draft_pool_size=1,
        target_replication_factor=1,
    )

    feasibility = configuration_feasibility(cfg, configuration)

    assert not feasibility.feasible
    assert "edge device memory budget" in feasibility.reason


def test_shared_gpu_configuration_can_be_feasible_under_server_budget() -> None:
    cfg = _base_cfg()
    configuration = Configuration(
        draft_model_id="draft-large",
        target_model_id="target-model",
        draft_placement="shared_gpu_pool",
        draft_pool_size=2,
        target_replication_factor=2,
    )

    feasibility = configuration_feasibility(cfg, configuration)

    assert feasibility.feasible
