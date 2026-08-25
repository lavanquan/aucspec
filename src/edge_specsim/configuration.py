from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelCatalogEntry:
    model_id: str
    memory_gb: float


@dataclass(frozen=True)
class ConfigurationBudget:
    device_memory_budget_gb: float
    server_gpu_memory_budget_gb: float
    available_target_gpus: int
    available_draft_gpus: int
    num_clients: int


@dataclass(frozen=True)
class Configuration:
    draft_model_id: str
    draft_placement: str
    draft_pool_size: int
    target_replication_factor: int
    target_model_id: str | None = None

    def slug(self) -> str:
        target_suffix = (
            ""
            if self.target_model_id is None
            else f"__target={self.target_model_id.split('/')[-1]}"
        )
        return (
            f"draft={self.draft_model_id.split('/')[-1]}"
            f"__placement={self.draft_placement}"
            f"__draftpool={self.draft_pool_size}"
            f"__targetrep={self.target_replication_factor}"
            f"{target_suffix}"
        )

    def to_overrides(self, base_cfg: dict[str, Any]) -> dict[str, Any]:
        cfg = {
            "models": dict(base_cfg.get("models", {})),
            "simulation": dict(base_cfg.get("simulation", {})),
            "verification_batching": dict(base_cfg.get("verification_batching", {})),
        }
        cfg["models"]["draft"] = self.draft_model_id
        if self.target_model_id is not None:
            cfg["models"]["target"] = self.target_model_id
        draft_devices = list(cfg["models"].get("draft_devices", []))
        if not draft_devices:
            raise ValueError("models.draft_devices must contain at least one device")
        cfg["models"]["draft_devices"] = draft_devices[: max(1, int(self.draft_pool_size))]
        if len(cfg["models"]["draft_devices"]) < max(1, int(self.draft_pool_size)):
            raise ValueError(
                "Configuration requested more draft replicas than the configured draft device list"
            )
        placement = self.normalized_placement()
        cfg["simulation"]["draft_execution_mode"] = (
            "simulation" if placement == "edge_device" else "shared_gpu_emulation"
        )
        cfg["verification_batching"]["target_replication_factor"] = max(
            1, int(self.target_replication_factor)
        )
        return cfg

    def normalized_placement(self) -> str:
        placement = str(self.draft_placement).strip().lower()
        if placement not in {"edge_device", "shared_gpu_pool"}:
            raise ValueError(
                "draft_placement must be one of: edge_device, shared_gpu_pool"
            )
        return placement


@dataclass(frozen=True)
class ConfigurationFeasibility:
    feasible: bool
    reason: str


def _catalog_from_config(cfg: dict[str, Any]) -> dict[str, ModelCatalogEntry]:
    outer_cfg = cfg.get("outer_optimization", {})
    catalog_cfg = outer_cfg.get("model_catalog", {})
    catalog: dict[str, ModelCatalogEntry] = {}
    for model_id, spec in catalog_cfg.items():
        catalog[str(model_id)] = ModelCatalogEntry(
            model_id=str(model_id),
            memory_gb=float(spec.get("memory_gb", 0.0)),
        )
    return catalog


def _budget_from_config(cfg: dict[str, Any]) -> ConfigurationBudget:
    outer_cfg = cfg.get("outer_optimization", {})
    budget_cfg = outer_cfg.get("budgets", {})
    simulation_cfg = cfg.get("simulation", {})
    models_cfg = cfg.get("models", {})
    draft_devices = list(models_cfg.get("draft_devices", []))
    return ConfigurationBudget(
        device_memory_budget_gb=float(budget_cfg.get("device_memory_budget_gb", 8.0)),
        server_gpu_memory_budget_gb=float(budget_cfg.get("server_gpu_memory_budget_gb", 24.0)),
        available_target_gpus=int(budget_cfg.get("available_target_gpus", 1)),
        available_draft_gpus=int(budget_cfg.get("available_draft_gpus", len(draft_devices))),
        num_clients=int(simulation_cfg.get("num_clients", 1)),
    )


def build_configuration_space(cfg: dict[str, Any]) -> list[Configuration]:
    outer_cfg = cfg.get("outer_optimization", {})
    models_cfg = cfg.get("models", {})
    target_model_choices = [
        str(value)
        for value in outer_cfg.get("target_model_choices", [models_cfg.get("target")])
        if value
    ]
    draft_model_choices = [
        str(value)
        for value in outer_cfg.get("draft_model_choices", [models_cfg.get("draft")])
        if value
    ]
    draft_placements = [
        str(value) for value in outer_cfg.get("draft_placements", ["edge_device", "shared_gpu_pool"])
    ]
    target_replication_choices = [
        int(value) for value in outer_cfg.get("target_replication_choices", [1])
    ]
    max_draft_pool = max(1, len(models_cfg.get("draft_devices", [])))
    draft_pool_sizes = [
        int(value)
        for value in outer_cfg.get("draft_pool_size_choices", list(range(1, max_draft_pool + 1)))
    ]

    configurations: list[Configuration] = []
    for target_model_id in target_model_choices:
        for draft_model_id in draft_model_choices:
            for draft_placement in draft_placements:
                for draft_pool_size in draft_pool_sizes:
                    for target_replication_factor in target_replication_choices:
                        configurations.append(
                            Configuration(
                                draft_model_id=draft_model_id,
                                target_model_id=target_model_id,
                                draft_placement=draft_placement,
                                draft_pool_size=draft_pool_size,
                                target_replication_factor=target_replication_factor,
                            )
                        )
    return configurations


def configuration_feasibility(
    cfg: dict[str, Any],
    configuration: Configuration,
) -> ConfigurationFeasibility:
    catalog = _catalog_from_config(cfg)
    budget = _budget_from_config(cfg)
    target_model_id = configuration.target_model_id or str(cfg.get("models", {}).get("target"))

    if configuration.draft_model_id not in catalog:
        return ConfigurationFeasibility(
            feasible=False,
            reason=f"draft model {configuration.draft_model_id!r} is missing from outer_optimization.model_catalog",
        )
    if target_model_id not in catalog:
        return ConfigurationFeasibility(
            feasible=False,
            reason=f"target model {target_model_id!r} is missing from outer_optimization.model_catalog",
        )

    draft_memory_gb = float(catalog[configuration.draft_model_id].memory_gb)
    target_memory_gb = float(catalog[target_model_id].memory_gb)
    placement = configuration.normalized_placement()

    if configuration.target_replication_factor <= 0:
        return ConfigurationFeasibility(False, "target_replication_factor must be positive")
    if configuration.draft_pool_size <= 0:
        return ConfigurationFeasibility(False, "draft_pool_size must be positive")
    if configuration.target_replication_factor > budget.available_target_gpus:
        return ConfigurationFeasibility(
            False,
            "target replication exceeds available target GPUs",
        )
    if target_memory_gb > budget.server_gpu_memory_budget_gb:
        return ConfigurationFeasibility(
            False,
            "target model does not fit within the per-GPU server memory budget",
        )
    if configuration.draft_pool_size > budget.available_draft_gpus:
        return ConfigurationFeasibility(
            False,
            "draft pool size exceeds available draft GPUs",
        )

    if placement == "edge_device":
        if draft_memory_gb > budget.device_memory_budget_gb:
            return ConfigurationFeasibility(
                False,
                "draft model does not fit within the edge device memory budget",
            )
    elif placement == "shared_gpu_pool":
        if draft_memory_gb > budget.server_gpu_memory_budget_gb:
            return ConfigurationFeasibility(
                False,
                "draft model does not fit within the shared draft GPU memory budget",
            )

    return ConfigurationFeasibility(True, "feasible")


def apply_configuration(base_cfg: dict[str, Any], configuration: Configuration) -> dict[str, Any]:
    cfg = dict(base_cfg)
    for section, overrides in configuration.to_overrides(base_cfg).items():
        cfg[section] = dict(cfg.get(section, {}))
        cfg[section].update(overrides)
    cfg.setdefault("outer_optimization", {})
    cfg["outer_optimization"]["selected_configuration"] = asdict(configuration)
    return cfg
