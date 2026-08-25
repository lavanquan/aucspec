from __future__ import annotations

import argparse
import asyncio
import compileall
import importlib
import math
import sys
import tempfile
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from edge_specsim.acceptance import (  # noqa: E402
    acceptance_probability,
    greedy_correction_token,
    residual_distribution,
    sample_correction_token,
)
from edge_specsim.bilevel import (  # noqa: E402
    BilevelOuterSolver,
    parse_outer_objective_spec,
)
from edge_specsim.channel import current_snr_db  # noqa: E402
from edge_specsim.configuration import (  # noqa: E402
    Configuration,
    apply_configuration,
    build_configuration_space,
    configuration_feasibility,
)
from edge_specsim.controller import OnlineController, paper_index_gamma  # noqa: E402
from edge_specsim.expectations import (  # noqa: E402
    expected_useful_tokens,
    expected_useful_tokens_from_profile,
)
from edge_specsim.models import ClientProfile  # noqa: E402
from edge_specsim.network import (  # noqa: E402
    estimate_uplink_bits_per_token,
    estimate_uplink_bytes_from_gamma,
    paper_square_root_allocation_weight,
)
from edge_specsim.roofline_budget import RooflineBudgetModel, RooflineKneeEntry  # noqa: E402


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    traceback_text: str | None = None


def _make_client() -> ClientProfile:
    return ClientProfile(
        client_id=0,
        prompt_queue=deque(),
        draft_model_id="draft",
        draft_device_class="medium",
        draft_tokens_per_second=32.0,
        draft_fixed_latency_ms=8.0,
        rtt_ms=40.0,
        uplink_mbps=20.0,
        downlink_mbps=40.0,
        uplink_base_snr_db=12.0,
        downlink_base_snr_db=14.0,
        uplink_snr_jitter_db=1.5,
        downlink_snr_jitter_db=1.5,
        snr_period_ms=1000.0,
        snr_phase_rad=0.0,
        wireless_weight=1.0,
        packet_loss=0.0,
        channel_model="iid_block_fading",
        snr_block_duration_ms=100.0,
        channel_seed=7,
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


def _run_compile_check() -> str:
    src_ok = compileall.compile_dir(str(ROOT / "src"), quiet=1)
    scripts_ok = compileall.compile_dir(str(ROOT / "scripts"), quiet=1)
    tests_ok = compileall.compile_dir(str(ROOT / "tests"), quiet=1)
    if not (src_ok and scripts_ok and tests_ok):
        raise AssertionError("compileall reported at least one failure")
    return "compileall passed for src/, scripts/, tests/"


def _run_import_check() -> str:
    modules = [
        "edge_specsim.controller",
        "edge_specsim.network",
        "edge_specsim.verification_queue",
        "edge_specsim.roofline_budget",
        "edge_specsim.metrics",
        "edge_specsim.bilevel",
    ]
    for module_name in modules:
        importlib.import_module(module_name)
    return f"imported {len(modules)} core modules"


def _run_acceptance_check() -> str:
    assert acceptance_probability(math.log(0.25), math.log(0.25)) == 1.0
    assert acceptance_probability(math.log(0.40), math.log(0.20)) == 1.0
    expected = 0.20 / 0.50
    assert math.isclose(
        acceptance_probability(math.log(0.20), math.log(0.50)),
        expected,
    )
    target_logprobs = {
        0: math.log(0.50),
        1: math.log(0.30),
        2: math.log(0.20),
    }
    draft_logprobs = {
        0: math.log(0.10),
        1: math.log(0.80),
        2: math.log(0.10),
    }
    assert greedy_correction_token(target_logprobs) == 0
    residual = residual_distribution(target_logprobs, draft_logprobs)
    assert 1 not in residual
    assert math.isclose(sum(residual.values()), 1.0)
    assert sample_correction_token(target_logprobs, draft_logprobs, 0.10) == 0
    return "acceptance algebra and correction sampling look consistent"


def _run_expectation_check() -> str:
    assert expected_useful_tokens(0, 0.8) == 1.0
    assert math.isclose(expected_useful_tokens(2, 0.5), 1.75)
    assert math.isclose(
        expected_useful_tokens_from_profile([0.9, 0.8, 0.5]),
        1.0 + 0.9 + 0.9 * 0.8 + 0.9 * 0.8 * 0.5,
    )
    return "phi(gamma, alpha) and positional expectation are correct"


def _run_controller_check() -> str:
    client = _make_client()
    client.alpha_ucb = 0.8
    gamma_controller = OnlineController(
        gamma_choices=[0, 1, 2, 4],
        V=100.0,
        min_tps=1.0,
        server_price_scale=1.0,
        policy="adaptive_index",
        use_virtual_queues=True,
        slot_ms=100.0,
    )
    gamma_controller.server_queue = 10.0
    gamma_controller.refresh_server_price()
    gamma_controller.set_verifier_theta_f_signal(0.1)
    gamma_low_theta_f = gamma_controller.choose_gamma(client, total_rounds=10)
    gamma_controller.set_verifier_theta_f_signal(100.0)
    gamma_high_theta_f = gamma_controller.choose_gamma(client, total_rounds=10)
    assert gamma_low_theta_f >= gamma_high_theta_f

    queue_controller = OnlineController(
        gamma_choices=[0, 1],
        V=1.0,
        min_tps=1.0,
        server_price_scale=1.0,
        policy="adaptive_index",
        use_virtual_queues=True,
        slot_ms=100.0,
    )
    server_queue = queue_controller.observe_server_batch(
        batch_id=1,
        verify_finish_ms=50.0,
        theta_f_ms_per_token=10.0,
        verifier_token_cost=18,
    )
    assert math.isclose(server_queue, 80.0)
    device_queue = queue_controller.observe_device_service(
        client_id=client.client_id,
        edge_finish_ms=50.0,
        gamma=4,
        tau_d_ms=20.0,
        kappa_over_r_ms=15.0,
    )
    assert math.isclose(device_queue, 40.0)
    assert paper_index_gamma(alpha_hat=0.8, c_value=0.8**3, gamma_max=8) == 2
    return (
        f"adaptive_index reacts to theta_f; queue updates give "
        f"Q={server_queue:.1f}, Qi={device_queue:.1f}"
    )


def _run_network_check() -> str:
    kappa = estimate_uplink_bits_per_token(distribution_payload="delta_proposal")
    expected_bits = 8.0 * (
        estimate_uplink_bytes_from_gamma(1, distribution_payload="delta_proposal")
        - estimate_uplink_bytes_from_gamma(0, distribution_payload="delta_proposal")
    )
    assert kappa == expected_bits
    weight = paper_square_root_allocation_weight(
        queue_price=9.0,
        payload_bits_per_token=4.0,
        gamma=4,
        spectral_efficiency=1.0,
    )
    assert math.isclose(weight, 12.0)
    return f"kappa={kappa:.1f} bits/token; square-root allocator matches closed form"


def _run_roofline_check() -> str:
    model = RooflineBudgetModel(
        [
            RooflineKneeEntry(
                batch_size=4,
                context_length=512,
                knee_verifier_tokens=20,
                theta_0_ms=5.0,
                theta_m_ms=40.0,
                theta_f_ms_per_token=2.0,
            ),
        ]
    )
    assert model.signal_for(4, 512, queued_verifier_tokens=10).regime == "memory_bound"
    assert model.signal_for(4, 512, queued_verifier_tokens=20).regime == "roofline_knee"
    assert model.signal_for(4, 512, queued_verifier_tokens=30).regime == "compute_bound"
    assert model.service_time_ms(4, 512, total_verifier_tokens=10) == 45.0
    assert model.service_time_ms(4, 512, total_verifier_tokens=30) == 65.0
    return "roofline knee and service model match paper form"


def _run_channel_check() -> str:
    client = _make_client()
    same_block_a = current_snr_db(client, "uplink", 10.0)
    same_block_b = current_snr_db(client, "uplink", 90.0)
    next_block = current_snr_db(client, "uplink", 150.0)
    assert same_block_a == same_block_b
    assert same_block_a != next_block
    return (
        f"iid_block_fading is stable within a block and changes across blocks "
        f"({same_block_a:.2f} dB -> {next_block:.2f} dB)"
    )


def _run_configuration_bilevel_check() -> str:
    cfg = _base_cfg()
    space = build_configuration_space(cfg)
    assert len(space) == 4
    infeasible = Configuration(
        draft_model_id="draft-large",
        target_model_id="target-model",
        draft_placement="edge_device",
        draft_pool_size=1,
        target_replication_factor=1,
    )
    assert not configuration_feasibility(cfg, infeasible).feasible
    feasible = Configuration(
        draft_model_id="draft-large",
        target_model_id="target-model",
        draft_placement="shared_gpu_pool",
        draft_pool_size=1,
        target_replication_factor=1,
    )
    assert configuration_feasibility(cfg, feasible).feasible
    configured = apply_configuration(cfg, feasible)
    assert configured["models"]["draft"] == "draft-large"
    assert configured["simulation"]["draft_execution_mode"] == "shared_gpu_emulation"

    objective = parse_outer_objective_spec(cfg, interactivity_targets=[0.0, 4.0])

    async def _runner(configuration: Configuration, _output_dir: Path) -> dict[str, object]:
        auc = 20.0 if configuration.draft_model_id == "draft-large" else 8.0
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
                        "headline_goodput_tps": 10.0,
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

    async def _solve() -> tuple[str, float]:
        with tempfile.TemporaryDirectory(prefix="quick-bilevel-") as tmp_dir:
            solver = BilevelOuterSolver(
                base_cfg=cfg,
                output_dir=Path(tmp_dir),
                objective_spec=objective,
                run_inner_problem=_runner,
                clean_output=True,
            )
            result = await solver.solve()
            best = result["best_evaluation"]
            assert best is not None
            return best.configuration.slug(), float(best.objective_value)

    best_slug, best_objective = asyncio.run(_solve())
    return f"bilevel outer solver selects {best_slug} with objective={best_objective:.3f}"


def _run_model_check(
    *,
    config_path: Path,
    num_questions: int,
    verbose: bool,
) -> str:
    validate_module = importlib.import_module("validate_correctness")
    cfg = validate_module._load_config(str(config_path))
    cfg = validate_module._apply_overrides(
        cfg,
        dataset_name=None,
        num_questions=num_questions,
        split=None,
        math_subject=None,
        target_model=None,
        draft_model=None,
    )
    asyncio.run(validate_module._validate_samples(cfg, verbose=verbose))
    return f"validated greedy correctness on {num_questions} sample(s) with real models"


def _execute_check(name: str, fn: Callable[[], str]) -> CheckResult:
    try:
        detail = fn()
        return CheckResult(name=name, ok=True, detail=detail)
    except Exception as exc:  # noqa: BLE001
        detail = f"{exc.__class__.__name__}: {exc}"
        return CheckResult(
            name=name,
            ok=False,
            detail=detail,
            traceback_text=traceback.format_exc(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fast smoke check for the project. "
            "By default this avoids loading models; add --with-models for a tiny end-to-end check."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--with-models", action="store_true")
    parser.add_argument("--num-questions", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    checks: list[tuple[str, Callable[[], str]]] = [
        ("compile", _run_compile_check),
        ("imports", _run_import_check),
        ("acceptance", _run_acceptance_check),
        ("expectations", _run_expectation_check),
        ("controller", _run_controller_check),
        ("network", _run_network_check),
        ("roofline", _run_roofline_check),
        ("channel", _run_channel_check),
        ("bilevel", _run_configuration_bilevel_check),
    ]
    if args.with_models:
        checks.append(
            (
                "models",
                lambda: _run_model_check(
                    config_path=ROOT / args.config,
                    num_questions=max(1, int(args.num_questions)),
                    verbose=args.verbose,
                ),
            )
        )

    results: list[CheckResult] = []
    print("Quick check started")
    print(f"project_root={ROOT}")
    print(f"with_models={args.with_models}")
    for name, fn in checks:
        result = _execute_check(name, fn)
        results.append(result)
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {name}: {result.detail}")
        if (not result.ok) and args.verbose and result.traceback_text:
            print(result.traceback_text.rstrip())

    passed = sum(1 for result in results if result.ok)
    failed = len(results) - passed
    print(f"\nSummary: {passed}/{len(results)} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
