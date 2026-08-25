from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_specsim.metrics import summarize_round_csv
from edge_specsim.simulator import EdgeSpecSimulator

DEFAULT_SCHEDULERS = [
    "fcfs",
    "random",
    "equal_tokens",
    "max_expected_accepted_tokens",
    "max_throughput",
    "max_expected_accepted_per_cost",
    "weighted_utility",
    "knapsack",
]


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_config(config_path: Path, cfg: dict) -> None:
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def _compute_metrics(csv_path: Path) -> dict[str, float | int | str]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rounds found in {csv_path}")
    summary = summarize_round_csv(df)
    scalar_summary = {
        key: value
        for key, value in summary.items()
        if not isinstance(value, pd.DataFrame)
    }
    batch_count = int(df["verification_batch_id"].nunique())
    scheduler_name = str(df["verification_scheduler"].iloc[0])

    return {
        "scheduler": scheduler_name,
        "rounds": int(len(df)),
        "batches": batch_count,
        **scalar_summary,
        "verify_token_cost": int(df["verification_batch_token_cost"].sum()),
    }


def _format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _print_summary_table(results: list[dict[str, float | int | str]]) -> None:
    columns = [
        "scheduler",
        "goodput_tps",
        "accepted_token_goodput_tps",
        "mean_client_interactivity_s_per_token",
        "min_client_interactivity_s_per_token",
        "interactivity_jain_fairness",
        "mean_ttft_ms",
        "mean_round_latency_ms",
        "p95_round_latency_ms",
        "token_fairness_jain",
        "interactivity_fairness_jain",
        "throughput_fairness_jain",
        "useful_tokens",
        "batches",
    ]
    widths = {
        column: max(len(column), *(len(_format_metric(result[column])) for result in results))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    divider = "  ".join("-" * widths[column] for column in columns)
    print(header)
    print(divider)
    for result in results:
        print(
            "  ".join(
                _format_metric(result[column]).ljust(widths[column]) for column in columns
            )
        )


def _apply_overrides(
    cfg: dict,
    *,
    scheduler_name: str,
    utility_lambda: float,
    output_csv: Path,
    seed: int | None,
    dataset_name: str | None,
    num_questions: int | None,
    split: str | None,
    math_subject: str | None,
    target_model: str | None,
    draft_model: str | None,
    num_clients: int | None,
) -> dict:
    cfg = dict(cfg)
    cfg["simulation"] = dict(cfg.get("simulation", {}))
    cfg["verification_batching"] = dict(cfg.get("verification_batching", {}))
    cfg["dataset"] = dict(cfg.get("dataset", {}))
    cfg["models"] = dict(cfg.get("models", {}))

    cfg["verification_batching"]["scheduler"] = scheduler_name
    cfg["verification_batching"]["utility_lambda"] = float(utility_lambda)
    cfg["simulation"]["output_csv"] = str(output_csv)
    if seed is not None:
        cfg["simulation"]["seed"] = int(seed)
        cfg["dataset"]["seed"] = int(seed)
    if num_clients is not None:
        cfg["simulation"]["num_clients"] = int(num_clients)
    if dataset_name is not None:
        cfg["dataset"]["name"] = dataset_name
    if num_questions is not None:
        cfg["dataset"]["num_questions"] = int(num_questions)
    if split is not None:
        cfg["dataset"]["split"] = split
    if math_subject is not None:
        cfg["dataset"]["math_subject"] = math_subject
    if target_model is not None:
        cfg["models"]["target"] = target_model
    if draft_model is not None:
        cfg["models"]["draft"] = draft_model
    return cfg


async def _run_single_scheduler(
    *,
    config_path: Path,
    scheduler_name: str,
    utility_lambda: float,
    output_dir: Path,
    seed: int | None,
    dataset_name: str | None,
    num_questions: int | None,
    split: str | None,
    math_subject: str | None,
    target_model: str | None,
    draft_model: str | None,
    num_clients: int | None,
    detailed_log: bool,
) -> dict[str, float | int | str]:
    base_cfg = _load_config(config_path)
    output_csv = output_dir / f"rounds_{scheduler_name}.csv"
    run_cfg = _apply_overrides(
        base_cfg,
        scheduler_name=scheduler_name,
        utility_lambda=utility_lambda,
        output_csv=output_csv,
        seed=seed,
        dataset_name=dataset_name,
        num_questions=num_questions,
        split=split,
        math_subject=math_subject,
        target_model=target_model,
        draft_model=draft_model,
        num_clients=num_clients,
    )

    with tempfile.TemporaryDirectory(prefix=f"edge-specsim-{scheduler_name}-") as tmp_dir:
        temp_config_path = Path(tmp_dir) / "config.yaml"
        _write_config(temp_config_path, run_cfg)
        simulator = EdgeSpecSimulator(
            str(temp_config_path),
            detailed_log=detailed_log,
        )
        await simulator.run()
    return _compute_metrics(output_csv)


async def _run_sweep(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, float | int | str]] = []
    for scheduler_name in args.schedulers:
        print(f"\n=== Running scheduler={scheduler_name} ===")
        metrics = await _run_single_scheduler(
            config_path=Path(args.config),
            scheduler_name=scheduler_name,
            utility_lambda=args.utility_lambda,
            output_dir=output_dir,
            seed=args.seed,
            dataset_name=args.dataset,
            num_questions=args.num_questions,
            split=args.split,
            math_subject=args.math_subject,
            target_model=args.target_model,
            draft_model=args.draft_model,
            num_clients=args.num_clients,
            detailed_log=args.detailed_log,
        )
        results.append(metrics)
        print(
            "scheduler={scheduler} goodput={goodput_tps:.3f} token/s "
            "accepted_goodput={accepted_token_goodput_tps:.3f} token/s "
            "min_interactivity={min_client_interactivity_s_per_token:.4f} s/token "
            "interactivity_jain={interactivity_jain_fairness:.3f} "
            "mean_latency={mean_round_latency_ms:.1f} ms "
            "p95_latency={p95_round_latency_ms:.1f} ms "
            "fairness={token_fairness_jain:.3f}".format(**metrics)
        )

    summary_csv = output_dir / "scheduler_summary.csv"
    summary_json = output_dir / "scheduler_summary.json"
    pd.DataFrame(results).to_csv(summary_csv, index=False)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\nScheduler sweep summary")
    _print_summary_table(results)
    print(f"\nWrote summary CSV to {summary_csv}")
    print(f"Wrote summary JSON to {summary_json}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the simulator across multiple verification schedulers and compare metrics."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--schedulers",
        nargs="+",
        default=DEFAULT_SCHEDULERS,
        choices=DEFAULT_SCHEDULERS,
        help="Schedulers to compare on the same base config/seed.",
    )
    parser.add_argument("--output-dir", default="results/scheduler_sweep")
    parser.add_argument("--seed", type=int, help="Override simulation and dataset seed")
    parser.add_argument("--dataset", choices=["gsm8k", "math", "cnn_dailymail"])
    parser.add_argument("--num-questions", type=int)
    parser.add_argument("--split")
    parser.add_argument("--math-subject")
    parser.add_argument("--target-model")
    parser.add_argument("--draft-model")
    parser.add_argument("--num-clients", type=int)
    parser.add_argument(
        "--utility-lambda",
        type=float,
        default=0.25,
        help="Penalty coefficient used by weighted_utility and knapsack.",
    )
    parser.add_argument(
        "--detailed-log",
        action="store_true",
        help="Preserve the full per-round traces in each scheduler CSV.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the output directory before writing new sweep results.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    asyncio.run(_run_sweep(args))


if __name__ == "__main__":
    main()
