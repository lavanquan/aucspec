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

DEFAULT_V_VALUES = [1, 10, 50, 100, 500, 1000]


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


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
    return {
        "V": float(df["controller_V"].iloc[0]),
        "policy": str(df["controller_policy"].iloc[0]),
        **scalar_summary,
        "mean_gamma": float(df["gamma"].mean()),
        "mean_z_queue": float(df["z_queue"].mean()),
        "max_z_queue": float(df["z_queue"].max()),
        "mean_server_queue": float(df["server_queue"].mean()),
        "max_server_queue": float(df["server_queue"].max()),
        "mean_device_queue": float(df["device_queue"].mean()),
        "max_device_queue": float(df["device_queue"].max()),
        "mean_lambda_price": float(df["lambda_price"].mean()),
        "max_lambda_price": float(df["lambda_price"].max()),
    }


def _format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _print_summary_table(results: list[dict[str, float | int | str]]) -> None:
    columns = [
        "V",
        "goodput_tps",
        "accepted_token_goodput_tps",
        "mean_client_interactivity_s_per_token",
        "min_client_interactivity_s_per_token",
        "interactivity_jain_fairness",
        "mean_ttft_ms",
        "mean_round_latency_ms",
        "mean_z_queue",
        "mean_server_queue",
        "mean_device_queue",
        "token_fairness_jain",
        "mean_gamma",
    ]
    widths = {
        column: max(len(column), *(len(_format_metric(result[column])) for result in results))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for result in results:
        print(
            "  ".join(
                _format_metric(result[column]).ljust(widths[column]) for column in columns
            )
        )


def _apply_overrides(
    cfg: dict,
    *,
    V_value: float,
    output_csv: Path,
    seed: int | None,
    dataset_name: str | None,
    num_questions: int | None,
    split: str | None,
    math_subject: str | None,
    target_model: str | None,
    draft_model: str | None,
    num_clients: int | None,
    controller_policy: str | None,
) -> dict:
    cfg = dict(cfg)
    cfg["simulation"] = dict(cfg.get("simulation", {}))
    cfg["controller"] = dict(cfg.get("controller", {}))
    cfg["dataset"] = dict(cfg.get("dataset", {}))
    cfg["models"] = dict(cfg.get("models", {}))

    cfg["controller"]["V"] = float(V_value)
    if controller_policy is not None:
        cfg["controller"]["policy"] = str(controller_policy)
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


async def _run_single_v(
    *,
    config_path: Path,
    V_value: float,
    output_dir: Path,
    seed: int | None,
    dataset_name: str | None,
    num_questions: int | None,
    split: str | None,
    math_subject: str | None,
    target_model: str | None,
    draft_model: str | None,
    num_clients: int | None,
    controller_policy: str | None,
    detailed_log: bool,
) -> dict[str, float | int | str]:
    base_cfg = _load_config(config_path)
    output_csv = output_dir / f"rounds_V_{str(V_value).replace('.', '_')}.csv"
    run_cfg = _apply_overrides(
        base_cfg,
        V_value=V_value,
        output_csv=output_csv,
        seed=seed,
        dataset_name=dataset_name,
        num_questions=num_questions,
        split=split,
        math_subject=math_subject,
        target_model=target_model,
        draft_model=draft_model,
        num_clients=num_clients,
        controller_policy=controller_policy,
    )

    with tempfile.TemporaryDirectory(prefix=f"edge-specsim-v-{V_value}-") as tmp_dir:
        temp_config_path = Path(tmp_dir) / "config.yaml"
        _write_config(temp_config_path, run_cfg)
        simulator = EdgeSpecSimulator(str(temp_config_path), detailed_log=detailed_log)
        await simulator.run()
    return _compute_metrics(output_csv)


async def _run_sweep(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, float | int | str]] = []
    for V_value in args.V_values:
        print(f"\n=== Running controller V={V_value:g} ===")
        metrics = await _run_single_v(
            config_path=Path(args.config),
            V_value=V_value,
            output_dir=output_dir,
            seed=args.seed,
            dataset_name=args.dataset,
            num_questions=args.num_questions,
            split=args.split,
            math_subject=args.math_subject,
            target_model=args.target_model,
            draft_model=args.draft_model,
            num_clients=args.num_clients,
            controller_policy=args.controller_policy,
            detailed_log=args.detailed_log,
        )
        results.append(metrics)
        print(
            "V={V:.0f} goodput={goodput_tps:.3f} token/s "
            "accepted_goodput={accepted_token_goodput_tps:.3f} token/s "
            "min_interactivity={min_client_interactivity_s_per_token:.4f} s/token "
            "interactivity_jain={interactivity_jain_fairness:.3f} "
            "z_mean={mean_z_queue:.3f} server_mean={mean_server_queue:.3f} "
            "device_mean={mean_device_queue:.3f}".format(**metrics)
        )

    summary_csv = output_dir / "controller_v_summary.csv"
    summary_json = output_dir / "controller_v_summary.json"
    pd.DataFrame(results).to_csv(summary_csv, index=False)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\nController V sweep summary")
    _print_summary_table(results)
    print(f"\nWrote summary CSV to {summary_csv}")
    print(f"Wrote summary JSON to {summary_json}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep the Lyapunov V parameter and compare throughput, fairness, and backlog."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="results/controller_v_sweep")
    parser.add_argument(
        "--V-values",
        type=_parse_float_list,
        default=DEFAULT_V_VALUES,
        help="Comma-separated list of controller V values.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", choices=["gsm8k", "math", "cnn_dailymail"])
    parser.add_argument("--num-questions", type=int)
    parser.add_argument("--split")
    parser.add_argument("--math-subject")
    parser.add_argument("--target-model")
    parser.add_argument("--draft-model")
    parser.add_argument("--num-clients", type=int)
    parser.add_argument(
        "--controller-policy",
        choices=[
            "target_only",
            "fixed_gamma",
            "random_gamma",
            "oracle_gamma",
            "adaptive_queue",
            "adaptive_ucb",
        ],
        help="Override controller.policy during the sweep.",
    )
    parser.add_argument("--detailed-log", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    asyncio.run(_run_sweep(args))


if __name__ == "__main__":
    main()
