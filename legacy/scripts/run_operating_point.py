from __future__ import annotations

import argparse
import asyncio
import json
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


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_config(config_path: Path, cfg: dict) -> None:
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _apply_seed(cfg: dict, seed: int, output_csv: Path) -> dict:
    cfg = dict(cfg)
    cfg["simulation"] = dict(cfg.get("simulation", {}))
    cfg["dataset"] = dict(cfg.get("dataset", {}))
    cfg["simulation"]["seed"] = int(seed)
    cfg["dataset"]["seed"] = int(seed)
    cfg["simulation"]["output_csv"] = str(output_csv)
    return cfg


def _scalar_summary(summary: dict[str, object]) -> dict[str, float | int | str]:
    return {
        key: value
        for key, value in summary.items()
        if not isinstance(value, pd.DataFrame)
    }


async def _run_single_seed(
    config_path: Path,
    output_dir: Path,
    seed: int,
    detailed_log: bool,
) -> dict[str, float | int | str]:
    cfg = _load_config(config_path)
    output_csv = output_dir / f"rounds_seed_{seed}.csv"
    seeded_cfg = _apply_seed(cfg, seed=seed, output_csv=output_csv)
    with tempfile.TemporaryDirectory(prefix=f"edge-specsim-seed-{seed}-") as tmp_dir:
        tmp_config = Path(tmp_dir) / "config.yaml"
        _write_config(tmp_config, seeded_cfg)
        simulator = EdgeSpecSimulator(str(tmp_config), detailed_log=detailed_log)
        await simulator.run()

    df = pd.read_csv(output_csv)
    summary = _scalar_summary(summarize_round_csv(df))
    summary["seed"] = seed
    return summary


def _print_summary_table(results: list[dict[str, float | int | str]]) -> None:
    columns = [
        "seed",
        "goodput_tps",
        "accepted_token_goodput_tps",
        "mean_client_interactivity_s_per_token",
        "min_client_interactivity_s_per_token",
        "mean_ttft_ms",
        "server_utilization",
        "acceptance_rate",
        "straggler_ratio",
    ]
    widths = {
        column: max(len(column), *(len(f"{result[column]}") for result in results))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for result in results:
        print(
            "  ".join(
                str(result[column]).ljust(widths[column]) for column in columns
            )
        )


async def _run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    cfg = _load_config(config_path)
    experiment_cfg = cfg.get("experiment", {})
    seeds = (
        _parse_int_list(args.seeds)
        if args.seeds
        else [int(seed) for seed in experiment_cfg.get("seeds", [cfg["simulation"]["seed"]])]
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, float | int | str]] = []
    for seed in seeds:
        print(f"\n=== Running operating point seed={seed} ===")
        metrics = await _run_single_seed(
            config_path=config_path,
            output_dir=output_dir,
            seed=seed,
            detailed_log=args.detailed_log,
        )
        results.append(metrics)
        print(
            "seed={seed} goodput={goodput_tps:.3f} accepted_goodput={accepted_token_goodput_tps:.3f} "
            "mean_interactivity={mean_client_interactivity_s_per_token:.4f} "
            "min_interactivity={min_client_interactivity_s_per_token:.4f} "
            "TTFT={mean_ttft_ms:.1f}ms".format(**metrics)
        )

    summary_df = pd.DataFrame(results)
    summary_csv = output_dir / "operating_point_summary.csv"
    summary_json = output_dir / "operating_point_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    aggregate = summary_df.mean(numeric_only=True).to_dict()
    aggregate["seed_count"] = len(results)
    aggregate_json = output_dir / "operating_point_aggregate.json"
    with aggregate_json.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    _print_summary_table(results)
    print(f"\nWrote summary CSV to {summary_csv}")
    print(f"Wrote aggregate JSON to {aggregate_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed operating point across one or more seeds, honoring the "
            "experiment warm-up and measurement windows from config."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="results/operating_point")
    parser.add_argument(
        "--seeds",
        help="Comma-separated seed list. Defaults to experiment.seeds from config.",
    )
    parser.add_argument("--detailed-log", action="store_true")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
