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


def _scalar_summary(summary: dict[str, object]) -> dict[str, float | int | str]:
    return {
        key: value
        for key, value in summary.items()
        if not isinstance(value, pd.DataFrame)
    }


def _baseline_specs(base_cfg: dict) -> list[dict[str, object]]:
    gamma_choices = [int(v) for v in base_cfg.get("controller", {}).get("gamma_choices", [0, 1, 2, 4, 8])]
    available_fixed = [gamma for gamma in [1, 2, 4, 8] if gamma in gamma_choices]
    learning_window = int(base_cfg.get("controller", {}).get("learning", {}).get("window_size", 64) or 64)
    specs: list[dict[str, object]] = [
        {
            "family": "model",
            "name": "target_only",
            "overrides": {"controller": {"policy": "target_only"}},
            "note": "",
        },
        {
            "family": "model",
            "name": "oracle_gamma",
            "overrides": {"controller": {"policy": "oracle_gamma"}},
            "note": "",
        },
        {
            "family": "controller",
            "name": "adaptive_queue",
            "overrides": {"controller": {"policy": "adaptive_queue"}},
            "note": "",
        },
        {
            "family": "controller",
            "name": "adaptive_ucb",
            "overrides": {"controller": {"policy": "adaptive_ucb"}},
            "note": "",
        },
    ]
    for gamma in available_fixed:
        specs.append(
            {
                "family": "model",
                "name": f"fixed_gamma_{gamma}",
                "overrides": {"controller": {"policy": "fixed_gamma", "fixed_gamma": gamma}},
                "note": "",
            }
        )
    for scheduler in [
        "fcfs",
        "random",
        "equal_tokens",
        "max_expected_accepted_tokens",
        "max_expected_accepted_per_cost",
    ]:
        specs.append(
            {
                "family": "scheduling",
                "name": scheduler,
                "overrides": {"verification_batching": {"scheduler": scheduler}},
                "note": "",
            }
        )
    for allocator in ["equal", "proportional", "queue-weighted"]:
        specs.append(
            {
                "family": "network",
                "name": allocator,
                "overrides": {
                    "clients": {
                        "shared_wireless": {
                            "uplink_allocator": allocator,
                            "downlink_allocator": allocator,
                        }
                    }
                },
                "note": "",
            }
        )
    specs.extend(
        [
            {
                "family": "learning",
                "name": "oracle_true_alpha",
                "overrides": {
                    "controller": {
                        "learning": {
                            "mode": "oracle_true_alpha",
                            "discount": 1.0,
                            "ucb_coefficient": 0.0,
                        }
                    }
                },
                "note": (
                    "Approximated as full-history empirical alpha without UCB/discount; "
                    "the greedy simulator has no latent stochastic alpha oracle."
                ),
            },
            {
                "family": "learning",
                "name": "static_empirical_alpha",
                "overrides": {
                    "controller": {
                        "learning": {
                            "mode": "static_empirical",
                            "discount": 1.0,
                            "ucb_coefficient": 0.0,
                        }
                    }
                },
                "note": "",
            },
            {
                "family": "learning",
                "name": "sliding_window_alpha",
                "overrides": {
                    "controller": {
                        "learning": {
                            "mode": "sliding_window",
                            "window_size": learning_window,
                            "discount": 1.0,
                            "ucb_coefficient": 0.0,
                        }
                    }
                },
                "note": "",
            },
            {
                "family": "learning",
                "name": "bernstein_censored_discounted",
                "overrides": {
                    "controller": {
                        "learning": {
                            "mode": "bernstein_censored_discounted",
                        }
                    }
                },
                "note": "",
            },
        ]
    )
    return specs


def _deep_merge(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


async def _run_single(
    base_cfg: dict,
    baseline: dict[str, object],
    seed: int,
    output_dir: Path,
    detailed_log: bool,
) -> dict[str, object]:
    cfg = json.loads(json.dumps(base_cfg))
    _deep_merge(cfg, baseline["overrides"])
    cfg["simulation"]["seed"] = int(seed)
    cfg["dataset"]["seed"] = int(seed)
    output_csv = output_dir / f"{baseline['family']}__{baseline['name']}__seed_{seed}.csv"
    cfg["simulation"]["output_csv"] = str(output_csv)
    with tempfile.TemporaryDirectory(prefix=f"baseline-{baseline['name']}-seed-{seed}-") as tmp_dir:
        temp_config = Path(tmp_dir) / "config.yaml"
        _write_config(temp_config, cfg)
        simulator = EdgeSpecSimulator(str(temp_config), detailed_log=detailed_log)
        await simulator.run()

    df = pd.read_csv(output_csv)
    metrics = _scalar_summary(summarize_round_csv(df))
    metrics.update(
        {
            "baseline_family": str(baseline["family"]),
            "baseline_name": str(baseline["name"]),
            "seed": int(seed),
            "baseline_note": str(baseline["note"]),
            "controller_policy": str(df["controller_policy"].iloc[0]),
            "verification_scheduler": str(df["verification_scheduler"].iloc[0]),
            "uplink_allocator": str(df["uplink_allocator"].iloc[0]),
            "downlink_allocator": str(df["downlink_allocator"].iloc[0]),
        }
    )
    return metrics


def _aggregate(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    grouped_rows: list[dict[str, object]] = []
    for (family, name), group in df.groupby(["baseline_family", "baseline_name"], sort=False):
        aggregate = group.mean(numeric_only=True).to_dict()
        aggregate.update(
            {
                "baseline_family": str(family),
                "baseline_name": str(name),
                "seed_count": int(len(group)),
                "baseline_note": str(group["baseline_note"].iloc[0]),
                "controller_policy": str(group["controller_policy"].iloc[0]),
                "verification_scheduler": str(group["verification_scheduler"].iloc[0]),
                "uplink_allocator": str(group["uplink_allocator"].iloc[0]),
                "downlink_allocator": str(group["downlink_allocator"].iloc[0]),
            }
        )
        grouped_rows.append(aggregate)
    return pd.DataFrame(grouped_rows)


async def _run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    base_cfg = _load_config(config_path)
    seeds = (
        _parse_int_list(args.seeds)
        if args.seeds
        else [int(seed) for seed in base_cfg.get("experiment", {}).get("seeds", [base_cfg["simulation"]["seed"]])]
    )
    baselines = _baseline_specs(base_cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, object]] = []
    for baseline in baselines:
        print(f"\n=== Baseline {baseline['family']} / {baseline['name']} ===")
        for seed in seeds:
            metrics = await _run_single(
                base_cfg=base_cfg,
                baseline=baseline,
                seed=seed,
                output_dir=output_dir,
                detailed_log=args.detailed_log,
            )
            per_seed_rows.append(metrics)
            print(
                "{baseline_name} seed={seed} goodput={goodput_tps:.3f} "
                "fairness={interactivity_jain_fairness:.3f} TTFT={mean_ttft_ms:.1f}ms".format(
                    **metrics
                )
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    summary_df = _aggregate(per_seed_rows)
    per_seed_csv = output_dir / "baseline_per_seed.csv"
    summary_csv = output_dir / "baseline_summary.csv"
    summary_json = output_dir / "baseline_summary.json"
    per_seed_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary_df.to_dict(orient="records"), handle, indent=2)
    print(f"\nWrote per-seed baseline CSV to {per_seed_csv}")
    print(f"Wrote baseline summary CSV to {summary_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run required baseline families on a fixed operating point and summarize them."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="results/baselines")
    parser.add_argument("--seeds", help="Comma-separated seeds. Defaults to experiment.seeds.")
    parser.add_argument("--detailed-log", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
