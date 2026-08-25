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


def _deep_merge(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def _scalar_summary(summary: dict[str, object]) -> dict[str, float | int | str]:
    return {
        key: value
        for key, value in summary.items()
        if not isinstance(value, pd.DataFrame)
    }


def _ablation_specs() -> list[dict[str, object]]:
    return [
        {
            "family": "kv_cache",
            "off_name": "no_kv_cache",
            "on_name": "with_kv_cache",
            "off_overrides": {
                "models": {
                    "draft_enable_kv_cache": False,
                    "target_enable_prefix_caching": False,
                }
            },
            "on_overrides": {
                "models": {
                    "draft_enable_kv_cache": True,
                    "target_enable_prefix_caching": True,
                }
            },
        },
        {
            "family": "network_delay",
            "off_name": "no_network_delay",
            "on_name": "with_network_delay",
            "off_overrides": {
                "clients": {
                    "rtt_ms_range": [0.0, 0.0],
                    "uplink_mbps_range": [1_000_000.0, 1_000_000.0],
                    "downlink_mbps_range": [1_000_000.0, 1_000_000.0],
                    "packet_loss_range": [0.0, 0.0],
                    "uplink_base_snr_db_range": [80.0, 80.0],
                    "downlink_base_snr_db_range": [80.0, 80.0],
                    "uplink_snr_jitter_db_range": [0.0, 0.0],
                    "downlink_snr_jitter_db_range": [0.0, 0.0],
                    "shared_wireless": {
                        "total_uplink_bandwidth_mhz": 1_000_000.0,
                        "total_downlink_bandwidth_mhz": 1_000_000.0,
                    },
                }
            },
            "on_overrides": {},
        },
        {
            "family": "gamma_policy",
            "off_name": "fixed_gamma",
            "on_name": "adaptive_gamma",
            "off_overrides": {
                "controller": {
                    "policy": "fixed_gamma",
                    "fixed_gamma": 4,
                }
            },
            "on_overrides": {
                "controller": {
                    "policy": "adaptive_ucb",
                }
            },
        },
        {
            "family": "ucb",
            "off_name": "no_ucb",
            "on_name": "with_ucb",
            "off_overrides": {
                "controller": {
                    "policy": "adaptive_queue",
                    "learning": {
                        "mode": "static_empirical",
                        "discount": 1.0,
                        "ucb_coefficient": 0.0,
                    },
                }
            },
            "on_overrides": {
                "controller": {
                    "policy": "adaptive_ucb",
                    "learning": {
                        "mode": "bernstein_censored_discounted",
                    },
                }
            },
        },
        {
            "family": "virtual_queue",
            "off_name": "no_virtual_queue",
            "on_name": "with_virtual_queue",
            "off_overrides": {
                "controller": {
                    "use_virtual_queues": False,
                }
            },
            "on_overrides": {
                "controller": {
                    "use_virtual_queues": True,
                }
            },
        },
        {
            "family": "batching",
            "off_name": "fcfs_batching",
            "on_name": "utility_batching",
            "off_overrides": {
                "verification_batching": {
                    "scheduler": "fcfs",
                }
            },
            "on_overrides": {
                "verification_batching": {
                    "scheduler": "weighted_utility",
                }
            },
        },
        {
            "family": "alpha_profile",
            "off_name": "prefix_level_alpha",
            "on_name": "position_level_alpha",
            "off_overrides": {
                "controller": {
                    "learning": {
                        "profile_mode": "prefix",
                    }
                }
            },
            "on_overrides": {
                "controller": {
                    "learning": {
                        "profile_mode": "position",
                    }
                }
            },
        },
    ]


async def _run_single(
    base_cfg: dict,
    family: str,
    variant_name: str,
    overrides: dict[str, object],
    seed: int,
    output_dir: Path,
    detailed_log: bool,
) -> dict[str, object]:
    cfg = json.loads(json.dumps(base_cfg))
    _deep_merge(cfg, overrides)
    cfg["simulation"]["seed"] = int(seed)
    cfg["dataset"]["seed"] = int(seed)
    output_csv = output_dir / f"{family}__{variant_name}__seed_{seed}.csv"
    cfg["simulation"]["output_csv"] = str(output_csv)
    with tempfile.TemporaryDirectory(prefix=f"ablation-{family}-{variant_name}-{seed}-") as tmp_dir:
        temp_config = Path(tmp_dir) / "config.yaml"
        _write_config(temp_config, cfg)
        simulator = EdgeSpecSimulator(str(temp_config), detailed_log=detailed_log)
        await simulator.run()

    df = pd.read_csv(output_csv)
    metrics = _scalar_summary(summarize_round_csv(df))
    metrics.update(
        {
            "ablation_family": family,
            "variant_name": variant_name,
            "seed": int(seed),
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
    for (family, variant), group in df.groupby(["ablation_family", "variant_name"], sort=False):
        aggregate = group.mean(numeric_only=True).to_dict()
        aggregate.update(
            {
                "ablation_family": str(family),
                "variant_name": str(variant),
                "seed_count": int(len(group)),
                "controller_policy": str(group["controller_policy"].iloc[0]),
                "verification_scheduler": str(group["verification_scheduler"].iloc[0]),
                "uplink_allocator": str(group["uplink_allocator"].iloc[0]),
                "downlink_allocator": str(group["downlink_allocator"].iloc[0]),
            }
        )
        grouped_rows.append(aggregate)
    return pd.DataFrame(grouped_rows)


def _build_paper_table(summary_df: pd.DataFrame, specs: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        family = str(spec["family"])
        family_df = summary_df[summary_df["ablation_family"] == family]
        if family_df.empty:
            continue
        off_name = str(spec["off_name"])
        on_name = str(spec["on_name"])
        off_row = family_df[family_df["variant_name"] == off_name]
        on_row = family_df[family_df["variant_name"] == on_name]
        if off_row.empty or on_row.empty:
            continue
        off = off_row.iloc[0]
        on = on_row.iloc[0]
        rows.append(
            {
                "ablation_family": family,
                "off_variant": off_name,
                "on_variant": on_name,
                "off_goodput_tps": float(off["goodput_tps"]),
                "on_goodput_tps": float(on["goodput_tps"]),
                "delta_goodput_tps": float(on["goodput_tps"] - off["goodput_tps"]),
                "off_interactivity_jain_fairness": float(off["interactivity_jain_fairness"]),
                "on_interactivity_jain_fairness": float(on["interactivity_jain_fairness"]),
                "delta_interactivity_jain_fairness": float(
                    on["interactivity_jain_fairness"] - off["interactivity_jain_fairness"]
                ),
                "off_mean_ttft_ms": float(off["mean_ttft_ms"]),
                "on_mean_ttft_ms": float(on["mean_ttft_ms"]),
                "delta_mean_ttft_ms": float(on["mean_ttft_ms"] - off["mean_ttft_ms"]),
                "off_mean_client_interactivity_s_per_token": float(
                    off["mean_client_interactivity_s_per_token"]
                ),
                "on_mean_client_interactivity_s_per_token": float(
                    on["mean_client_interactivity_s_per_token"]
                ),
                "delta_mean_client_interactivity_s_per_token": float(
                    on["mean_client_interactivity_s_per_token"]
                    - off["mean_client_interactivity_s_per_token"]
                ),
            }
        )
    return pd.DataFrame(rows)


async def _run(args: argparse.Namespace) -> None:
    base_cfg = _load_config(Path(args.config))
    seeds = (
        _parse_int_list(args.seeds)
        if args.seeds
        else [
            int(seed)
            for seed in base_cfg.get("experiment", {}).get(
                "seeds",
                [base_cfg["simulation"]["seed"]],
            )
        ]
    )
    specs = _ablation_specs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, object]] = []
    for spec in specs:
        family = str(spec["family"])
        print(f"\n=== Ablation {family} ===")
        for variant_name, overrides in [
            (str(spec["off_name"]), spec["off_overrides"]),
            (str(spec["on_name"]), spec["on_overrides"]),
        ]:
            for seed in seeds:
                metrics = await _run_single(
                    base_cfg=base_cfg,
                    family=family,
                    variant_name=variant_name,
                    overrides=overrides,
                    seed=seed,
                    output_dir=output_dir,
                    detailed_log=args.detailed_log,
                )
                per_seed_rows.append(metrics)
                print(
                    "{variant_name} seed={seed} goodput={goodput_tps:.3f} "
                    "fairness={interactivity_jain_fairness:.3f} TTFT={mean_ttft_ms:.1f}ms".format(
                        **metrics
                    )
                )

    per_seed_df = pd.DataFrame(per_seed_rows)
    summary_df = _aggregate(per_seed_rows)
    paper_table_df = _build_paper_table(summary_df, specs)

    per_seed_csv = output_dir / "ablation_per_seed.csv"
    summary_csv = output_dir / "ablation_summary.csv"
    summary_json = output_dir / "ablation_summary.json"
    paper_table_csv = output_dir / "paper_table.csv"

    per_seed_df.to_csv(per_seed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    paper_table_df.to_csv(paper_table_csv, index=False)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary_df.to_dict(orient="records"), handle, indent=2)

    print(f"\nWrote per-seed ablation CSV to {per_seed_csv}")
    print(f"Wrote ablation summary CSV to {summary_csv}")
    print(f"Wrote ablation paper table CSV to {paper_table_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired ablations on a fixed operating point and summarize them."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="results/ablations")
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
