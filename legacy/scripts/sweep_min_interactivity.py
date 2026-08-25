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

from edge_specsim.metrics import summarize_round_csv, trapezoidal_auc
from edge_specsim.metrics import upper_concave_hull

DEFAULT_INTERACTIVITY_TARGETS = [0.0, 1.0, 2.0, 4.0, 6.0]
DEFAULT_POLICIES = [
    "target_only",
    "fixed_gamma",
    "adaptive_queue",
    "adaptive_ucb",
]
DEFAULT_V_VALUES = [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]


def _save_frontier_plot(
    raw_frontier_df: pd.DataFrame,
    hull_frontier_df: pd.DataFrame,
    output_path: Path,
) -> str | None:
    feasible_df = raw_frontier_df[raw_frontier_df["feasible"].fillna(False).astype(bool)].copy()
    if feasible_df.empty or hull_frontier_df.empty:
        return "No feasible frontier points were available to plot."
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return "matplotlib is not installed; skipped frontier.png generation."

    feasible_df = feasible_df.sort_values("interactivity_target_tps")
    hull_frontier_df = hull_frontier_df.sort_values("interactivity_target_tps")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(
        feasible_df["service_rate_target_tps"].astype(float),
        feasible_df["estimated_best_goodput_tps"].astype(float),
        label="Raw sampled operating points",
        marker="o",
        alpha=0.7,
    )
    ax.plot(
        hull_frontier_df["service_rate_target_tps"].astype(float),
        hull_frontier_df["estimated_best_goodput_tps"].astype(float),
        label="Estimated upper concave hull",
        marker="o",
        linewidth=2.0,
    )
    ax.set_xlabel("Minimum Service-Rate Constraint x (token/s)")
    ax.set_ylabel("Estimated Best Goodput over Sampled Candidates (token/s)")
    ax.set_title("Estimated Throughput-Interactivity Frontier")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return None


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_config(config_path: Path, cfg: dict) -> None:
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _scalar_summary(summary: dict[str, object]) -> dict[str, float | int | str]:
    return {
        key: value
        for key, value in summary.items()
        if not isinstance(value, pd.DataFrame)
    }


def _candidate_descriptor(policy: str, V_value: float, fixed_gamma: int | None) -> str:
    if policy == "fixed_gamma":
        return f"{policy}:gamma={fixed_gamma}:V={V_value:g}"
    return f"{policy}:V={V_value:g}"


def _build_candidates(
    cfg: dict,
    policies: list[str],
    V_values: list[float],
    fixed_gamma_values: list[int] | None,
) -> list[dict[str, float | int | str]]:
    controller_cfg = cfg.get("controller", {})
    gamma_choices = [int(value) for value in controller_cfg.get("gamma_choices", [0, 1, 2, 4])]
    positive_gamma_choices = [value for value in gamma_choices if value > 0]
    if fixed_gamma_values is None:
        fixed_gamma_values = positive_gamma_choices

    candidates: list[dict[str, float | int | str]] = []
    for policy in policies:
        normalized_policy = str(policy).lower()
        if normalized_policy == "fixed_gamma":
            for gamma in fixed_gamma_values:
                for V_value in V_values:
                    candidates.append(
                        {
                            "policy": normalized_policy,
                            "V": float(V_value),
                            "fixed_gamma": int(gamma),
                            "candidate_id": _candidate_descriptor(
                                normalized_policy,
                                float(V_value),
                                int(gamma),
                            ),
                        }
                    )
        else:
            for V_value in V_values:
                candidates.append(
                    {
                        "policy": normalized_policy,
                        "V": float(V_value),
                        "fixed_gamma": int(controller_cfg.get("fixed_gamma", 0)),
                        "candidate_id": _candidate_descriptor(
                            normalized_policy,
                            float(V_value),
                            None,
                        ),
                    }
                )
    return candidates


def _apply_candidate_and_seed(
    base_cfg: dict,
    candidate: dict[str, float | int | str],
    seed: int,
    output_csv: Path,
    interactivity_target_tps: float,
) -> dict:
    cfg = dict(base_cfg)
    cfg["simulation"] = dict(cfg.get("simulation", {}))
    cfg["dataset"] = dict(cfg.get("dataset", {}))
    cfg["controller"] = dict(cfg.get("controller", {}))

    cfg["simulation"]["seed"] = int(seed)
    cfg["dataset"]["seed"] = int(seed)
    cfg["simulation"]["output_csv"] = str(output_csv)
    cfg["simulation"]["min_interactivity_tps"] = float(interactivity_target_tps)
    cfg["controller"]["policy"] = str(candidate["policy"])
    cfg["controller"]["V"] = float(candidate["V"])
    cfg["controller"]["min_interactivity_tps"] = float(interactivity_target_tps)
    if str(candidate["policy"]) == "fixed_gamma":
        cfg["controller"]["fixed_gamma"] = int(candidate["fixed_gamma"])
    return cfg


async def _run_candidate_seed(
    config_path: Path,
    output_dir: Path,
    candidate: dict[str, float | int | str],
    seed: int,
    interactivity_target_tps: float,
    detailed_log: bool,
) -> dict[str, float | int | str]:
    from edge_specsim.simulator import EdgeSpecSimulator

    base_cfg = _load_config(config_path)
    candidate_slug = str(candidate["candidate_id"]).replace(":", "__")
    output_csv = output_dir / f"{candidate_slug}__seed_{seed}.csv"
    run_cfg = _apply_candidate_and_seed(
        base_cfg,
        candidate,
        seed=seed,
        output_csv=output_csv,
        interactivity_target_tps=interactivity_target_tps,
    )
    with tempfile.TemporaryDirectory(prefix=f"edge-specsim-{candidate_slug}-seed-{seed}-") as tmp_dir:
        temp_config_path = Path(tmp_dir) / "config.yaml"
        _write_config(temp_config_path, run_cfg)
        simulator = EdgeSpecSimulator(str(temp_config_path), detailed_log=detailed_log)
        await simulator.run()

    df = pd.read_csv(output_csv)
    summary = _scalar_summary(summarize_round_csv(df))
    summary.update(
        {
            "seed": seed,
            "policy": str(candidate["policy"]),
            "V": float(candidate["V"]),
            "fixed_gamma": int(candidate["fixed_gamma"]),
            "candidate_id": str(candidate["candidate_id"]),
        }
    )
    return summary


def _aggregate_candidate_runs(
    rows: list[dict[str, float | int | str]],
    interactivity_target_tps: float,
) -> dict[str, float | int | str]:
    df = pd.DataFrame(rows)
    violation_mask = df["min_client_service_rate_tps"].astype(float) < float(interactivity_target_tps)
    aggregate = df.mean(numeric_only=True).to_dict()
    aggregate.update(
        {
            "candidate_id": str(df["candidate_id"].iloc[0]),
            "policy": str(df["policy"].iloc[0]),
            "V": float(df["V"].iloc[0]),
            "fixed_gamma": int(df["fixed_gamma"].iloc[0]),
            "seed_count": int(len(df)),
            "interactivity_target_tps": float(interactivity_target_tps),
            "min_seed_service_rate_tps": float(df["min_client_service_rate_tps"].min()),
            "min_seed_interactivity_s_per_token": float(
                df["min_client_interactivity_s_per_token"].min()
            ),
            "constraint_satisfied": bool((~violation_mask).all()),
            "violation_fraction": float(violation_mask.mean()),
            "all_seeds_feasible": bool(
                (df["min_client_service_rate_tps"].astype(float) >= interactivity_target_tps).all()
            ),
        }
    )
    return aggregate


def _print_frontier(results: list[dict[str, float | int | str]]) -> None:
    if not results:
        return
    columns = [
        "service_rate_target_tps",
        "feasible",
        "estimated_best_goodput_tps",
        "best_policy",
        "best_V",
        "best_fixed_gamma",
        "best_observed_min_service_rate_tps",
        "best_mean_client_interactivity_s_per_token",
    ]
    widths = {
        column: max(len(column), *(len(str(result[column])) for result in results))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for result in results:
        print("  ".join(str(result[column]).ljust(widths[column]) for column in columns))


def _build_upper_concave_hull(
    frontier_rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    feasible_rows = [
        row.copy()
        for row in frontier_rows
        if bool(row["feasible"]) and not math.isnan(float(row["estimated_best_goodput_tps"]))
    ]
    feasible_rows.sort(key=lambda row: float(row["service_rate_target_tps"]))
    if not feasible_rows:
        return []

    xs: list[float] = []
    ys: list[float] = []
    deduped_rows: list[dict[str, float | int | str]] = []
    for row in feasible_rows:
        x = float(row["service_rate_target_tps"])
        y = float(row["estimated_best_goodput_tps"])
        if deduped_rows and abs(x - xs[-1]) <= 1e-12:
            if y > ys[-1]:
                xs[-1] = x
                ys[-1] = y
                deduped_rows[-1] = row
            continue
        xs.append(x)
        ys.append(y)
        deduped_rows.append(row)

    hull_indices = upper_concave_hull(xs, ys)
    hull_rows: list[dict[str, float | int | str]] = []
    for hull_rank, index in enumerate(hull_indices):
        row = deduped_rows[index].copy()
        row["hull_rank"] = hull_rank
        hull_rows.append(row)
    return hull_rows


def _compute_auc_summary(frontier_rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    hull_rows = _build_upper_concave_hull(frontier_rows)
    xs = [float(row["service_rate_target_tps"]) for row in hull_rows]
    ys = [float(row["estimated_best_goodput_tps"]) for row in hull_rows]
    return {
        "estimated_auc_trapezoidal": trapezoidal_auc(xs, ys),
        "num_frontier_points": len(frontier_rows),
        "num_hull_points": len(hull_rows),
        "x_min": xs[0] if xs else float("nan"),
        "x_max": xs[-1] if xs else float("nan"),
        "frontier_semantics": "estimated_best_over_sampled_candidates",
        "constraint_metric": "service_rate_tps",
        "objective_metric": "goodput_tps",
    }


def _build_family_frontier(
    candidate_rows: list[dict[str, float | int | str]],
    policy_family: str,
) -> list[dict[str, float | int | str]]:
    family_rows = [
        row for row in candidate_rows if str(row["policy"]) == str(policy_family)
    ]
    if not family_rows:
        return []

    grouped: dict[float, list[dict[str, float | int | str]]] = {}
    for row in family_rows:
        target = float(row["service_rate_target_tps"])
        grouped.setdefault(target, []).append(row)

    frontier_rows: list[dict[str, float | int | str]] = []
    for target, rows in sorted(grouped.items()):
        feasible_rows = [row for row in rows if bool(row["all_seeds_feasible"])]
        if not feasible_rows:
            frontier_rows.append(
                {
                    "service_rate_target_tps": target,
                    "interactivity_target_tps": target,
                    "feasible": False,
                    "estimated_best_goodput_tps": float("nan"),
                    "goodput_star_tps": float("nan"),
                    "best_policy": str(policy_family),
                    "best_V": float("nan"),
                    "best_fixed_gamma": -1,
                    "best_candidate_id": "",
                    "best_observed_min_service_rate_tps": float("nan"),
                    "best_min_client_service_rate_tps": float("nan"),
                    "best_mean_client_interactivity_s_per_token": float("nan"),
                    "best_interactivity_jain_fairness": float("nan"),
                    "best_mean_ttft_ms": float("nan"),
                    "frontier_semantics": "estimated_best_over_sampled_candidates",
                }
            )
            continue
        best = max(feasible_rows, key=lambda row: float(row["goodput_tps"]))
        frontier_rows.append(
            {
                "service_rate_target_tps": target,
                "interactivity_target_tps": target,
                "feasible": True,
                "estimated_best_goodput_tps": float(best["goodput_tps"]),
                "goodput_star_tps": float(best["goodput_tps"]),
                "best_policy": str(policy_family),
                "best_V": float(best["V"]),
                "best_fixed_gamma": int(best["fixed_gamma"]),
                "best_candidate_id": str(best["candidate_id"]),
                "best_observed_min_service_rate_tps": float(best["min_seed_service_rate_tps"]),
                "best_min_client_service_rate_tps": float(best["min_seed_service_rate_tps"]),
                "best_mean_client_interactivity_s_per_token": float(
                    best["mean_client_interactivity_s_per_token"]
                ),
                "best_interactivity_jain_fairness": float(best["interactivity_jain_fairness"]),
                "best_mean_ttft_ms": float(best["mean_ttft_ms"]),
                "frontier_semantics": "estimated_best_over_sampled_candidates",
            }
        )
    return frontier_rows


def _build_paper_table(candidate_rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    if not candidate_rows:
        return pd.DataFrame()

    policy_families = sorted({str(row["policy"]) for row in candidate_rows})
    table_rows: list[dict[str, float | int | str]] = []
    for family in policy_families:
        family_frontier = _build_family_frontier(candidate_rows, family)
        family_hull = _build_upper_concave_hull(family_frontier)
        auc_summary = _compute_auc_summary(family_frontier)
        feasible_frontier = [row for row in family_frontier if bool(row["feasible"])]
        if feasible_frontier:
            operating_point = min(
                feasible_frontier,
                key=lambda row: (
                    float(row["service_rate_target_tps"]),
                    -float(row["estimated_best_goodput_tps"]),
                ),
            )
            table_rows.append(
                {
                    "policy_family": family,
                    "estimated_headline_service_rate_target_tps": float(
                        operating_point["service_rate_target_tps"]
                    ),
                    "headline_interactivity_target_tps": float(
                        operating_point["service_rate_target_tps"]
                    ),
                    "estimated_headline_goodput_tps": float(
                        operating_point["estimated_best_goodput_tps"]
                    ),
                    "headline_goodput_tps": float(operating_point["estimated_best_goodput_tps"]),
                    "estimated_headline_interactivity_jain_fairness": float(
                        operating_point["best_interactivity_jain_fairness"]
                    ),
                    "headline_interactivity_jain_fairness": float(
                        operating_point["best_interactivity_jain_fairness"]
                    ),
                    "estimated_headline_mean_ttft_ms": float(operating_point["best_mean_ttft_ms"]),
                    "headline_mean_ttft_ms": float(operating_point["best_mean_ttft_ms"]),
                    "estimated_auc_trapezoidal": float(auc_summary["estimated_auc_trapezoidal"]),
                    "auc_trapezoidal": float(auc_summary["estimated_auc_trapezoidal"]),
                    "num_frontier_points": int(auc_summary["num_frontier_points"]),
                    "num_hull_points": int(auc_summary["num_hull_points"]),
                    "best_V": float(operating_point["best_V"]),
                    "best_fixed_gamma": int(operating_point["best_fixed_gamma"]),
                    "best_candidate_id": str(operating_point["best_candidate_id"]),
                    "frontier_semantics": "estimated_best_over_sampled_candidates",
                }
            )
        else:
            table_rows.append(
                {
                    "policy_family": family,
                    "estimated_headline_service_rate_target_tps": float("nan"),
                    "headline_interactivity_target_tps": float("nan"),
                    "estimated_headline_goodput_tps": float("nan"),
                    "headline_goodput_tps": float("nan"),
                    "estimated_headline_interactivity_jain_fairness": float("nan"),
                    "headline_interactivity_jain_fairness": float("nan"),
                    "estimated_headline_mean_ttft_ms": float("nan"),
                    "headline_mean_ttft_ms": float("nan"),
                    "estimated_auc_trapezoidal": float(auc_summary["estimated_auc_trapezoidal"]),
                    "auc_trapezoidal": float(auc_summary["estimated_auc_trapezoidal"]),
                    "num_frontier_points": int(auc_summary["num_frontier_points"]),
                    "num_hull_points": int(auc_summary["num_hull_points"]),
                    "best_V": float("nan"),
                    "best_fixed_gamma": -1,
                    "best_candidate_id": "",
                    "frontier_semantics": "estimated_best_over_sampled_candidates",
                }
            )
    return pd.DataFrame(table_rows)


async def _run(args: argparse.Namespace) -> None:
    await run_frontier_sweep(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        interactivity_targets=(
            _parse_float_list(args.interactivity_targets)
            if args.interactivity_targets
            else None
        ),
        policies=(
            [part.strip() for part in args.policies.split(",") if part.strip()]
            if args.policies
            else None
        ),
        V_values=(_parse_float_list(args.V_values) if args.V_values else None),
        fixed_gamma_values=(
            _parse_int_list(args.fixed_gamma_values)
            if args.fixed_gamma_values
            else None
        ),
        seeds=(_parse_int_list(args.seeds) if args.seeds else None),
        detailed_log=args.detailed_log,
        clean_output=args.clean_output,
    )


async def run_frontier_sweep(
    config_path: Path,
    output_dir: Path,
    interactivity_targets: list[float] | None = None,
    policies: list[str] | None = None,
    V_values: list[float] | None = None,
    fixed_gamma_values: list[int] | None = None,
    seeds: list[int] | None = None,
    detailed_log: bool = False,
    clean_output: bool = False,
    stop_after_consecutive_infeasible: int | None = None,
) -> dict[str, object]:
    config_path = Path(config_path)
    base_cfg = _load_config(config_path)
    experiment_cfg = base_cfg.get("experiment", {})
    seeds = (
        list(seeds)
        if seeds is not None
        else [int(seed) for seed in experiment_cfg.get("seeds", [base_cfg["simulation"]["seed"]])]
    )
    targets = (
        list(interactivity_targets)
        if interactivity_targets is not None
        else DEFAULT_INTERACTIVITY_TARGETS
    )
    policies = list(policies) if policies is not None else DEFAULT_POLICIES
    V_values = list(V_values) if V_values is not None else DEFAULT_V_VALUES
    candidates = _build_candidates(base_cfg, policies, V_values, fixed_gamma_values)

    output_dir = Path(output_dir)
    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, float | int | str]] = []
    frontier_rows: list[dict[str, float | int | str]] = []
    consecutive_infeasible = 0

    for target_tps in targets:
        print(f"\n=== Sweeping interactivity target {target_tps:g} token/s ===")
        candidate_summaries: list[dict[str, float | int | str]] = []
        for candidate in candidates:
            print(
                "candidate={candidate_id} policy={policy} V={V:g} fixed_gamma={fixed_gamma}".format(
                    **candidate
                )
            )
            seed_rows: list[dict[str, float | int | str]] = []
            for seed in seeds:
                metrics = await _run_candidate_seed(
                    config_path=config_path,
                    output_dir=output_dir,
                    candidate=candidate,
                    seed=seed,
                    interactivity_target_tps=float(target_tps),
                    detailed_log=detailed_log,
                )
                seed_rows.append(metrics)
            candidate_summary = _aggregate_candidate_runs(seed_rows, target_tps)
            candidate_summaries.append(candidate_summary)
            candidate_rows.append(candidate_summary)
            print(
                "  mean_goodput={goodput_tps:.3f} min_service_rate={min_seed_service_rate_tps:.3f} "
                "feasible={all_seeds_feasible}".format(**candidate_summary)
            )

        feasible_candidates = [
            summary
            for summary in candidate_summaries
            if bool(summary["all_seeds_feasible"])
        ]
        if feasible_candidates:
            consecutive_infeasible = 0
            best = max(feasible_candidates, key=lambda row: float(row["goodput_tps"]))
            frontier_rows.append(
                {
                    "service_rate_target_tps": float(target_tps),
                    "interactivity_target_tps": float(target_tps),
                    "feasible": True,
                    "estimated_best_goodput_tps": float(best["goodput_tps"]),
                    "goodput_star_tps": float(best["goodput_tps"]),
                    "best_policy": str(best["policy"]),
                    "best_V": float(best["V"]),
                    "best_fixed_gamma": int(best["fixed_gamma"]),
                    "best_candidate_id": str(best["candidate_id"]),
                    "best_observed_min_service_rate_tps": float(
                        best["min_seed_service_rate_tps"]
                    ),
                    "best_min_client_service_rate_tps": float(
                        best["min_seed_service_rate_tps"]
                    ),
                    "best_mean_client_interactivity_s_per_token": float(
                        best["mean_client_interactivity_s_per_token"]
                    ),
                    "frontier_semantics": "estimated_best_over_sampled_candidates",
                }
            )
            print(
                "best target={target:g} goodput*={goodput:.3f} policy={policy} V={V:g} gamma={gamma}".format(
                    target=target_tps,
                    goodput=float(best["goodput_tps"]),
                    policy=str(best["policy"]),
                    V=float(best["V"]),
                    gamma=int(best["fixed_gamma"]),
                )
            )
        else:
            consecutive_infeasible += 1
            frontier_rows.append(
                {
                    "service_rate_target_tps": float(target_tps),
                    "interactivity_target_tps": float(target_tps),
                    "feasible": False,
                    "estimated_best_goodput_tps": float("nan"),
                    "goodput_star_tps": float("nan"),
                    "best_policy": "",
                    "best_V": float("nan"),
                    "best_fixed_gamma": -1,
                    "best_candidate_id": "",
                    "best_observed_min_service_rate_tps": float("nan"),
                    "best_min_client_service_rate_tps": float("nan"),
                    "best_mean_client_interactivity_s_per_token": float("nan"),
                    "frontier_semantics": "estimated_best_over_sampled_candidates",
                }
            )
            print(f"no feasible candidate for target={target_tps:g}")
            if (
                stop_after_consecutive_infeasible is not None
                and stop_after_consecutive_infeasible > 0
                and consecutive_infeasible >= int(stop_after_consecutive_infeasible)
            ):
                print(
                    "Stopping frontier sweep early after "
                    f"{consecutive_infeasible} consecutive infeasible targets."
                )
                break

    candidate_summary_csv = output_dir / "interactivity_candidate_summary.csv"
    raw_frontier_csv = output_dir / "frontier_raw.csv"
    frontier_csv = output_dir / "frontier.csv"
    frontier_json = output_dir / "interactivity_frontier.json"
    frontier_png = output_dir / "frontier.png"
    auc_json = output_dir / "auc.json"
    paper_table_csv = output_dir / "paper_table.csv"
    candidate_df = pd.DataFrame(candidate_rows)
    raw_frontier_df = pd.DataFrame(frontier_rows)
    hull_frontier_rows = _build_upper_concave_hull(frontier_rows)
    frontier_df = pd.DataFrame(hull_frontier_rows)
    paper_table_df = _build_paper_table(candidate_rows)
    candidate_df.to_csv(candidate_summary_csv, index=False)
    raw_frontier_df.to_csv(raw_frontier_csv, index=False)
    frontier_df.to_csv(frontier_csv, index=False)
    paper_table_df.to_csv(paper_table_csv, index=False)
    with frontier_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "raw_frontier": frontier_rows,
                "upper_concave_hull": hull_frontier_rows,
            },
            handle,
            indent=2,
        )
    auc_summary = _compute_auc_summary(frontier_rows)
    with auc_json.open("w", encoding="utf-8") as handle:
        json.dump(auc_summary, handle, indent=2)
    plot_warning = _save_frontier_plot(raw_frontier_df, frontier_df, frontier_png)
    if plot_warning is not None:
        print(plot_warning)

    print("\nY*(x) upper concave hull")
    _print_frontier(hull_frontier_rows)
    print(f"\nEstimated AUC={auc_summary['estimated_auc_trapezoidal']:.6f}")
    print(f"Wrote candidate summary CSV to {candidate_summary_csv}")
    print(f"Wrote raw frontier CSV to {raw_frontier_csv}")
    print(f"Wrote frontier CSV to {frontier_csv}")
    print(f"Wrote paper table CSV to {paper_table_csv}")
    print(f"Wrote AUC JSON to {auc_json}")
    if plot_warning is None:
        print(f"Wrote frontier plot to {frontier_png}")
    return {
        "candidate_summary_csv": candidate_summary_csv,
        "raw_frontier_csv": raw_frontier_csv,
        "frontier_csv": frontier_csv,
        "frontier_json": frontier_json,
        "frontier_png": frontier_png,
        "auc_json": auc_json,
        "paper_table_csv": paper_table_csv,
        "candidate_df": candidate_df,
        "raw_frontier_df": raw_frontier_df,
        "frontier_df": frontier_df,
        "paper_table_df": paper_table_df,
        "auc_summary": auc_summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep minimum client service-rate constraints and estimate the best "
            "goodput frontier Y*(x) across controller configurations."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="results/interactivity_frontier")
    parser.add_argument(
        "--interactivity-targets",
        help="Comma-separated minimum service-rate targets in token/s. Default: 0,1,2,4,6.",
    )
    parser.add_argument(
        "--policies",
        help=(
            "Comma-separated controller policies to consider. Default: "
            "target_only,fixed_gamma,adaptive_queue,adaptive_ucb"
        ),
    )
    parser.add_argument(
        "--V-values",
        help="Comma-separated controller V values to consider.",
    )
    parser.add_argument(
        "--fixed-gamma-values",
        help="Comma-separated fixed gamma values when policy=fixed_gamma.",
    )
    parser.add_argument(
        "--seeds",
        help="Comma-separated seeds. Defaults to experiment.seeds from config.",
    )
    parser.add_argument("--detailed-log", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
