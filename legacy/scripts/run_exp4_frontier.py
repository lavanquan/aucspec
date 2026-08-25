from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_specsim.metrics import summarize_round_csv, trapezoidal_auc, upper_concave_hull

DEFAULT_X_REQ_VALUES = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
DEFAULT_V_VALUES = [50.0]
DEFAULT_FIXED_GAMMAS = [2, 4, 8]
DEFAULT_BATCH_SIZES = [4, 8, 16, 32, 64]

EXP4A_FAMILIES = [
    # "TargetOnly",
    # "Fixed-2",
    # "Fixed-4",
    "Fixed-8",
    # "BestFixed",
    "Proposed-Oracle",
    "Proposed-UCB",
]
EXP4B_FAMILIES = [
    "TargetOnly",
    "BestFixed",
    "Proposed-Oracle",
    "Proposed-UCB",
    "ExhaustiveOracle",
]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _parse_float_list(value: str | None, default: list[float]) -> list[float]:
    if not value:
        return list(default)
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)
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


def _candidate_id(policy: str, V_value: float, fixed_gamma: int) -> str:
    if policy == "fixed_gamma":
        return f"{policy}:gamma={fixed_gamma}"
    if policy == "target_only":
        return policy
    return f"{policy}:V={V_value:g}"


def _build_candidates(
    *,
    V_values: list[float],
    fixed_gamma_values: list[int],
) -> list[dict[str, float | int | str]]:
    candidates: list[dict[str, float | int | str]] = [
        {
            "candidate_id": "target_only",
            "policy": "target_only",
            "V": 0.0,
            "fixed_gamma": 0,
        }
    ]
    for gamma in fixed_gamma_values:
        candidates.append(
            {
                "candidate_id": _candidate_id("fixed_gamma", 0.0, gamma),
                "policy": "fixed_gamma",
                "V": 0.0,
                "fixed_gamma": int(gamma),
            }
        )
    for policy in ["oracle_gamma", "adaptive_ucb"]:
        for V_value in V_values:
            candidates.append(
                {
                    "candidate_id": _candidate_id(policy, float(V_value), 0),
                    "policy": policy,
                    "V": float(V_value),
                    "fixed_gamma": 0,
                }
            )
    return candidates


def _default_fixed_gamma_values_from_cfg(cfg: dict) -> list[int]:
    controller_cfg = cfg.get("controller", {})
    gamma_choices = [int(value) for value in controller_cfg.get("gamma_choices", DEFAULT_FIXED_GAMMAS)]
    positive = [value for value in gamma_choices if value > 0]
    return positive if positive else list(DEFAULT_FIXED_GAMMAS)


def _family_matches(
    family: str,
    row: dict[str, float | int | str],
) -> bool:
    policy = str(row["policy"])
    fixed_gamma = int(row["fixed_gamma"])
    if family == "TargetOnly":
        return policy == "target_only"
    if family == "Fixed-2":
        return policy == "fixed_gamma" and fixed_gamma == 2
    if family == "Fixed-4":
        return policy == "fixed_gamma" and fixed_gamma == 4
    if family == "Fixed-8":
        return policy == "fixed_gamma" and fixed_gamma == 8
    if family == "BestFixed":
        return policy == "fixed_gamma"
    if family == "Proposed-Oracle":
        return policy == "oracle_gamma"
    if family == "Proposed-UCB":
        return policy == "adaptive_ucb"
    if family == "ExhaustiveOracle":
        return True
    raise ValueError(f"Unknown family: {family}")


def _candidate_slug(candidate: dict[str, float | int | str]) -> str:
    return str(candidate["candidate_id"]).replace(":", "__")


def _configure_run(
    base_cfg: dict,
    *,
    candidate: dict[str, float | int | str],
    seed: int,
    batch_size: int,
    x_req: float,
    output_csv: Path,
) -> dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["simulation"]["seed"] = int(seed)
    cfg["dataset"]["seed"] = int(seed)
    cfg["simulation"]["output_csv"] = str(output_csv)
    cfg["simulation"]["min_interactivity_tps"] = float(x_req)
    cfg["controller"]["policy"] = str(candidate["policy"])
    cfg["controller"]["V"] = float(candidate["V"])
    cfg["controller"]["fixed_gamma"] = int(candidate["fixed_gamma"])
    cfg["controller"]["min_interactivity_tps"] = float(x_req)
    cfg["verification_batching"]["max_batch_size"] = int(batch_size)
    gamma_choices = [int(value) for value in cfg["controller"].get("gamma_choices", [0, 1, 2, 4, 8])]
    cfg["verification_batching"]["verify_token_budget"] = int(batch_size) * (max(gamma_choices) + 1)
    return cfg


async def _run_candidate_seed(
    *,
    base_cfg: dict,
    candidate: dict[str, float | int | str],
    seed: int,
    batch_size: int,
    x_req: float,
    output_dir: Path,
    detailed_log: bool,
) -> dict[str, float | int | str]:
    from edge_specsim.simulator import EdgeSpecSimulator

    candidate_slug = _candidate_slug(candidate)
    batch_slug = f"B{batch_size}"
    x_slug = f"xreq_{str(x_req).replace('.', '_')}"
    output_csv = output_dir / "traces" / batch_slug / x_slug / f"{candidate_slug}__seed_{seed}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not output_csv.exists():
        run_cfg = _configure_run(
            base_cfg,
            candidate=candidate,
            seed=seed,
            batch_size=batch_size,
            x_req=x_req,
            output_csv=output_csv,
        )
        with tempfile.TemporaryDirectory(
            prefix=f"exp4-{batch_slug}-{x_slug}-{candidate_slug}-seed-{seed}-"
        ) as tmp_dir:
            temp_config = Path(tmp_dir) / "config.yaml"
            _write_yaml(temp_config, run_cfg)
            simulator = EdgeSpecSimulator(str(temp_config), detailed_log=detailed_log)
            await simulator.run()

    df = pd.read_csv(output_csv)
    summary = _scalar_summary(summarize_round_csv(df))
    summary.update(
        {
            "candidate_id": str(candidate["candidate_id"]),
            "policy": str(candidate["policy"]),
            "V": float(candidate["V"]),
            "fixed_gamma": int(candidate["fixed_gamma"]),
            "seed": int(seed),
            "batch_size": int(batch_size),
            "x_req_tps": float(x_req),
            "round_csv": str(output_csv),
        }
    )
    return summary


def _aggregate_seed_rows(
    seed_rows: list[dict[str, float | int | str]],
) -> dict[str, float | int | str]:
    df = pd.DataFrame(seed_rows)
    x_req = float(df["x_req_tps"].iloc[0])
    violation_mask = df["min_client_service_rate_tps"].astype(float) < x_req
    aggregate = df.mean(numeric_only=True).to_dict()
    aggregate.update(
        {
            "candidate_id": str(df["candidate_id"].iloc[0]),
            "policy": str(df["policy"].iloc[0]),
            "V": float(df["V"].iloc[0]),
            "fixed_gamma": int(df["fixed_gamma"].iloc[0]),
            "batch_size": int(df["batch_size"].iloc[0]),
            "x_req_tps": x_req,
            "seed_count": int(len(df)),
            "constraint_satisfied": bool((~violation_mask).all()),
            "violation_fraction": float(violation_mask.mean()),
            "min_seed_service_rate_tps": float(df["min_client_service_rate_tps"].min()),
            "trace_glob": str(
                Path(df["round_csv"].iloc[0]).parent / f"{_candidate_slug(df.iloc[0].to_dict())}__seed_*.csv"
            ),
        }
    )
    return aggregate


def _build_family_points(
    candidate_rows: list[dict[str, float | int | str]],
    families: list[str],
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, float], list[dict[str, float | int | str]]] = {}
    for row in candidate_rows:
        x_req = float(row["x_req_tps"])
        for family in families:
            if _family_matches(family, row):
                grouped.setdefault((family, x_req), []).append(row)

    family_points: list[dict[str, float | int | str]] = []
    for (family, x_req), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        feasible = [row for row in rows if bool(row["constraint_satisfied"])]
        if not feasible:
            family_points.append(
                {
                    "policy_family": family,
                    "x_req_tps": float(x_req),
                    "constraint_satisfied": False,
                    "violation_fraction": 1.0,
                    "achieved_goodput_tps": float("nan"),
                    "achieved_min_interactivity_tps": float("nan"),
                    "achieved_mean_interactivity_tps": float("nan"),
                    "batch_size": int(rows[0]["batch_size"]),
                    "best_V": float("nan"),
                    "best_fixed_gamma": -1,
                    "best_candidate_id": "",
                    "best_mean_ttft_ms": float("nan"),
                }
            )
            continue
        best = max(feasible, key=lambda row: float(row["goodput_tps"]))
        family_points.append(
            {
                "policy_family": family,
                "x_req_tps": float(x_req),
                "constraint_satisfied": True,
                "violation_fraction": float(best["violation_fraction"]),
                "achieved_goodput_tps": float(best["goodput_tps"]),
                "achieved_min_interactivity_tps": float(best["min_seed_service_rate_tps"]),
                "achieved_mean_interactivity_tps": float(best["mean_client_service_rate_tps"]),
                "batch_size": int(best["batch_size"]),
                "best_V": float(best["V"]),
                "best_fixed_gamma": int(best["fixed_gamma"]),
                "best_candidate_id": str(best["candidate_id"]),
                "best_mean_ttft_ms": float(best["mean_ttft_ms"]),
            }
        )
    return family_points


def _family_table(points: list[dict[str, float | int | str]], x_values: list[float]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    x_max = max(x_values) if x_values else 0.0
    families = sorted({str(point["policy_family"]) for point in points})
    for family in families:
        family_points = [
            point for point in points if str(point["policy_family"]) == family and bool(point["constraint_satisfied"])
        ]
        family_points.sort(key=lambda point: float(point["x_req_tps"]))
        xs = [float(point["x_req_tps"]) for point in family_points]
        ys = [float(point["achieved_goodput_tps"]) for point in family_points]
        auc = trapezoidal_auc(xs, ys) if len(xs) >= 2 else 0.0
        point_by_x = {float(point["x_req_tps"]): point for point in family_points}
        rows.append(
            {
                "policy_family": family,
                "Y_at_x_2": float(point_by_x[2.0]["achieved_goodput_tps"]) if 2.0 in point_by_x else float("nan"),
                "Y_at_x_6": float(point_by_x[6.0]["achieved_goodput_tps"]) if 6.0 in point_by_x else float("nan"),
                "max_feasible_x": max(xs) if xs else float("nan"),
                "auc_trapezoidal": auc,
                "normalized_auc": (auc / x_max) if x_max > 0.0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _build_hull_rows(
    points: list[dict[str, float | int | str]],
    *,
    x_key: str,
    y_key: str,
) -> list[dict[str, float | int | str]]:
    feasible = [
        point.copy()
        for point in points
        if bool(point["constraint_satisfied"])
        and math.isfinite(float(point[x_key]))
        and math.isfinite(float(point[y_key]))
    ]
    feasible.sort(key=lambda point: (float(point[x_key]), float(point[y_key])))
    if not feasible:
        return []

    deduped: list[dict[str, float | int | str]] = []
    xs: list[float] = []
    ys: list[float] = []
    for point in feasible:
        x = float(point[x_key])
        y = float(point[y_key])
        if deduped and abs(x - xs[-1]) <= 1e-12:
            if y > ys[-1]:
                deduped[-1] = point
                xs[-1] = x
                ys[-1] = y
            continue
        deduped.append(point)
        xs.append(x)
        ys.append(y)
    hull_indices = upper_concave_hull(xs, ys)
    rows: list[dict[str, float | int | str]] = []
    for rank, index in enumerate(hull_indices):
        row = deduped[index].copy()
        row["hull_rank"] = rank
        rows.append(row)
    return rows


def _save_line_plot(
    points_df: pd.DataFrame,
    *,
    x_key: str,
    y_key: str,
    series_key: str,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    diagonal: bool = False,
) -> str | None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-codex"))
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return "matplotlib is not installed; skipped figure generation."

    fig, ax = plt.subplots(figsize=(7.5, 4.75))
    for series_value, group in points_df.groupby(series_key, sort=False):
        group = group.sort_values(x_key)
        feasible = group[group["constraint_satisfied"].fillna(False).astype(bool)]
        infeasible = group[~group["constraint_satisfied"].fillna(False).astype(bool)]
        if not feasible.empty:
            ax.plot(
                feasible[x_key].astype(float),
                feasible[y_key].astype(float),
                marker="o",
                linewidth=2.0,
                label=str(series_value),
            )
        if not infeasible.empty:
            ax.scatter(
                infeasible[x_key].astype(float),
                infeasible[y_key].astype(float),
                facecolors="none",
                edgecolors="black",
                marker="o",
            )
    if diagonal and not points_df.empty:
        xs = sorted(points_df[x_key].dropna().astype(float).tolist())
        if xs:
            ax.plot(xs, xs, linestyle="--", color="gray", linewidth=1.5, label="y=x")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return None


def _load_trace(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _build_exp4a_aux_logs(
    points: list[dict[str, float | int | str]],
    seed_run_rows: dict[tuple[int, float, str], list[dict[str, float | int | str]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gamma_rows: list[dict[str, float | int | str]] = []
    slot_rows: list[dict[str, float | int | str]] = []
    z_rows: list[dict[str, float | int | str]] = []
    for point in points:
        if not bool(point["constraint_satisfied"]):
            continue
        key = (
            int(point["batch_size"]),
            float(point["x_req_tps"]),
            str(point["best_candidate_id"]),
        )
        for seed_row in seed_run_rows.get(key, []):
            trace = _load_trace(str(seed_row["round_csv"]))
            seed = int(seed_row["seed"])
            family = str(point["policy_family"])
            x_req = float(point["x_req_tps"])
            gamma_counts = (
                trace.groupby("gamma").size().reset_index(name="round_count")
            )
            for _, gamma_row in gamma_counts.iterrows():
                gamma_rows.append(
                    {
                        "policy_family": family,
                        "x_req_tps": x_req,
                        "seed": seed,
                        "gamma": int(gamma_row["gamma"]),
                        "round_count": int(gamma_row["round_count"]),
                    }
                )
            slot_summary = (
                trace.groupby(["client_id", "control_slot_index"], as_index=False)["useful_tokens"].sum()
            )
            for _, slot_row in slot_summary.iterrows():
                slot_rows.append(
                    {
                        "policy_family": family,
                        "x_req_tps": x_req,
                        "seed": seed,
                        "client_id": int(slot_row["client_id"]),
                        "control_slot_index": int(slot_row["control_slot_index"]),
                        "useful_tokens": float(slot_row["useful_tokens"]),
                    }
                )
            z_summary = (
                trace.sort_values(["client_id", "control_slot_index", "round_id"])
                .groupby(["client_id", "control_slot_index"], as_index=False)
                .tail(1)
            )
            for _, z_row in z_summary.iterrows():
                z_rows.append(
                    {
                        "policy_family": family,
                        "x_req_tps": x_req,
                        "seed": seed,
                        "client_id": int(z_row["client_id"]),
                        "control_slot_index": int(z_row["control_slot_index"]),
                        "z_queue": float(z_row["z_queue"]),
                        "device_queue": float(z_row["device_queue"]),
                        "server_queue": float(z_row["server_queue"]),
                    }
                )
    return pd.DataFrame(gamma_rows), pd.DataFrame(slot_rows), pd.DataFrame(z_rows)


async def _run_batch_sweep(
    *,
    base_cfg: dict,
    batch_size: int,
    x_req_values: list[float],
    V_values: list[float],
    fixed_gamma_values: list[int],
    output_dir: Path,
    detailed_log: bool,
    stop_after_consecutive_infeasible: int | None,
) -> tuple[list[dict[str, float | int | str]], dict[tuple[int, float, str], list[dict[str, float | int | str]]]]:
    candidates = _build_candidates(V_values=V_values, fixed_gamma_values=fixed_gamma_values)
    candidate_rows: list[dict[str, float | int | str]] = []
    seed_run_rows: dict[tuple[int, float, str], list[dict[str, float | int | str]]] = {}
    consecutive_infeasible = 0
    for x_req in x_req_values:
        print(f"\n=== Exp4 batch={batch_size} x_req={x_req:g} ===")
        x_candidate_rows: list[dict[str, float | int | str]] = []
        for candidate in candidates:
            print(
                "candidate={candidate_id} policy={policy} V={V:g} gamma={fixed_gamma}".format(
                    **candidate
                )
            )
            seed_rows: list[dict[str, float | int | str]] = []
            for seed in [int(value) for value in base_cfg.get("experiment", {}).get("seeds", [base_cfg["simulation"]["seed"]])]:
                metrics = await _run_candidate_seed(
                    base_cfg=base_cfg,
                    candidate=candidate,
                    seed=seed,
                    batch_size=batch_size,
                    x_req=float(x_req),
                    output_dir=output_dir,
                    detailed_log=detailed_log,
                )
                seed_rows.append(metrics)
            aggregate = _aggregate_seed_rows(seed_rows)
            x_candidate_rows.append(aggregate)
            candidate_rows.append(aggregate)
            seed_run_rows[(batch_size, float(x_req), str(candidate["candidate_id"]))] = seed_rows
            print(
                "  goodput={goodput_tps:.3f} x_min={min_seed_service_rate_tps:.3f} "
                "feasible={constraint_satisfied} violation_fraction={violation_fraction:.3f}".format(
                    **aggregate
                )
            )
        if any(bool(row["constraint_satisfied"]) for row in x_candidate_rows):
            consecutive_infeasible = 0
        else:
            consecutive_infeasible += 1
            if (
                stop_after_consecutive_infeasible is not None
                and stop_after_consecutive_infeasible > 0
                and consecutive_infeasible >= int(stop_after_consecutive_infeasible)
            ):
                print(
                    "Stopping Exp4 sweep early after "
                    f"{consecutive_infeasible} consecutive infeasible x_req values."
                )
                break
    return candidate_rows, seed_run_rows


async def _run_exp4a(
    *,
    base_cfg: dict,
    x_req_values: list[float],
    V_values: list[float],
    fixed_gamma_values: list[int],
    batch_size: int,
    output_dir: Path,
    detailed_log: bool,
) -> None:
    candidate_rows, seed_run_rows = await _run_batch_sweep(
        base_cfg=base_cfg,
        batch_size=batch_size,
        x_req_values=x_req_values,
        V_values=V_values,
        fixed_gamma_values=fixed_gamma_values,
        output_dir=output_dir,
        detailed_log=detailed_log,
        stop_after_consecutive_infeasible=3,
    )
    family_points = _build_family_points(candidate_rows, EXP4A_FAMILIES)
    points_df = pd.DataFrame(family_points)
    table_df = _family_table(family_points, x_req_values)
    gamma_df, slot_df, z_df = _build_exp4a_aux_logs(family_points, seed_run_rows)

    points_csv = output_dir / "exp4a_policy_frontier.csv"
    table_csv = output_dir / "exp4a_table.csv"
    gamma_csv = output_dir / "exp4a_gamma_distribution.csv"
    slot_csv = output_dir / "exp4a_per_client_useful_tokens_by_slot.csv"
    z_csv = output_dir / "exp4a_z_trajectory.csv"
    fig_goodput = output_dir / "figure_4a_1_goodput_vs_xreq.png"
    fig_interactivity = output_dir / "figure_4a_2_xmin_vs_xreq.png"

    pd.DataFrame(candidate_rows).to_csv(output_dir / "exp4a_candidate_summary.csv", index=False)
    points_df.to_csv(points_csv, index=False)
    table_df.to_csv(table_csv, index=False)
    gamma_df.to_csv(gamma_csv, index=False)
    slot_df.to_csv(slot_csv, index=False)
    z_df.to_csv(z_csv, index=False)
    goodput_note = _save_line_plot(
        points_df,
        x_key="x_req_tps",
        y_key="achieved_goodput_tps",
        series_key="policy_family",
        output_path=fig_goodput,
        title="Exp 4A: Goodput vs Required Minimum Interactivity",
        x_label="Required Minimum Interactivity x_req (token/s/client)",
        y_label="Achieved System Goodput Y (token/s)",
    )
    interactivity_note = _save_line_plot(
        points_df,
        x_key="x_req_tps",
        y_key="achieved_min_interactivity_tps",
        series_key="policy_family",
        output_path=fig_interactivity,
        title="Exp 4A: Achieved Minimum Interactivity vs Requirement",
        x_label="Required Minimum Interactivity x_req (token/s/client)",
        y_label="Achieved Minimum Interactivity x_min (token/s/client)",
        diagonal=True,
    )
    print(f"\nWrote Exp4A frontier CSV to {points_csv}")
    print(f"Wrote Exp4A summary table to {table_csv}")
    print(f"Wrote Exp4A Figure 4A-1 to {fig_goodput}")
    print(f"Wrote Exp4A Figure 4A-2 to {fig_interactivity}")
    if goodput_note:
        print(f"Figure 4A-1 note: {goodput_note}")
    if interactivity_note:
        print(f"Figure 4A-2 note: {interactivity_note}")


def _build_exp4b_envelope(
    family_points: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, float], list[dict[str, float | int | str]]] = {}
    for point in family_points:
        grouped.setdefault((str(point["policy_family"]), float(point["x_req_tps"])), []).append(point)
    envelope_rows: list[dict[str, float | int | str]] = []
    for (family, x_req), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        feasible = [row for row in rows if bool(row["constraint_satisfied"])]
        if not feasible:
            envelope_rows.append(
                {
                    "policy_family": family,
                    "x_req_tps": x_req,
                    "constraint_satisfied": False,
                    "batch_size_star": -1,
                    "achieved_min_interactivity_tps": float("nan"),
                    "achieved_goodput_tps": float("nan"),
                    "best_V": float("nan"),
                    "best_fixed_gamma": -1,
                    "best_candidate_id": "",
                }
            )
            continue
        best = max(feasible, key=lambda row: float(row["achieved_goodput_tps"]))
        envelope_rows.append(
            {
                **best,
                "batch_size_star": int(best["batch_size"]),
            }
        )
    return envelope_rows


async def _run_exp4b(
    *,
    base_cfg: dict,
    x_req_values: list[float],
    V_values: list[float],
    fixed_gamma_values: list[int],
    batch_sizes: list[int],
    output_dir: Path,
    detailed_log: bool,
) -> None:
    all_family_points: list[dict[str, float | int | str]] = []
    all_candidate_rows: list[dict[str, float | int | str]] = []
    for batch_size in batch_sizes:
        batch_output_dir = output_dir / f"B{batch_size}"
        candidate_rows, _ = await _run_batch_sweep(
            base_cfg=base_cfg,
            batch_size=batch_size,
            x_req_values=x_req_values,
            V_values=V_values,
            fixed_gamma_values=fixed_gamma_values,
            output_dir=batch_output_dir,
            detailed_log=detailed_log,
            stop_after_consecutive_infeasible=None,
        )
        family_points = _build_family_points(candidate_rows, EXP4B_FAMILIES)
        all_candidate_rows.extend(candidate_rows)
        all_family_points.extend(family_points)

    candidate_operating_points_df = pd.DataFrame(all_family_points)
    envelope_rows = _build_exp4b_envelope(all_family_points)
    envelope_df = pd.DataFrame(envelope_rows)

    hull_rows: list[dict[str, float | int | str]] = []
    table_rows: list[dict[str, float | int | str]] = []
    for family in EXP4B_FAMILIES:
        family_envelope = [
            row for row in envelope_rows if str(row["policy_family"]) == family
        ]
        family_hull = _build_hull_rows(
            family_envelope,
            x_key="achieved_min_interactivity_tps",
            y_key="achieved_goodput_tps",
        )
        hull_rows.extend(family_hull)
        xs = [float(row["achieved_min_interactivity_tps"]) for row in family_hull]
        ys = [float(row["achieved_goodput_tps"]) for row in family_hull]
        auc = trapezoidal_auc(xs, ys) if len(xs) >= 2 else 0.0
        feasible = [row for row in family_envelope if bool(row["constraint_satisfied"])]
        table_rows.append(
            {
                "policy_family": family,
                "max_feasible_x_req": max((float(row["x_req_tps"]) for row in feasible), default=float("nan")),
                "auc_trapezoidal": auc,
                "normalized_auc": (auc / max(x_req_values)) if x_req_values else float("nan"),
            }
        )

    hull_df = pd.DataFrame(hull_rows)
    table_df = pd.DataFrame(table_rows)
    candidate_df = pd.DataFrame(all_candidate_rows)

    candidate_points_csv = output_dir / "exp4b_candidate_operating_points.csv"
    envelope_csv = output_dir / "exp4b_raw_envelope.csv"
    hull_csv = output_dir / "exp4b_concave_hull_points.csv"
    auc_points_csv = output_dir / "exp4b_auc_integration_points.csv"
    table_csv = output_dir / "exp4b_table.csv"
    figure_png = output_dir / "figure_4b_full_frontier_envelope.png"

    candidate_df.to_csv(output_dir / "exp4b_candidate_summary.csv", index=False)
    candidate_operating_points_df.to_csv(candidate_points_csv, index=False)
    envelope_df.to_csv(envelope_csv, index=False)
    hull_df.to_csv(hull_csv, index=False)
    hull_df.to_csv(auc_points_csv, index=False)
    table_df.to_csv(table_csv, index=False)
    _save_line_plot(
        envelope_df,
        x_key="achieved_min_interactivity_tps",
        y_key="achieved_goodput_tps",
        series_key="policy_family",
        output_path=figure_png,
        title="Exp 4B: Full Goodput-Interactivity Envelope",
        x_label="Achieved Minimum Interactivity x_min (token/s/client)",
        y_label="Maximum System Goodput Y (token/s)",
    )
    print(f"\nWrote Exp4B envelope CSV to {envelope_csv}")
    print(f"Wrote Exp4B hull CSV to {hull_csv}")


async def _run(args: argparse.Namespace) -> None:
    base_cfg = _load_yaml(Path(args.base_config))
    exp_cfg = _load_yaml(Path(args.exp_config))
    base_cfg = _deep_merge(base_cfg, exp_cfg)
    if args.seeds:
        base_cfg["experiment"]["seeds"] = _parse_int_list(args.seeds, [])
    output_dir = Path(args.output_dir)
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_req_values = _parse_float_list(args.x_req_values, DEFAULT_X_REQ_VALUES)
    V_values = _parse_float_list(args.V_values, DEFAULT_V_VALUES)
    fixed_gamma_values = (
        _parse_int_list(args.fixed_gamma_values, DEFAULT_FIXED_GAMMAS)
        if args.fixed_gamma_values
        else _default_fixed_gamma_values_from_cfg(base_cfg)
    )
    batch_sizes = _parse_int_list(args.batch_sizes, DEFAULT_BATCH_SIZES)

    manifest = {
        "base_config": str(Path(args.base_config)),
        "exp_config": str(Path(args.exp_config)),
        "x_req_values": x_req_values,
        "V_values": V_values,
        "fixed_gamma_values": fixed_gamma_values,
        "batch_sizes": batch_sizes,
        "B_semantics": "verification_batching.max_batch_size",
        "exhaustive_oracle_semantics": (
            "best feasible operating point over sampled TargetOnly, fixed-gamma, "
            "oracle_gamma, and adaptive_ucb candidates"
        ),
        "caveats": [
            "BestFixed is currently selected from the same sweep candidate set; a separate "
            "calibration/training prompt split is still needed to match the experimental plan exactly.",
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    if args.mode in {"4a", "all"}:
        await _run_exp4a(
            base_cfg=base_cfg,
            x_req_values=x_req_values,
            V_values=V_values,
            fixed_gamma_values=fixed_gamma_values,
            batch_size=int(args.exp4a_batch_size),
            output_dir=output_dir / "exp4a",
            detailed_log=args.detailed_log,
        )
    if args.mode in {"4b", "all"}:
        await _run_exp4b(
            base_cfg=base_cfg,
            x_req_values=x_req_values,
            V_values=V_values,
            fixed_gamma_values=fixed_gamma_values,
            batch_sizes=batch_sizes,
            output_dir=output_dir / "exp4b",
            detailed_log=args.detailed_log,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Exp 4 goodput-interactivity frontier studies: "
            "Exp 4A fixed-batch SLO sweep and Exp 4B full batch-size envelope."
        )
    )
    parser.add_argument("--base-config", default="configs/default.yaml")
    parser.add_argument("--exp-config", default="configs/experiments/exp4_frontier.yaml")
    parser.add_argument("--mode", choices=["4a", "4b", "all"], default="4a")
    parser.add_argument("--output-dir", default="results/exp4_frontier")
    parser.add_argument("--x-req-values")
    parser.add_argument("--V-values")
    parser.add_argument("--fixed-gamma-values")
    parser.add_argument("--batch-sizes")
    parser.add_argument("--exp4a-batch-size", type=int, default=16)
    parser.add_argument("--seeds", help="Comma-separated seed override.")
    parser.add_argument("--detailed-log", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
