from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_specsim.metrics import summarize_round_csv

DEFAULT_CONCURRENCY_VALUES = [1, 2, 4, 8, 16, 32, 48, 64]
DEFAULT_GAMMA_VALUES = [0, 1, 2, 4, 6, 8, 12, 16]
DEFAULT_X_REQ_VALUES = [0.0]
DEFAULT_BASELINES = [
    "TargetOnly",
    "Fixed-4",
    "Fixed-8",
    "LoadOnly",
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


def _parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_list(value: str | None, default: list[float]) -> list[float]:
    if not value:
        return list(default)
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _scalar_summary(summary: dict[str, object]) -> dict[str, float | int | str]:
    return {
        key: value
        for key, value in summary.items()
        if not isinstance(value, pd.DataFrame)
    }


def _candidate_slug(candidate: dict[str, float | int | str]) -> str:
    return str(candidate["candidate_id"]).replace(":", "__")


def _build_candidates(
    *,
    gamma_values: list[int],
    controller_V: float,
) -> list[dict[str, float | int | str]]:
    candidates: list[dict[str, float | int | str]] = [
        {
            "candidate_id": "target_only",
            "family": "TargetOnly",
            "policy": "target_only",
            "fixed_gamma": 0,
            "V": 0.0,
            "is_exhaustive_fixed": True,
        },
        {
            "candidate_id": "load_only",
            "family": "LoadOnly",
            "policy": "load_only",
            "fixed_gamma": 0,
            "V": float(controller_V),
            "is_exhaustive_fixed": False,
        },
        {
            "candidate_id": "proposed_oracle",
            "family": "Proposed-Oracle",
            "policy": "oracle_gamma",
            "fixed_gamma": 0,
            "V": float(controller_V),
            "is_exhaustive_fixed": False,
        },
        {
            "candidate_id": "proposed_ucb",
            "family": "Proposed-UCB",
            "policy": "adaptive_ucb",
            "fixed_gamma": 0,
            "V": float(controller_V),
            "is_exhaustive_fixed": False,
        },
    ]
    for gamma in gamma_values:
        family = f"Fixed-{gamma}" if gamma > 0 else "TargetOnly"
        candidates.append(
            {
                "candidate_id": f"fixed_gamma:{gamma}",
                "family": family,
                "policy": "fixed_gamma",
                "fixed_gamma": int(gamma),
                "V": 0.0,
                "is_exhaustive_fixed": True,
            }
        )
    return candidates


def _configure_run(
    base_cfg: dict,
    *,
    candidate: dict[str, float | int | str],
    seed: int,
    concurrency: int,
    x_req: float,
    output_csv: Path,
    gamma_values: list[int],
) -> dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["simulation"]["seed"] = int(seed)
    cfg["dataset"]["seed"] = int(seed)
    cfg["simulation"]["num_clients"] = int(concurrency)
    cfg["simulation"]["output_csv"] = str(output_csv)
    cfg["simulation"]["min_interactivity_tps"] = float(x_req)
    cfg["controller"]["policy"] = str(candidate["policy"])
    cfg["controller"]["fixed_gamma"] = int(candidate["fixed_gamma"])
    cfg["controller"]["V"] = float(candidate["V"])
    cfg["controller"]["min_interactivity_tps"] = float(x_req)
    cfg["controller"]["gamma_choices"] = [int(gamma) for gamma in gamma_values]
    cfg["simulation"]["gamma_max"] = int(max(gamma_values))
    cfg["verification_batching"]["max_batch_size"] = int(concurrency)
    cfg["verification_batching"]["verify_token_budget"] = int(concurrency) * (
        int(max(gamma_values)) + 1
    )
    return cfg


async def _run_candidate_seed(
    *,
    base_cfg: dict,
    candidate: dict[str, float | int | str],
    seed: int,
    concurrency: int,
    x_req: float,
    gamma_values: list[int],
    output_dir: Path,
    detailed_log: bool,
) -> dict[str, float | int | str]:
    from edge_specsim.simulator import EdgeSpecSimulator

    candidate_slug = _candidate_slug(candidate)
    b_slug = f"B{concurrency}"
    x_slug = f"xreq_{str(x_req).replace('.', '_')}"
    output_csv = output_dir / "traces" / b_slug / x_slug / f"{candidate_slug}__seed_{seed}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not output_csv.exists():
        run_cfg = _configure_run(
            base_cfg,
            candidate=candidate,
            seed=seed,
            concurrency=concurrency,
            x_req=x_req,
            output_csv=output_csv,
            gamma_values=gamma_values,
        )
        with tempfile.TemporaryDirectory(
            prefix=f"exp5-{b_slug}-{x_slug}-{candidate_slug}-seed-{seed}-"
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
            "family": str(candidate["family"]),
            "policy": str(candidate["policy"]),
            "fixed_gamma": int(candidate["fixed_gamma"]),
            "V": float(candidate["V"]),
            "seed": int(seed),
            "concurrency": int(concurrency),
            "x_req_tps": float(x_req),
            "round_csv": str(output_csv),
            "mean_gamma": float(df["gamma"].mean()) if "gamma" in df.columns and not df.empty else 0.0,
            "mean_verify_latency_ms": (
                float(df["verify_ms"].mean()) if "verify_ms" in df.columns and not df.empty else 0.0
            ),
            "mean_batch_size": (
                float(df["verification_batch_size"].mean())
                if "verification_batch_size" in df.columns and not df.empty
                else 0.0
            ),
            "verification_tokens_per_useful_token": (
                float(df["verification_batch_token_cost"].sum()) / max(1.0, float(df["useful_tokens"].sum()))
                if {"verification_batch_token_cost", "useful_tokens"}.issubset(df.columns)
                else float("nan")
            ),
            "target_utilization": float(summary.get("server_utilization", 0.0)),
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
            "family": str(df["family"].iloc[0]),
            "policy": str(df["policy"].iloc[0]),
            "fixed_gamma": int(df["fixed_gamma"].iloc[0]),
            "V": float(df["V"].iloc[0]),
            "concurrency": int(df["concurrency"].iloc[0]),
            "x_req_tps": x_req,
            "seed_count": int(len(df)),
            "constraint_satisfied": bool((~violation_mask).all()),
            "violation_fraction": float(violation_mask.mean()),
        }
    )
    return aggregate


def _select_exhaustive_oracle(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[int, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        if not bool(row["is_exhaustive_fixed"]):
            continue
        key = (int(row["concurrency"]), float(row["x_req_tps"]))
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, float | int | str]] = []
    for key, candidates in sorted(grouped.items()):
        feasible = [row for row in candidates if bool(row["constraint_satisfied"])]
        pool = feasible if feasible else candidates
        best = max(pool, key=lambda row: float(row["goodput_tps"]))
        selected.append(
            {
                "candidate_id": "exhaustive_oracle",
                "family": "ExhaustiveOracle",
                "policy": "fixed_gamma",
                "fixed_gamma": int(best["fixed_gamma"]),
                "oracle_gamma_star": int(best["fixed_gamma"]),
                "V": 0.0,
                "concurrency": int(best["concurrency"]),
                "x_req_tps": float(best["x_req_tps"]),
                "seed_count": int(best["seed_count"]),
                "constraint_satisfied": bool(best["constraint_satisfied"]),
                "violation_fraction": float(best["violation_fraction"]),
                "goodput_tps": float(best["goodput_tps"]),
                "min_client_service_rate_tps": float(best["min_client_service_rate_tps"]),
                "mean_client_service_rate_tps": float(best["mean_client_service_rate_tps"]),
                "target_utilization": float(best.get("target_utilization", 0.0)),
                "mean_gamma": float(best["fixed_gamma"]),
                "mean_verify_latency_ms": float(best.get("mean_verify_latency_ms", 0.0)),
                "mean_batch_size": float(best.get("mean_batch_size", 0.0)),
                "verification_tokens_per_useful_token": float(
                    best.get("verification_tokens_per_useful_token", float("nan"))
                ),
            }
        )
    return selected


def _save_line_plot(
    df: pd.DataFrame,
    *,
    x_key: str,
    y_key: str,
    series_key: str,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return "matplotlib is not installed; skipped figure generation."

    fig, ax = plt.subplots(figsize=(7.5, 4.75))
    for series_name, group in df.groupby(series_key, sort=False):
        group = group.sort_values(x_key)
        ax.plot(
            group[x_key].astype(float),
            group[y_key].astype(float),
            marker="o",
            linewidth=2.0,
            label=str(series_name),
        )
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


def _build_gamma_histogram_rows(
    rows_by_key: dict[tuple[int, float, str], list[dict[str, float | int | str]]],
    selected_rows: list[dict[str, float | int | str]],
) -> pd.DataFrame:
    histogram_rows: list[dict[str, float | int | str]] = []
    for row in selected_rows:
        if str(row["family"]) not in {"Proposed-Oracle", "Proposed-UCB", "ExhaustiveOracle"}:
            continue
        key = (int(row["concurrency"]), float(row["x_req_tps"]), str(row["candidate_id"]))
        if str(row["family"]) == "ExhaustiveOracle":
            key = (int(row["concurrency"]), float(row["x_req_tps"]), f"fixed_gamma:{int(row['fixed_gamma'])}")
        for seed_row in rows_by_key.get(key, []):
            trace = pd.read_csv(str(seed_row["round_csv"]))
            gamma_counts = trace.groupby("gamma").size().reset_index(name="round_count")
            for _, gamma_row in gamma_counts.iterrows():
                histogram_rows.append(
                    {
                        "family": str(row["family"]),
                        "concurrency": int(row["concurrency"]),
                        "x_req_tps": float(row["x_req_tps"]),
                        "seed": int(seed_row["seed"]),
                        "gamma": int(gamma_row["gamma"]),
                        "round_count": int(gamma_row["round_count"]),
                    }
                )
    return pd.DataFrame(histogram_rows)


def _build_table_rows(selected_rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    df = pd.DataFrame(selected_rows)
    rows: list[dict[str, float | int | str]] = []
    for (concurrency, x_req), group in df.groupby(["concurrency", "x_req_tps"], sort=True):
        by_family = {str(row["family"]): row for row in group.to_dict("records")}
        oracle = by_family.get("ExhaustiveOracle", {})
        proposed_oracle = by_family.get("Proposed-Oracle", {})
        proposed_ucb = by_family.get("Proposed-UCB", {})
        rows.append(
            {
                "concurrency": int(concurrency),
                "x_req_tps": float(x_req),
                "oracle_gamma_star": int(oracle.get("oracle_gamma_star", -1)),
                "proposed_oracle_mean_gamma": float(proposed_oracle.get("mean_gamma", float("nan"))),
                "proposed_ucb_mean_gamma": float(proposed_ucb.get("mean_gamma", float("nan"))),
                "proposed_oracle_over_oracle_goodput": (
                    float(proposed_oracle.get("goodput_tps", float("nan")))
                    / max(1e-9, float(oracle.get("goodput_tps", float("nan"))))
                    if oracle
                    else float("nan")
                ),
                "proposed_ucb_over_oracle_goodput": (
                    float(proposed_ucb.get("goodput_tps", float("nan")))
                    / max(1e-9, float(oracle.get("goodput_tps", float("nan"))))
                    if oracle
                    else float("nan")
                ),
                "oracle_target_utilization": float(oracle.get("target_utilization", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


async def _run(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(config_path)
    gamma_values = _parse_int_list(args.gamma_values, DEFAULT_GAMMA_VALUES)
    concurrency_values = _parse_int_list(args.concurrency_values, DEFAULT_CONCURRENCY_VALUES)
    x_req_values = _parse_float_list(args.x_req_values, DEFAULT_X_REQ_VALUES)
    seeds = _parse_int_list(args.seeds, list(base_cfg.get("experiment", {}).get("seeds", [1, 2, 3, 4, 5])))
    controller_V = float(args.V if args.V is not None else base_cfg.get("controller", {}).get("V", 50.0))

    candidates = _build_candidates(gamma_values=gamma_values, controller_V=controller_V)
    aggregate_rows: list[dict[str, float | int | str]] = []
    rows_by_key: dict[tuple[int, float, str], list[dict[str, float | int | str]]] = {}

    for x_req in x_req_values:
        for concurrency in concurrency_values:
            for candidate in candidates:
                if (
                    str(candidate["family"]) not in DEFAULT_BASELINES
                    and bool(candidate["is_exhaustive_fixed"]) is False
                ):
                    continue
                seed_rows: list[dict[str, float | int | str]] = []
                for seed in seeds:
                    seed_row = await _run_candidate_seed(
                        base_cfg=base_cfg,
                        candidate=candidate,
                        seed=seed,
                        concurrency=concurrency,
                        x_req=x_req,
                        gamma_values=gamma_values,
                        output_dir=output_dir,
                        detailed_log=args.detailed_log,
                    )
                    seed_rows.append(seed_row)
                aggregate = _aggregate_seed_rows(seed_rows)
                aggregate["is_exhaustive_fixed"] = bool(candidate["is_exhaustive_fixed"])
                aggregate_rows.append(aggregate)
                rows_by_key[(concurrency, x_req, str(candidate["candidate_id"]))] = seed_rows

    oracle_rows = _select_exhaustive_oracle(aggregate_rows)
    selected_rows = [
        row
        for row in aggregate_rows
        if str(row["family"]) in DEFAULT_BASELINES
        and str(row["family"]) != "ExhaustiveOracle"
    ] + oracle_rows

    all_candidates_df = pd.DataFrame(aggregate_rows).sort_values(
        ["x_req_tps", "concurrency", "family", "fixed_gamma"]
    )
    selected_df = pd.DataFrame(selected_rows).sort_values(
        ["x_req_tps", "concurrency", "family", "fixed_gamma"]
    )
    table_df = _build_table_rows(selected_rows)
    gamma_hist_df = _build_gamma_histogram_rows(rows_by_key, selected_rows)

    all_candidates_df.to_csv(output_dir / "all_candidate_metrics.csv", index=False)
    selected_df.to_csv(output_dir / "selected_policy_metrics.csv", index=False)
    table_df.to_csv(output_dir / "table_exp5.csv", index=False)
    gamma_hist_df.to_csv(output_dir / "gamma_histograms.csv", index=False)

    figure_notes: list[str] = []
    for x_req in x_req_values:
        x_slug = f"xreq_{str(x_req).replace('.', '_')}"
        x_df = selected_df[selected_df["x_req_tps"].astype(float) == float(x_req)].copy()
        if x_df.empty:
            continue

        fig1_rows: list[dict[str, float | int | str]] = []
        for _, row in x_df.iterrows():
            family = str(row["family"])
            if family == "ExhaustiveOracle":
                fig1_rows.append(
                    {
                        "series": "Oracle gamma*",
                        "concurrency": int(row["concurrency"]),
                        "value": float(row["oracle_gamma_star"]),
                    }
                )
            elif family == "Proposed-Oracle":
                fig1_rows.append(
                    {
                        "series": "Proposed-Oracle mean gamma",
                        "concurrency": int(row["concurrency"]),
                        "value": float(row["mean_gamma"]),
                    }
                )
            elif family == "Proposed-UCB":
                fig1_rows.append(
                    {
                        "series": "Proposed-UCB mean gamma",
                        "concurrency": int(row["concurrency"]),
                        "value": float(row["mean_gamma"]),
                    }
                )
        fig1_df = pd.DataFrame(fig1_rows)
        if not fig1_df.empty:
            note = _save_line_plot(
                fig1_df,
                x_key="concurrency",
                y_key="value",
                series_key="series",
                output_path=output_dir / f"figure_5_1__{x_slug}.png",
                title=f"Exp 5 Figure 5-1 (x_req={x_req:g})",
                x_label="Concurrency B",
                y_label="Gamma",
            )
            if note:
                figure_notes.append(note)

        fig2_df = x_df[
            x_df["family"].isin(
                ["TargetOnly", "Fixed-4", "Fixed-8", "Proposed-UCB", "ExhaustiveOracle"]
            )
        ].copy()
        if not fig2_df.empty:
            note = _save_line_plot(
                fig2_df.rename(columns={"family": "series", "goodput_tps": "value"}),
                x_key="concurrency",
                y_key="value",
                series_key="series",
                output_path=output_dir / f"figure_5_2__{x_slug}.png",
                title=f"Exp 5 Figure 5-2 (x_req={x_req:g})",
                x_label="Concurrency B",
                y_label="System goodput Y (tokens/s)",
            )
            if note:
                figure_notes.append(note)

        oracle_goodput = (
            x_df[x_df["family"] == "ExhaustiveOracle"][["concurrency", "goodput_tps"]]
            .rename(columns={"goodput_tps": "oracle_goodput_tps"})
        )
        fig3_base = x_df[x_df["family"].isin(["Proposed-Oracle", "Proposed-UCB"])].merge(
            oracle_goodput,
            on="concurrency",
            how="left",
        )
        if not fig3_base.empty:
            fig3_base["value"] = (
                fig3_base["goodput_tps"].astype(float)
                / fig3_base["oracle_goodput_tps"].astype(float).clip(lower=1e-9)
            )
            note = _save_line_plot(
                fig3_base.rename(columns={"family": "series"}),
                x_key="concurrency",
                y_key="value",
                series_key="series",
                output_path=output_dir / f"figure_5_3__{x_slug}.png",
                title=f"Exp 5 Figure 5-3 (x_req={x_req:g})",
                x_label="Concurrency B",
                y_label="Proposed goodput / Oracle goodput",
            )
            if note:
                figure_notes.append(note)

    manifest = {
        "experiment": "exp5_optimal_gamma_vs_concurrency",
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "concurrency_values": concurrency_values,
        "gamma_values": gamma_values,
        "x_req_values": x_req_values,
        "seeds": seeds,
        "controller_V": controller_V,
        "baselines": DEFAULT_BASELINES,
        "artifacts": {
            "all_candidate_metrics_csv": str(output_dir / "all_candidate_metrics.csv"),
            "selected_policy_metrics_csv": str(output_dir / "selected_policy_metrics.csv"),
            "table_csv": str(output_dir / "table_exp5.csv"),
            "gamma_histograms_csv": str(output_dir / "gamma_histograms.csv"),
        },
        "notes": figure_notes,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp 5: optimal gamma versus concurrency.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "exp5_concurrency.yaml"),
        help="Base YAML config for Exp 5.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "exp5_concurrency"),
        help="Directory for Exp 5 outputs.",
    )
    parser.add_argument(
        "--concurrency-values",
        default=None,
        help="Comma-separated concurrency values. Default: 1,2,4,8,16,32,48,64",
    )
    parser.add_argument(
        "--gamma-values",
        default=None,
        help="Comma-separated gamma values for exhaustive fixed sweep.",
    )
    parser.add_argument(
        "--x-req-values",
        default=None,
        help="Comma-separated x_req values. Default: 0.0",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds. Default comes from config.experiment.seeds.",
    )
    parser.add_argument(
        "--V",
        type=float,
        default=None,
        help="Override controller V for Proposed and LoadOnly runs.",
    )
    parser.add_argument(
        "--detailed-log",
        action="store_true",
        help="Enable simulator detailed logging.",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
