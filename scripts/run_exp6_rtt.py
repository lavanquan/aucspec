from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_specsim.metrics import summarize_round_csv

DEFAULT_RTT_VALUES = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0]
DEFAULT_GAMMA_VALUES = [0, 1, 2, 4, 6, 8, 12, 16]
DEFAULT_FIXED_BASELINE_GAMMAS = [2, 4, 8]
DEFAULT_X_REQ_VALUES = [0.0, 4.0]
DEFAULT_BASELINES = [
    "TargetOnly",
    "Fixed-2",
    "Fixed-4",
    "Fixed-8",
    "BestFixed",
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
            "is_oracle_candidate": True,
            "is_best_fixed_candidate": False,
        },
        {
            "candidate_id": "proposed_ucb",
            "family": "Proposed-UCB",
            "policy": "adaptive_ucb",
            "fixed_gamma": 0,
            "V": float(controller_V),
            "is_oracle_candidate": False,
            "is_best_fixed_candidate": False,
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
                "is_oracle_candidate": True,
                "is_best_fixed_candidate": int(gamma) in DEFAULT_FIXED_BASELINE_GAMMAS,
            }
        )
    return candidates


def _configure_run(
    base_cfg: dict,
    *,
    candidate: dict[str, float | int | str],
    seed: int,
    rtt_ms: float,
    x_req: float,
    output_csv: Path,
    gamma_values: list[int],
) -> dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["simulation"]["seed"] = int(seed)
    cfg["dataset"]["seed"] = int(seed)
    cfg["simulation"]["output_csv"] = str(output_csv)
    cfg["simulation"]["min_interactivity_tps"] = float(x_req)
    cfg["controller"]["policy"] = str(candidate["policy"])
    cfg["controller"]["fixed_gamma"] = int(candidate["fixed_gamma"])
    cfg["controller"]["V"] = float(candidate["V"])
    cfg["controller"]["min_interactivity_tps"] = float(x_req)
    cfg["controller"]["gamma_choices"] = [int(gamma) for gamma in gamma_values]
    cfg["simulation"]["gamma_max"] = int(max(gamma_values))
    cfg["clients"]["rtt_ms_range"] = [float(rtt_ms), float(rtt_ms)]
    cfg["clients"]["uplink_mbps_range"] = [20.0, 20.0]
    cfg["clients"]["downlink_mbps_range"] = [50.0, 50.0]
    client_classes = cfg["clients"].get("client_classes", {})
    for class_cfg in client_classes.values():
        class_cfg["rtt_ms"] = [float(rtt_ms), float(rtt_ms)]
        class_cfg["uplink_mbps"] = [20.0, 20.0]
        class_cfg["downlink_mbps"] = [50.0, 50.0]
    return cfg


async def _run_candidate_seed(
    *,
    base_cfg: dict,
    candidate: dict[str, float | int | str],
    seed: int,
    rtt_ms: float,
    x_req: float,
    gamma_values: list[int],
    output_dir: Path,
    detailed_log: bool,
) -> dict[str, float | int | str]:
    from edge_specsim.simulator import EdgeSpecSimulator

    candidate_slug = _candidate_slug(candidate)
    rtt_slug = f"rtt_{str(rtt_ms).replace('.', '_')}"
    x_slug = f"xreq_{str(x_req).replace('.', '_')}"
    output_csv = output_dir / "traces" / rtt_slug / x_slug / f"{candidate_slug}__seed_{seed}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not output_csv.exists():
        run_cfg = _configure_run(
            base_cfg,
            candidate=candidate,
            seed=seed,
            rtt_ms=rtt_ms,
            x_req=x_req,
            output_csv=output_csv,
            gamma_values=gamma_values,
        )
        with tempfile.TemporaryDirectory(
            prefix=f"exp6-{rtt_slug}-{x_slug}-{candidate_slug}-seed-{seed}-"
        ) as tmp_dir:
            temp_config = Path(tmp_dir) / "config.yaml"
            _write_yaml(temp_config, run_cfg)
            simulator = EdgeSpecSimulator(str(temp_config), detailed_log=detailed_log)
            await simulator.run()

    df = pd.read_csv(output_csv)
    summary = _scalar_summary(summarize_round_csv(df))
    rounds_per_completed_prompt = (
        float(df["round_id"].count()) / max(1.0, float(df["sample_id"].nunique()))
        if {"round_id", "sample_id"}.issubset(df.columns) and not df.empty
        else float("nan")
    )
    summary.update(
        {
            "candidate_id": str(candidate["candidate_id"]),
            "family": str(candidate["family"]),
            "policy": str(candidate["policy"]),
            "fixed_gamma": int(candidate["fixed_gamma"]),
            "V": float(candidate["V"]),
            "seed": int(seed),
            "rtt_ms": float(rtt_ms),
            "x_req_tps": float(x_req),
            "round_csv": str(output_csv),
            "mean_gamma": float(df["gamma"].mean()) if "gamma" in df.columns and not df.empty else 0.0,
            "mean_round_latency_ms": float(df["round_latency_ms"].mean()) if "round_latency_ms" in df.columns and not df.empty else 0.0,
            "mean_draft_latency_ms": float(df["draft_latency_ms"].mean()) if "draft_latency_ms" in df.columns and not df.empty else 0.0,
            "mean_upload_ms": float(df["upload_ms"].mean()) if "upload_ms" in df.columns and not df.empty else 0.0,
            "mean_server_wait_ms": float(df["server_wait_ms"].mean()) if "server_wait_ms" in df.columns and not df.empty else 0.0,
            "mean_verify_ms": float(df["verify_ms"].mean()) if "verify_ms" in df.columns and not df.empty else 0.0,
            "mean_download_ms": float(df["download_ms"].mean()) if "download_ms" in df.columns and not df.empty else 0.0,
            "mean_committed_tokens_per_round": float(df["committed_token_count"].mean()) if "committed_token_count" in df.columns and not df.empty else 0.0,
            "rounds_per_completed_prompt": rounds_per_completed_prompt,
            "mean_verify_latency_ms": float(df["verify_ms"].mean()) if "verify_ms" in df.columns and not df.empty else 0.0,
            "mean_batch_size": float(df["verification_batch_size"].mean()) if "verification_batch_size" in df.columns and not df.empty else 0.0,
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
            "rtt_ms": float(df["rtt_ms"].iloc[0]),
            "x_req_tps": x_req,
            "seed_count": int(len(df)),
            "constraint_satisfied": bool((~violation_mask).all()),
            "violation_fraction": float(violation_mask.mean()),
        }
    )
    return aggregate


def _select_best_fixed(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[float, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        if not bool(row["is_best_fixed_candidate"]):
            continue
        key = (float(row["rtt_ms"]), float(row["x_req_tps"]))
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, float | int | str]] = []
    for _, candidates in sorted(grouped.items()):
        feasible = [row for row in candidates if bool(row["constraint_satisfied"])]
        pool = feasible if feasible else candidates
        best = max(pool, key=lambda row: float(row["goodput_tps"]))
        selected.append(
            {
                **best,
                "candidate_id": "best_fixed",
                "family": "BestFixed",
                "best_fixed_gamma": int(best["fixed_gamma"]),
            }
        )
    return selected


def _select_exhaustive_oracle(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[float, float], list[dict[str, float | int | str]]] = {}
    for row in rows:
        if not bool(row["is_oracle_candidate"]):
            continue
        key = (float(row["rtt_ms"]), float(row["x_req_tps"]))
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, float | int | str]] = []
    for _, candidates in sorted(grouped.items()):
        feasible = [row for row in candidates if bool(row["constraint_satisfied"])]
        pool = feasible if feasible else candidates
        best = max(pool, key=lambda row: float(row["goodput_tps"]))
        selected.append(
            {
                **best,
                "candidate_id": "exhaustive_oracle",
                "family": "ExhaustiveOracle",
                "oracle_gamma_star": int(best["fixed_gamma"]),
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


def _build_table_rows(
    selected_rows: list[dict[str, float | int | str]],
) -> pd.DataFrame:
    df = pd.DataFrame(selected_rows)
    rows: list[dict[str, float | int | str]] = []
    for (rtt_ms, x_req), group in df.groupby(["rtt_ms", "x_req_tps"], sort=True):
        by_family = {str(row["family"]): row for row in group.to_dict("records")}
        best_fixed = by_family.get("BestFixed", {})
        oracle = by_family.get("ExhaustiveOracle", {})
        proposed = by_family.get("Proposed-UCB", {})
        rows.append(
            {
                "rtt_ms": float(rtt_ms),
                "x_req_tps": float(x_req),
                "best_fixed_gamma": int(best_fixed.get("best_fixed_gamma", best_fixed.get("fixed_gamma", -1))),
                "oracle_gamma": int(oracle.get("oracle_gamma_star", -1)),
                "proposed_goodput_tps": float(proposed.get("goodput_tps", float("nan"))),
                "proposed_over_bestfixed": (
                    float(proposed.get("goodput_tps", float("nan")))
                    / max(1e-9, float(best_fixed.get("goodput_tps", float("nan"))))
                    if best_fixed
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_gamma_histogram_rows(
    rows_by_key: dict[tuple[float, float, str], list[dict[str, float | int | str]]],
    selected_rows: list[dict[str, float | int | str]],
) -> pd.DataFrame:
    histogram_rows: list[dict[str, float | int | str]] = []
    for row in selected_rows:
        family = str(row["family"])
        candidate_id = str(row["candidate_id"])
        if family == "ExhaustiveOracle":
            candidate_id = f"fixed_gamma:{int(row['oracle_gamma_star'])}"
        elif family == "BestFixed":
            candidate_id = f"fixed_gamma:{int(row['best_fixed_gamma'])}"
        key = (float(row["rtt_ms"]), float(row["x_req_tps"]), candidate_id)
        for seed_row in rows_by_key.get(key, []):
            trace = pd.read_csv(str(seed_row["round_csv"]))
            gamma_counts = trace.groupby("gamma").size().reset_index(name="round_count")
            for _, gamma_row in gamma_counts.iterrows():
                histogram_rows.append(
                    {
                        "family": family,
                        "rtt_ms": float(row["rtt_ms"]),
                        "x_req_tps": float(row["x_req_tps"]),
                        "seed": int(seed_row["seed"]),
                        "gamma": int(gamma_row["gamma"]),
                        "round_count": int(gamma_row["round_count"]),
                    }
                )
    return pd.DataFrame(histogram_rows)


def _build_decomposition_rows(
    rows_by_key: dict[tuple[float, float, str], list[dict[str, float | int | str]]],
    selected_rows: list[dict[str, float | int | str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decomposition_rows: list[dict[str, float | int | str]] = []
    client_rtt_rows: list[dict[str, float | int | str]] = []
    for row in selected_rows:
        family = str(row["family"])
        candidate_id = str(row["candidate_id"])
        if family == "ExhaustiveOracle":
            candidate_id = f"fixed_gamma:{int(row['oracle_gamma_star'])}"
        elif family == "BestFixed":
            candidate_id = f"fixed_gamma:{int(row['best_fixed_gamma'])}"
        key = (float(row["rtt_ms"]), float(row["x_req_tps"]), candidate_id)
        for seed_row in rows_by_key.get(key, []):
            trace = pd.read_csv(str(seed_row["round_csv"]))
            decomposition_rows.append(
                {
                    "family": family,
                    "rtt_ms": float(row["rtt_ms"]),
                    "x_req_tps": float(row["x_req_tps"]),
                    "seed": int(seed_row["seed"]),
                    "fixed_delay_ms": float(trace["rtt_ms"].mean()) if "rtt_ms" in trace.columns and not trace.empty else 0.0,
                    "mean_draft_latency_ms": float(trace["draft_latency_ms"].mean()) if "draft_latency_ms" in trace.columns and not trace.empty else 0.0,
                    "mean_upload_ms": float(trace["upload_ms"].mean()) if "upload_ms" in trace.columns and not trace.empty else 0.0,
                    "mean_server_wait_ms": float(trace["server_wait_ms"].mean()) if "server_wait_ms" in trace.columns and not trace.empty else 0.0,
                    "mean_verify_ms": float(trace["verify_ms"].mean()) if "verify_ms" in trace.columns and not trace.empty else 0.0,
                    "mean_download_ms": float(trace["download_ms"].mean()) if "download_ms" in trace.columns and not trace.empty else 0.0,
                    "mean_round_duration_ms": float(trace["round_latency_ms"].mean()) if "round_latency_ms" in trace.columns and not trace.empty else 0.0,
                    "rounds_per_completed_prompt": (
                        float(trace["round_id"].count()) / max(1.0, float(trace["sample_id"].nunique()))
                        if {"round_id", "sample_id"}.issubset(trace.columns) and not trace.empty
                        else float("nan")
                    ),
                    "mean_committed_tokens_per_round": float(trace["committed_token_count"].mean()) if "committed_token_count" in trace.columns and not trace.empty else 0.0,
                }
            )
            if {"client_id", "rtt_ms"}.issubset(trace.columns):
                client_rtt = trace.groupby("client_id", as_index=False)["rtt_ms"].mean()
                for _, rtt_row in client_rtt.iterrows():
                    client_rtt_rows.append(
                        {
                            "family": family,
                            "rtt_ms": float(row["rtt_ms"]),
                            "x_req_tps": float(row["x_req_tps"]),
                            "seed": int(seed_row["seed"]),
                            "client_id": int(rtt_row["client_id"]),
                            "client_rtt_ms": float(rtt_row["rtt_ms"]),
                        }
                    )
    return pd.DataFrame(decomposition_rows), pd.DataFrame(client_rtt_rows)


async def _run(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(config_path)
    gamma_values = _parse_int_list(args.gamma_values, DEFAULT_GAMMA_VALUES)
    rtt_values = _parse_float_list(args.rtt_values, DEFAULT_RTT_VALUES)
    x_req_values = _parse_float_list(args.x_req_values, DEFAULT_X_REQ_VALUES)
    seeds = _parse_int_list(args.seeds, list(base_cfg.get("experiment", {}).get("seeds", [1, 2, 3, 4, 5])))
    controller_V = float(args.V if args.V is not None else base_cfg.get("controller", {}).get("V", 50.0))

    candidates = _build_candidates(gamma_values=gamma_values, controller_V=controller_V)
    aggregate_rows: list[dict[str, float | int | str]] = []
    rows_by_key: dict[tuple[float, float, str], list[dict[str, float | int | str]]] = {}

    for x_req in x_req_values:
        for rtt_ms in rtt_values:
            for candidate in candidates:
                seed_rows: list[dict[str, float | int | str]] = []
                for seed in seeds:
                    seed_row = await _run_candidate_seed(
                        base_cfg=base_cfg,
                        candidate=candidate,
                        seed=seed,
                        rtt_ms=rtt_ms,
                        x_req=x_req,
                        gamma_values=gamma_values,
                        output_dir=output_dir,
                        detailed_log=args.detailed_log,
                    )
                    seed_rows.append(seed_row)
                aggregate = _aggregate_seed_rows(seed_rows)
                aggregate["is_oracle_candidate"] = bool(candidate["is_oracle_candidate"])
                aggregate["is_best_fixed_candidate"] = bool(candidate["is_best_fixed_candidate"])
                aggregate_rows.append(aggregate)
                rows_by_key[(float(rtt_ms), float(x_req), str(candidate["candidate_id"]))] = seed_rows

    best_fixed_rows = _select_best_fixed(aggregate_rows)
    oracle_rows = _select_exhaustive_oracle(aggregate_rows)
    selected_rows = [
        row
        for row in aggregate_rows
        if str(row["family"]) in {"TargetOnly", "Fixed-2", "Fixed-4", "Fixed-8", "Proposed-UCB"}
    ] + best_fixed_rows + oracle_rows

    all_candidates_df = pd.DataFrame(aggregate_rows).sort_values(
        ["x_req_tps", "rtt_ms", "family", "fixed_gamma"]
    )
    selected_df = pd.DataFrame(selected_rows).sort_values(
        ["x_req_tps", "rtt_ms", "family", "fixed_gamma"]
    )
    table_df = _build_table_rows(selected_rows)
    gamma_hist_df = _build_gamma_histogram_rows(rows_by_key, selected_rows)
    decomposition_df, client_rtt_df = _build_decomposition_rows(rows_by_key, selected_rows)

    all_candidates_df.to_csv(output_dir / "all_candidate_metrics.csv", index=False)
    selected_df.to_csv(output_dir / "selected_policy_metrics.csv", index=False)
    table_df.to_csv(output_dir / "table_exp6.csv", index=False)
    gamma_hist_df.to_csv(output_dir / "gamma_histograms.csv", index=False)
    decomposition_df.to_csv(output_dir / "decomposition_summary.csv", index=False)
    client_rtt_df.to_csv(output_dir / "client_rtt_logs.csv", index=False)

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
                        "rtt_ms": float(row["rtt_ms"]),
                        "value": float(row["oracle_gamma_star"]),
                    }
                )
            elif family == "Proposed-UCB":
                fig1_rows.append(
                    {
                        "series": "Proposed-UCB mean gamma",
                        "rtt_ms": float(row["rtt_ms"]),
                        "value": float(row["mean_gamma"]),
                    }
                )
        fig1_df = pd.DataFrame(fig1_rows)
        if not fig1_df.empty:
            note = _save_line_plot(
                fig1_df,
                x_key="rtt_ms",
                y_key="value",
                series_key="series",
                output_path=output_dir / f"figure_6_1__{x_slug}.png",
                title=f"Exp 6 Figure 6-1 (x_req={x_req:g})",
                x_label="RTT (ms)",
                y_label="Gamma",
            )
            if note:
                figure_notes.append(note)

        fig2_df = x_df[
            x_df["family"].isin(DEFAULT_BASELINES)
        ].copy()
        if not fig2_df.empty:
            note = _save_line_plot(
                fig2_df.rename(columns={"family": "series", "goodput_tps": "value"}),
                x_key="rtt_ms",
                y_key="value",
                series_key="series",
                output_path=output_dir / f"figure_6_2__{x_slug}.png",
                title=f"Exp 6 Figure 6-2 (x_req={x_req:g})",
                x_label="RTT (ms)",
                y_label="System goodput Y (tokens/s)",
            )
            if note:
                figure_notes.append(note)

        target_only = (
            x_df[x_df["family"] == "TargetOnly"][["rtt_ms", "goodput_tps"]]
            .rename(columns={"goodput_tps": "target_only_goodput_tps"})
        )
        fig3_df = x_df[x_df["family"].isin(["Fixed-2", "Fixed-4", "Fixed-8", "BestFixed", "Proposed-UCB", "ExhaustiveOracle"])].merge(
            target_only,
            on="rtt_ms",
            how="left",
        )
        if not fig3_df.empty:
            fig3_df["value"] = (
                fig3_df["goodput_tps"].astype(float)
                / fig3_df["target_only_goodput_tps"].astype(float).clip(lower=1e-9)
            )
            note = _save_line_plot(
                fig3_df.rename(columns={"family": "series"}),
                x_key="rtt_ms",
                y_key="value",
                series_key="series",
                output_path=output_dir / f"figure_6_3__{x_slug}.png",
                title=f"Exp 6 Figure 6-3 (x_req={x_req:g})",
                x_label="RTT (ms)",
                y_label="Speedup over TargetOnly",
            )
            if note:
                figure_notes.append(note)

    manifest = {
        "experiment": "exp6_rtt_amortization",
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "rtt_values_ms": rtt_values,
        "gamma_values": gamma_values,
        "x_req_values": x_req_values,
        "seeds": seeds,
        "controller_V": controller_V,
        "baselines": DEFAULT_BASELINES,
        "artifacts": {
            "all_candidate_metrics_csv": str(output_dir / "all_candidate_metrics.csv"),
            "selected_policy_metrics_csv": str(output_dir / "selected_policy_metrics.csv"),
            "table_csv": str(output_dir / "table_exp6.csv"),
            "gamma_histograms_csv": str(output_dir / "gamma_histograms.csv"),
            "decomposition_summary_csv": str(output_dir / "decomposition_summary.csv"),
            "client_rtt_logs_csv": str(output_dir / "client_rtt_logs.csv"),
        },
        "notes": figure_notes,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp 6: RTT amortization.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "experiments" / "exp6_rtt.yaml"),
        help="Base YAML config for Exp 6.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "exp6_rtt"),
        help="Directory for Exp 6 outputs.",
    )
    parser.add_argument(
        "--rtt-values",
        default=None,
        help="Comma-separated RTT values in milliseconds.",
    )
    parser.add_argument(
        "--gamma-values",
        default=None,
        help="Comma-separated gamma values for exhaustive fixed sweep.",
    )
    parser.add_argument(
        "--x-req-values",
        default=None,
        help="Comma-separated x_req values. Default: 0,4",
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
        help="Override controller V for Proposed-UCB.",
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
