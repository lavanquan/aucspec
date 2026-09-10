from __future__ import annotations

import math
from typing import Any

import pandas as pd


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(pd.Series(values, dtype=float).quantile(q))


def jain_fairness(values: list[float]) -> float:
    if not values:
        return 1.0
    denominator = len(values) * sum(value * value for value in values)
    if denominator <= 0.0:
        return 1.0
    numerator = sum(values) ** 2
    return numerator / denominator


def proportional_fairness_utility(values: list[float], epsilon: float = 1e-9) -> float:
    stabilized_epsilon = max(1e-12, float(epsilon))
    if not values:
        return 0.0
    return float(sum(math.log(max(stabilized_epsilon, value + stabilized_epsilon)) for value in values))


def interactivity_seconds_per_token(
    useful_tokens: float,
    elapsed_ms: float,
) -> float:
    if useful_tokens <= 0.0:
        return math.inf
    return max(0.0, elapsed_ms) / 1000.0 / useful_tokens


def goodput_tokens_per_second(
    useful_tokens: float,
    elapsed_ms: float,
) -> float:
    if elapsed_ms <= 0.0:
        return 0.0 if useful_tokens <= 0.0 else math.inf
    return useful_tokens / (elapsed_ms / 1000.0)


def trapezoidal_auc(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return 0.0
    auc = 0.0
    for left, right, y_left, y_right in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        delta_x = float(right) - float(left)
        if delta_x < 0.0:
            raise ValueError("xs must be sorted in non-decreasing order")
        auc += 0.5 * (float(y_left) + float(y_right)) * delta_x
    return auc


def upper_concave_hull(xs: list[float], ys: list[float], tol: float = 1e-12) -> list[int]:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) <= 2:
        return list(range(len(xs)))

    hull: list[int] = []
    for index, (x_value, y_value) in enumerate(zip(xs, ys)):
        x = float(x_value)
        y = float(y_value)
        if hull and x < float(xs[hull[-1]]) - tol:
            raise ValueError("xs must be sorted in non-decreasing order")
        while len(hull) >= 2:
            left = hull[-2]
            middle = hull[-1]
            x_left = float(xs[left])
            x_middle = float(xs[middle])
            y_left = float(ys[left])
            y_middle = float(ys[middle])
            if abs(x - x_left) <= tol:
                break
            interpolated = y_left + (y - y_left) * ((x_middle - x_left) / (x - x_left))
            if y_middle <= interpolated + tol:
                hull.pop()
                continue
            break
        hull.append(index)
    return hull


def _require_columns(df: pd.DataFrame, required_columns: set[str], name: str) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for {name}: {sorted(missing)}")


def _measurement_window_bounds(df: pd.DataFrame) -> tuple[float, float | None]:
    warmup_end_ms = 0.0
    measurement_end_ms: float | None = None
    if "experiment_warmup_end_ms" in df.columns and not df["experiment_warmup_end_ms"].dropna().empty:
        warmup_end_ms = float(df["experiment_warmup_end_ms"].dropna().iloc[0])
    if (
        "experiment_measurement_end_ms" in df.columns
        and not df["experiment_measurement_end_ms"].dropna().empty
    ):
        measurement_end_ms = float(df["experiment_measurement_end_ms"].dropna().iloc[0])
    return warmup_end_ms, measurement_end_ms


def measurement_rounds(df: pd.DataFrame) -> pd.DataFrame:
    if "in_measurement_window" in df.columns:
        mask = df["in_measurement_window"].fillna(False).astype(bool)
        measured = df[mask].copy()
        return measured if not measured.empty else df.copy()
    return df.copy()


def _with_derived_sample_start_time(df: pd.DataFrame) -> pd.DataFrame:
    if "sample_start_time_ms" in df.columns:
        return df
    derived = df.copy()
    if "sample_start_ms" in derived.columns:
        derived["sample_start_time_ms"] = derived["sample_start_ms"]
        return derived
    if "draft_start_ms" in derived.columns:
        derived["sample_start_time_ms"] = (
            derived.groupby(["client_id", "sample_id"], sort=False)["draft_start_ms"]
            .transform("min")
            .astype(float)
        )
        return derived
    return derived


def compute_sample_interactivity(df: pd.DataFrame) -> pd.DataFrame:
    df = _with_derived_sample_start_time(df)
    _require_columns(
        df,
        {
            "client_id",
            "sample_id",
            "useful_tokens",
            "draft_start_ms",
            "client_receive_ms",
            "sample_start_time_ms",
        },
        "interactivity metrics",
    )

    grouped = df.groupby(["client_id", "sample_id"], sort=False)
    per_sample = (
        grouped.agg(
            useful_tokens=("useful_tokens", "sum"),
            accepted_tokens=("accepted_length", "sum"),
            sample_start_ms=("sample_start_time_ms", "min"),
            first_token_receive_ms=(
                "client_receive_ms",
                lambda values: float("nan"),
            ),
            completion_ms=("client_receive_ms", "max"),
            rounds=("round_id", "count"),
        )
        .reset_index()
    )

    first_token_receive: list[float] = []
    for (_, _), sample_df in grouped:
        useful_rows = sample_df[sample_df["useful_tokens"].astype(float) > 0.0]
        if useful_rows.empty:
            first_token_receive.append(float("nan"))
        else:
            first_token_receive.append(float(useful_rows["client_receive_ms"].min()))
    per_sample["first_token_receive_ms"] = first_token_receive

    per_sample["e2e_ms"] = (
        per_sample["completion_ms"].astype(float) - per_sample["sample_start_ms"].astype(float)
    ).clip(lower=0.0)
    per_sample["ttft_ms"] = (
        per_sample["first_token_receive_ms"].astype(float)
        - per_sample["sample_start_ms"].astype(float)
    ).clip(lower=0.0)
    per_sample.loc[per_sample["first_token_receive_ms"].isna(), "ttft_ms"] = float("nan")
    per_sample["interactivity_s_per_token"] = [
        interactivity_seconds_per_token(useful, elapsed_ms)
        for useful, elapsed_ms in zip(
            per_sample["useful_tokens"].astype(float),
            per_sample["e2e_ms"].astype(float),
        )
    ]
    per_sample["goodput_tps"] = [
        goodput_tokens_per_second(useful, elapsed_ms)
        for useful, elapsed_ms in zip(
            per_sample["useful_tokens"].astype(float),
            per_sample["e2e_ms"].astype(float),
        )
    ]

    inter_token_latency_ms: list[float] = []
    for _, row in per_sample.iterrows():
        useful_tokens = float(row["useful_tokens"])
        ttft_ms = float(row["ttft_ms"]) if pd.notna(row["ttft_ms"]) else float("nan")
        completion_ms = float(row["completion_ms"])
        if useful_tokens <= 1.0 or not math.isfinite(ttft_ms):
            inter_token_latency_ms.append(0.0)
            continue
        trailing_time_ms = max(0.0, completion_ms - (float(row["sample_start_ms"]) + ttft_ms))
        inter_token_latency_ms.append(trailing_time_ms / max(useful_tokens - 1.0, 1.0))
    per_sample["inter_token_latency_ms"] = inter_token_latency_ms
    return per_sample


def measurement_complete_samples(df: pd.DataFrame) -> pd.DataFrame:
    per_sample = compute_sample_interactivity(df)
    warmup_end_ms, measurement_end_ms = _measurement_window_bounds(df)
    mask = per_sample["sample_start_ms"].astype(float) >= warmup_end_ms
    if measurement_end_ms is not None:
        mask = mask & (per_sample["completion_ms"].astype(float) <= measurement_end_ms)
    measured = per_sample[mask].copy()
    return measured if not measured.empty else per_sample


def compute_client_interactivity(df: pd.DataFrame) -> pd.DataFrame:
    per_sample = measurement_complete_samples(df)
    per_client = (
        per_sample.groupby("client_id")
        .agg(
            useful_tokens=("useful_tokens", "sum"),
            accepted_tokens=("accepted_tokens", "sum"),
            active_e2e_ms=("e2e_ms", "sum"),
            samples=("sample_id", "count"),
            mean_sample_interactivity_s_per_token=(
                "interactivity_s_per_token",
                "mean",
            ),
            p95_sample_interactivity_s_per_token=(
                "interactivity_s_per_token",
                lambda values: percentile(list(values), 0.95),
            ),
            mean_ttft_ms=("ttft_ms", "mean"),
            p95_ttft_ms=("ttft_ms", lambda values: percentile(list(values), 0.95)),
            mean_inter_token_latency_ms=("inter_token_latency_ms", "mean"),
            p95_inter_token_latency_ms=(
                "inter_token_latency_ms",
                lambda values: percentile(list(values), 0.95),
            ),
        )
        .reset_index()
    )
    per_client["interactivity_s_per_token"] = [
        interactivity_seconds_per_token(useful, elapsed_ms)
        for useful, elapsed_ms in zip(
            per_client["useful_tokens"].astype(float),
            per_client["active_e2e_ms"].astype(float),
        )
    ]
    per_client["goodput_tps"] = [
        goodput_tokens_per_second(useful, elapsed_ms)
        for useful, elapsed_ms in zip(
            per_client["useful_tokens"].astype(float),
            per_client["active_e2e_ms"].astype(float),
        )
    ]
    per_client["accepted_token_goodput_tps"] = [
        goodput_tokens_per_second(accepted, elapsed_ms)
        for accepted, elapsed_ms in zip(
            per_client["accepted_tokens"].astype(float),
            per_client["active_e2e_ms"].astype(float),
        )
    ]
    return per_client


def compute_sustained_service_rates(
    df: pd.DataFrame,
    t_measurement_s: float,
    admitted_client_ids: list[int] | None = None,
) -> pd.DataFrame:
    """CAPACITY_AUC_NSTAR_IMPLEMENTATION.md Section 11.

    x_hat_i = (useful committed tokens delivered to client i during the
    measurement window) / T_measurement, with the SAME T_measurement
    denominator for every admitted client. This is the primary capacity
    metric and is deliberately NOT the per-sample `active_e2e_ms`-based
    `compute_client_interactivity()` rate.

    An admitted client that received no committed tokens has rate 0 and is
    kept in the frame (so `min` over the frame is a true floor). Pass
    `admitted_client_ids` (e.g. the nested-population catalog prefix) to
    include zero-service clients that never appear in `df`.
    """
    if t_measurement_s <= 0.0:
        raise ValueError("t_measurement_s must be positive")
    meas = measurement_rounds(df) if "in_measurement_window" in df.columns else df
    useful_by_client = (
        meas.groupby("client_id")["useful_tokens"].sum().astype(float).to_dict()
    )
    if admitted_client_ids is None:
        admitted_client_ids = sorted(useful_by_client.keys())
    rows = []
    for cid in admitted_client_ids:
        useful = float(useful_by_client.get(cid, 0.0))
        rows.append(
            {
                "client_id": int(cid),
                "useful_tokens_measured": useful,
                "sustained_service_rate_tps": useful / t_measurement_s,
            }
        )
    return pd.DataFrame(rows)


def summarize_sustained_service_rates(rates_df: pd.DataFrame) -> dict[str, float]:
    vals = sorted(float(v) for v in rates_df["sustained_service_rate_tps"].tolist())
    if not vals:
        return {
            "min_sustained_service_rate_tps": 0.0,
            "p10_sustained_service_rate_tps": 0.0,
            "median_sustained_service_rate_tps": 0.0,
            "mean_sustained_service_rate_tps": 0.0,
            "max_sustained_service_rate_tps": 0.0,
            "n_admitted": 0,
        }
    return {
        "min_sustained_service_rate_tps": vals[0],
        "p10_sustained_service_rate_tps": percentile(vals, 0.10),
        "median_sustained_service_rate_tps": percentile(vals, 0.50),
        "mean_sustained_service_rate_tps": sum(vals) / len(vals),
        "max_sustained_service_rate_tps": vals[-1],
        "n_admitted": len(vals),
    }


def compute_system_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    _require_columns(
        df,
        {
            "useful_tokens",
            "accepted_length",
            "gamma",
            "round_latency_ms",
            "virtual_system_time_ms",
            "verification_batch_id",
            "verification_batch_service_ms",
            "edge_service_ms",
            "worker_id",
            "client_id",
        },
        "system metrics",
    )
    if df.empty:
        raise ValueError("No rows available for system metrics")

    round_df = measurement_rounds(df)
    per_sample = measurement_complete_samples(df)
    per_client = compute_client_interactivity(df)

    useful_tokens = float(round_df["useful_tokens"].sum())
    accepted_tokens = float(round_df["accepted_length"].sum())
    proposed_tokens = float(round_df["gamma"].sum())
    warmup_end_ms, measurement_end_ms = _measurement_window_bounds(df)
    if measurement_end_ms is not None:
        makespan_ms = max(0.0, measurement_end_ms - warmup_end_ms)
    else:
        makespan_ms = max(
            0.0,
            float(round_df["virtual_system_time_ms"].max()) - warmup_end_ms,
        )
    round_latencies = round_df["round_latency_ms"].astype(float).tolist()
    interactivity_by_client = per_client["interactivity_s_per_token"].astype(float).tolist()
    service_rate_by_client = [1.0 / max(value, 1e-9) for value in interactivity_by_client]
    useful_by_client = per_client["useful_tokens"].astype(float).tolist()
    throughput_by_client = per_client["goodput_tps"].astype(float).tolist()
    sample_completion_ms = per_sample["e2e_ms"].astype(float).tolist()
    sample_ttft_ms = per_sample["ttft_ms"].dropna().astype(float).tolist()
    sample_itl_ms = per_sample["inter_token_latency_ms"].astype(float).tolist()

    batch_service_by_id = (
        round_df.groupby("verification_batch_id")["verification_batch_service_ms"].max().astype(float)
    )
    server_busy_ms = float(batch_service_by_id.sum())
    server_utilization = (
        0.0 if makespan_ms <= 0.0 else min(1.0, server_busy_ms / makespan_ms)
    )

    worker_service = (
        round_df[round_df["worker_id"].astype(int) >= 0]
        .groupby("worker_id")["edge_service_ms"]
        .sum()
        .astype(float)
    )
    worker_utilizations = (
        []
        if makespan_ms <= 0.0
        else [min(1.0, service_ms / makespan_ms) for service_ms in worker_service.tolist()]
    )

    mean_completion_ms = float(pd.Series(sample_completion_ms, dtype=float).mean())
    max_completion_ms = float(max(sample_completion_ms)) if sample_completion_ms else 0.0
    straggler_ratio = (
        0.0 if mean_completion_ms <= 0.0 else max_completion_ms / mean_completion_ms
    )

    return {
        "useful_tokens": int(useful_tokens),
        "accepted_tokens": int(accepted_tokens),
        "measurement_useful_rounds": int(len(round_df)),
        "measurement_complete_samples": int(len(per_sample)),
        "measurement_start_ms": warmup_end_ms,
        "measurement_end_ms": measurement_end_ms if measurement_end_ms is not None else float("nan"),
        "goodput_tps": goodput_tokens_per_second(useful_tokens, makespan_ms),
        "accepted_token_goodput_tps": goodput_tokens_per_second(
            accepted_tokens,
            makespan_ms,
        ),
        "completion_time_ms": makespan_ms,
        "mean_completion_time_ms": mean_completion_ms,
        "p95_completion_time_ms": percentile(sample_completion_ms, 0.95),
        "mean_ttft_ms": float(pd.Series(sample_ttft_ms, dtype=float).mean())
        if sample_ttft_ms
        else 0.0,
        "p95_ttft_ms": percentile(sample_ttft_ms, 0.95),
        "mean_inter_token_latency_ms": float(pd.Series(sample_itl_ms, dtype=float).mean())
        if sample_itl_ms
        else 0.0,
        "p95_inter_token_latency_ms": percentile(sample_itl_ms, 0.95),
        "server_utilization": server_utilization,
        "draft_gpu_utilization_mean": float(pd.Series(worker_utilizations, dtype=float).mean())
        if worker_utilizations
        else 0.0,
        "draft_gpu_utilization_max": max(worker_utilizations) if worker_utilizations else 0.0,
        "acceptance_rate": 0.0 if proposed_tokens <= 0.0 else accepted_tokens / proposed_tokens,
        "straggler_ratio": straggler_ratio,
        "mean_round_latency_ms": float(round_df["round_latency_ms"].mean()),
        "p95_round_latency_ms": percentile(round_latencies, 0.95),
        "mean_client_interactivity_s_per_token": float(
            per_client["interactivity_s_per_token"].mean()
        ),
        "min_client_interactivity_s_per_token": min(interactivity_by_client)
        if interactivity_by_client
        else 0.0,
        "max_client_interactivity_s_per_token": max(interactivity_by_client)
        if interactivity_by_client
        else 0.0,
        "mean_client_service_rate_tps": float(pd.Series(service_rate_by_client, dtype=float).mean())
        if service_rate_by_client
        else 0.0,
        "min_client_service_rate_tps": min(service_rate_by_client)
        if service_rate_by_client
        else 0.0,
        "max_client_service_rate_tps": max(service_rate_by_client)
        if service_rate_by_client
        else 0.0,
        "p95_client_interactivity_s_per_token": percentile(interactivity_by_client, 0.95),
        "mean_sample_interactivity_s_per_token": float(
            per_sample["interactivity_s_per_token"].mean()
        ),
        "p95_sample_interactivity_s_per_token": percentile(
            per_sample["interactivity_s_per_token"].astype(float).tolist(),
            0.95,
        ),
        "interactivity_jain_fairness": jain_fairness(interactivity_by_client),
        "interactivity_proportional_fairness_utility": proportional_fairness_utility(
            interactivity_by_client
        ),
        "service_rate_jain_fairness": jain_fairness(service_rate_by_client),
        "service_rate_proportional_fairness_utility": proportional_fairness_utility(
            service_rate_by_client
        ),
        "token_fairness_jain": jain_fairness(useful_by_client),
        "throughput_fairness_jain": jain_fairness(throughput_by_client),
        "interactivity_fairness_jain": jain_fairness(service_rate_by_client),
    }


def summarize_round_csv(df: pd.DataFrame) -> dict[str, Any]:
    per_sample = compute_sample_interactivity(df)
    per_client = compute_client_interactivity(df)
    summary = compute_system_metrics(df)
    summary["per_sample"] = per_sample
    summary["per_client"] = per_client
    return summary
