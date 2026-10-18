"""Timing / accounting invariants for one simulator round CSV (no GPU, read-only).

Checks the virtual single-verifier schedule and the round-latency decomposition,
and reports measured-vs-modeled batch timing and logical-vs-processed tokens.
Writes a compact JSON next to the run's compact/ directory when --out is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPS_MS = 1e-3


def check(csv_path: Path, expected_model: str) -> dict:
    df = pd.read_csv(csv_path)
    df = df[df["verification_batch_id"].notna()].copy()
    res: dict = {"csv": csv_path.name, "n_rounds": int(len(df))}
    fails: list[str] = []

    b = df.groupby("verification_batch_id")
    # --- consistency inside a batch
    inconsistent = {
        col: int((b[col].nunique() > 1).sum())
        for col in ("verify_finish_ms", "batch_start_ms", "batch_ready_ms",
                    "verification_batch_wall_ms", "verification_batch_service_ms")
    }
    res["batches_with_inconsistent_field"] = inconsistent
    if any(inconsistent.values()):
        fails.append("requests of one batch disagree on a batch-level timestamp or duration")
    first = b.first().sort_values("batch_start_ms")
    n_batches = len(first)
    res["n_batches"] = int(n_batches)
    res["rows_per_batch_equals_batch_size"] = bool(
        (b.size().reindex(first.index) == first["verification_batch_size"]).all()
    )
    if not res["rows_per_batch_equals_batch_size"]:
        fails.append("rows per batch != verification_batch_size")

    # --- semantics and occupancy
    models = sorted(set(first["verification_service_time_model"].astype(str)))
    res["service_time_model"] = models
    if models != [expected_model]:
        fails.append(f"service_time_model {models} != {expected_model}")
    wall = first["verification_batch_wall_ms"].to_numpy()
    modeled = first["verification_batch_service_ms"].to_numpy()
    occ = (first["verify_finish_ms"] - first["batch_start_ms"]).to_numpy()
    size = first["verification_batch_size"].to_numpy()
    res["occupancy_minus_modeled_ms_max_abs"] = float(np.abs(occ - modeled).max())
    if np.abs(occ - modeled).max() > EPS_MS:
        fails.append("virtual occupancy (finish-start) != modeled service time")
    expected_modeled = wall if expected_model == "batch_wall" else wall / size
    res["modeled_minus_expected_ms_max_abs"] = float(np.abs(modeled - expected_modeled).max())
    if np.abs(modeled - expected_modeled).max() > EPS_MS:
        fails.append(f"modeled service != {'batch wall' if expected_model == 'batch_wall' else 'wall/B'}")

    meas = first["verification_batch_measured_service_ms"].to_numpy()  # event-loop measurement
    res["measured_event_loop_over_wall"] = {
        "mean": float(np.mean(meas / wall)), "min": float((meas / wall).min()), "max": float((meas / wall).max())}
    res["modeled_over_wall_mean"] = float(np.mean(modeled / wall))
    res["wall_ms"] = {"mean": float(wall.mean()), "p50": float(np.median(wall)), "p95": float(np.quantile(wall, .95)),
                      "max": float(wall.max())}
    res["batch_size"] = {"mean": float(size.mean()), "max": int(size.max()),
                         "counts": {int(k): int(v) for k, v in pd.Series(size).value_counts().sort_index().items()}}
    # linear fit wall ~ a + c*B (diagnostic only; not a marginal-cost estimate)
    if len(set(size)) > 1:
        c, a = np.polyfit(size, wall, 1)
        res["wall_vs_B_linear_fit_diagnostic"] = {"intercept_ms": float(a), "slope_ms_per_request": float(c)}

    # --- single-verifier schedule
    start = first["batch_start_ms"].to_numpy()
    fin = first["verify_finish_ms"].to_numpy()
    overlap = start[1:] - fin[:-1]
    res["min_gap_between_consecutive_batches_ms"] = float(overlap.min()) if len(overlap) else None
    res["overlapping_batch_pairs"] = int((overlap < -EPS_MS).sum())
    if res["overlapping_batch_pairs"]:
        fails.append("consecutive batches overlap on the virtual verifier")
    ids_in_start_order = first.index.to_numpy()
    res["batch_id_order_matches_start_order"] = bool((np.diff(ids_in_start_order) > 0).all())
    ready = first["batch_ready_ms"].to_numpy()
    res["start_ge_ready_violations"] = int((start < ready - EPS_MS).sum())
    if res["start_ge_ready_violations"]:
        fails.append("batch started before it was ready")
    res["server_arrival_le_ready_violations"] = int((df["server_arrival_ms"] > df["batch_ready_ms"] + EPS_MS).sum())
    if res["server_arrival_le_ready_violations"]:
        fails.append("request arrived after its batch was ready")

    # --- round latency decomposition (verifier time counted once)
    df["queue_wait"] = df["batch_start_ms"] - df["server_arrival_ms"]
    df["service"] = df["verify_finish_ms"] - df["batch_start_ms"]
    decomposed = (df["draft_latency_ms"] + df["rtt_ms"] / 2 + df["upload_ms"] + df["queue_wait"]
                  + df["service"] + df["rtt_ms"] / 2 + df["download_ms"])
    resid = (df["virtual_round_latency_ms"] - decomposed).abs()
    res["round_latency_decomposition_resid_ms_max"] = float(resid.max())
    with_amortized = decomposed + df["target_latency_ms"]
    res["resid_if_amortized_target_latency_were_added_ms_mean"] = float(
        (df["virtual_round_latency_ms"] - with_amortized).abs().mean())
    if resid.max() > EPS_MS:
        fails.append("round latency != draft+uplink+queue+service+downlink (verifier time mis-counted)")
    res["receive_minus_finish_equals_rtt_half_plus_download_max_err_ms"] = float(
        (df["client_receive_ms"] - df["verify_finish_ms"] - df["rtt_ms"] / 2 - df["download_ms"]).abs().max())

    # --- tokens
    res["logical_verifier_tokens_per_batch_mean"] = float(first["verification_batch_token_cost"].mean())
    res["processed_tokens_per_batch_mean"] = float(first["verification_batch_processed_tokens"].mean())
    res["processed_over_logical_mean"] = float(
        (first["verification_batch_processed_tokens"] / first["verification_batch_token_cost"]).mean())
    res["context_tokens_per_request_mean"] = float(
        (first["verification_batch_context_tokens_sum"] / size).mean())
    res["budget_binding_batches"] = int((first["verification_batch_token_cost"] >= first["verification_batch_token_budget"]).sum())

    # --- verifier utilisation over the whole run and queueing
    span = float(fin.max() - start.min())
    res["verifier_busy_fraction"] = float(modeled.sum() / span) if span > 0 else None
    res["mean_queue_wait_ms"] = float(df["queue_wait"].mean())

    # --- controller-visible server state (logged, price is zero in this mode)
    res["server_queue_max"] = float(df["server_queue"].max())
    res["server_queue_positive_rows"] = int((df["server_queue"] > 0).sum())
    res["verifier_price_coefficient_note"] = "theta_f stamped on requests is not logged; see report"

    res["failures"] = fails
    res["pass"] = not fails
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--model", default="batch_wall", choices=["batch_wall", "amortized_per_request"])
    p.add_argument("--out", default=None)
    a = p.parse_args()
    res = check(Path(a.csv), a.model)
    text = json.dumps(res, indent=2)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0 if res["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
