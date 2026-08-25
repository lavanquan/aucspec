from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_specsim.models import DraftResult
from edge_specsim.target_client import VLLMCandidateVerifier

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
DEFAULT_GAMMAS = [0, 1, 2, 4, 8, 16]
DEFAULT_CONTEXT_LENGTHS = [128, 512, 1024, 2048, 4096]


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _make_verifier(cfg: dict, seed: int | None) -> VLLMCandidateVerifier:
    model_cfg = cfg["models"]
    sim_cfg = cfg["simulation"]
    effective_seed = int(sim_cfg["seed"] if seed is None else seed)
    return VLLMCandidateVerifier(
        model_name=model_cfg["target"],
        gpu_memory_utilization=float(model_cfg.get("target_gpu_memory_utilization", 0.88)),
        max_model_len=int(model_cfg.get("target_max_model_len", 8192)),
        dtype=str(model_cfg.get("target_dtype", "auto")),
        enforce_eager=bool(model_cfg.get("target_enforce_eager", False)),
        enable_prefix_caching=bool(model_cfg.get("target_enable_prefix_caching", True)),
        sampling_temperature=0.0,
        seed=effective_seed,
    )


def _non_special_token_pool(verifier: VLLMCandidateVerifier) -> list[int]:
    token_ids = verifier.tokenizer.encode(
        " alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron",
        add_special_tokens=False,
    )
    special_ids = {
        token_id
        for token_id in (
            verifier.tokenizer.eos_token_id,
            verifier.tokenizer.bos_token_id,
            verifier.tokenizer.pad_token_id,
            verifier.tokenizer.unk_token_id,
        )
        if token_id is not None
    }
    pool: list[int] = []
    seen: set[int] = set()
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id in special_ids or token_id in seen:
            continue
        pool.append(token_id)
        seen.add(token_id)
    if not pool:
        raise RuntimeError("Unable to derive a pool of non-special tokenizer ids for profiling.")
    return pool


def _build_context_token_ids(
    filler_pool: list[int],
    context_length: int,
    variant: int,
) -> list[int]:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    sequence: list[int] = []
    start = variant % len(filler_pool)
    index = start
    while len(sequence) < context_length:
        sequence.append(filler_pool[index])
        index = (index + 1) % len(filler_pool)
    return sequence[:context_length]


def _build_accepted_proposal(
    verifier: VLLMCandidateVerifier,
    context_token_ids: list[int],
    gamma: int,
) -> list[int]:
    proposal: list[int] = []
    prefix = list(context_token_ids)
    for _ in range(gamma):
        next_token_id = verifier._generate_one_greedy_token(prefix)
        if next_token_id is None:
            break
        proposal.append(int(next_token_id))
        prefix.append(int(next_token_id))
    return proposal


def _make_batch_requests(
    verifier: VLLMCandidateVerifier,
    filler_pool: list[int],
    batch_size: int,
    gamma: int,
    context_length: int,
) -> list[tuple[list[int], DraftResult, int]]:
    requests: list[tuple[list[int], DraftResult, int]] = []
    for batch_index in range(batch_size):
        context_token_ids = _build_context_token_ids(
            filler_pool,
            context_length=context_length,
            variant=(context_length + 17 * batch_index + 31 * gamma),
        )
        proposal_token_ids = _build_accepted_proposal(verifier, context_token_ids, gamma)
        draft = DraftResult(
            client_id=batch_index,
            token_ids=proposal_token_ids,
            draft_logprobs=[None for _ in proposal_token_ids],
            text="",
            latency_ms=0.0,
            worker_id=-1,
            distribution_payload="delta_proposal",
        )
        requests.append((context_token_ids, draft, len(proposal_token_ids) + 1))
    return requests


def _measure_batch_latency_ms(
    verifier: VLLMCandidateVerifier,
    requests: list[tuple[list[int], DraftResult, int]],
    repeats: int,
) -> tuple[float, float]:
    latencies_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        verifier.verify_batch_sync(requests)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    mean_ms = float(sum(latencies_ms) / len(latencies_ms))
    std_ms = float(np.std(latencies_ms)) if len(latencies_ms) > 1 else 0.0
    return mean_ms, std_ms


def _feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    features = pd.DataFrame(
        {
            "bias": 1.0,
            "batch_size": df["batch_size"].astype(float),
            "gamma": df["gamma"].astype(float),
            "context_length": df["context_length"].astype(float),
            "total_candidate_tokens": df["total_candidate_tokens"].astype(float),
            "total_verifier_tokens": df["total_verifier_tokens"].astype(float),
            "batch_x_context": df["batch_size"].astype(float) * df["context_length"].astype(float),
            "candidate_x_context": (
                df["total_candidate_tokens"].astype(float)
                * df["context_length"].astype(float)
            ),
            "verifier_x_context": (
                df["total_verifier_tokens"].astype(float)
                * df["context_length"].astype(float)
            ),
            "verifier_tokens_sq": df["total_verifier_tokens"].astype(float) ** 2,
        }
    )
    return features.to_numpy(dtype=float), list(features.columns)


def _fit_profile_model(df: pd.DataFrame) -> dict[str, object]:
    design, feature_names = _feature_matrix(df)
    targets = df["wall_time_ms"].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
    predictions = design @ coefficients
    residuals = targets - predictions
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    total_sum_squares = float(np.sum(np.square(targets - np.mean(targets))))
    residual_sum_squares = float(np.sum(np.square(residuals)))
    r2 = 1.0 if total_sum_squares <= 0.0 else 1.0 - residual_sum_squares / total_sum_squares
    return {
        "feature_names": feature_names,
        "coefficients": {name: float(value) for name, value in zip(feature_names, coefficients)},
        "predictions": predictions,
        "rmse_ms": rmse,
        "r2": float(r2),
    }


def _annotate_regimes(df: pd.DataFrame, saturation_threshold: float) -> pd.DataFrame:
    profiled = df.copy()
    profiled["roofline_knee_verifier_tokens"] = 0
    profiled["roofline_knee_gamma"] = 0.0
    profiled["peak_verifier_tokens_per_s"] = 0.0
    profiled["regime"] = "memory_bound"

    for (batch_size, context_length), group in profiled.groupby(["batch_size", "context_length"]):
        ordered = group.sort_values("total_verifier_tokens")
        peak_tps = float(ordered["verifier_tokens_per_s"].max())
        threshold = saturation_threshold * peak_tps
        knee_row = ordered[ordered["verifier_tokens_per_s"] >= threshold].head(1)
        if knee_row.empty:
            knee_row = ordered.tail(1)
        knee_verifier_tokens = int(knee_row["total_verifier_tokens"].iloc[0])
        knee_gamma = max(0.0, knee_verifier_tokens / max(1, int(batch_size)) - 1.0)

        mask = (
            (profiled["batch_size"] == batch_size)
            & (profiled["context_length"] == context_length)
        )
        profiled.loc[mask, "roofline_knee_verifier_tokens"] = knee_verifier_tokens
        profiled.loc[mask, "roofline_knee_gamma"] = knee_gamma
        profiled.loc[mask, "peak_verifier_tokens_per_s"] = peak_tps

        below_knee = mask & (profiled["total_verifier_tokens"] < knee_verifier_tokens)
        at_knee = mask & (profiled["total_verifier_tokens"] == knee_verifier_tokens)
        above_knee = mask & (profiled["total_verifier_tokens"] > knee_verifier_tokens)
        profiled.loc[below_knee, "regime"] = "memory_bound"
        profiled.loc[at_knee, "regime"] = "roofline_knee"
        profiled.loc[above_knee, "regime"] = "compute_bound"

    return profiled


def _profile_target(args: argparse.Namespace) -> pd.DataFrame:
    cfg = _load_config(Path(args.config))
    verifier = _make_verifier(cfg, seed=args.seed)
    filler_pool = _non_special_token_pool(verifier)

    warmup_requests = _make_batch_requests(
        verifier,
        filler_pool,
        batch_size=min(2, max(args.batch_sizes)),
        gamma=min(1, max(args.gammas)),
        context_length=min(args.context_lengths),
    )
    for _ in range(max(0, args.warmup_runs)):
        verifier.verify_batch_sync(warmup_requests)

    rows: list[dict[str, object]] = []
    for batch_size in args.batch_sizes:
        for gamma in args.gammas:
            for context_length in args.context_lengths:
                requests = _make_batch_requests(
                    verifier,
                    filler_pool,
                    batch_size=batch_size,
                    gamma=gamma,
                    context_length=context_length,
                )
                mean_ms, std_ms = _measure_batch_latency_ms(
                    verifier,
                    requests,
                    repeats=args.repeats,
                )
                realized_candidate_tokens = sum(len(draft.token_ids) for _, draft, _ in requests)
                total_verifier_tokens = sum(len(draft.token_ids) + 1 for _, draft, _ in requests)
                rows.append(
                    {
                        "batch_size": batch_size,
                        "gamma": gamma,
                        "context_length": context_length,
                        "total_candidate_tokens": realized_candidate_tokens,
                        "total_verifier_tokens": total_verifier_tokens,
                        "wall_time_ms": mean_ms,
                        "wall_time_std_ms": std_ms,
                        "per_request_ms": mean_ms / max(1, batch_size),
                        "candidate_tokens_per_s": (
                            realized_candidate_tokens / max(mean_ms / 1000.0, 1e-9)
                        ),
                        "verifier_tokens_per_s": (
                            total_verifier_tokens / max(mean_ms / 1000.0, 1e-9)
                        ),
                        "prefix_caching_enabled": bool(
                            cfg["models"].get("target_enable_prefix_caching", True)
                        ),
                    }
                )
                print(
                    "profiled B={batch_size:2d} gamma={gamma:2d} L={context_length:4d} "
                    "Gamma={total_candidate_tokens:3d} T={wall_time_ms:8.2f} ms".format(
                        batch_size=batch_size,
                        gamma=gamma,
                        context_length=context_length,
                        total_candidate_tokens=realized_candidate_tokens,
                        wall_time_ms=mean_ms,
                    )
                )

    profiled = pd.DataFrame(rows)
    fit = _fit_profile_model(profiled)
    profiled["predicted_wall_time_ms"] = fit["predictions"]
    profiled["fit_error_ms"] = profiled["wall_time_ms"] - profiled["predicted_wall_time_ms"]
    profiled["fit_abs_pct_error"] = np.where(
        profiled["wall_time_ms"].to_numpy(dtype=float) > 0.0,
        np.abs(profiled["fit_error_ms"]) / profiled["wall_time_ms"] * 100.0,
        0.0,
    )
    profiled = _annotate_regimes(profiled, saturation_threshold=args.knee_threshold)

    summary = {
        "model": cfg["models"]["target"],
        "seed": int(cfg["simulation"]["seed"] if args.seed is None else args.seed),
        "batch_sizes": args.batch_sizes,
        "gammas": args.gammas,
        "context_lengths": args.context_lengths,
        "repeats": args.repeats,
        "warmup_runs": args.warmup_runs,
        "fit_rmse_ms": fit["rmse_ms"],
        "fit_r2": fit["r2"],
        "fit_coefficients": fit["coefficients"],
        "theta_0_ms": float(fit["coefficients"].get("bias", 0.0)),
        "knee_threshold": args.knee_threshold,
        "regime_counts": profiled["regime"].value_counts().to_dict(),
        "roofline_knees": [
            {
                "batch_size": int(batch_size),
                "context_length": int(context_length),
                "knee_verifier_tokens": int(group["roofline_knee_verifier_tokens"].iloc[0]),
                "knee_gamma": float(group["roofline_knee_gamma"].iloc[0]),
                "peak_verifier_tokens_per_s": float(group["peak_verifier_tokens_per_s"].iloc[0]),
                "theta_0_ms": float(fit["coefficients"].get("bias", 0.0)),
                "theta_f_ms_per_token": float(
                    1000.0 / max(1e-9, float(group["peak_verifier_tokens_per_s"].iloc[0]))
                ),
                "theta_m_ms": float(
                    int(group["roofline_knee_verifier_tokens"].iloc[0])
                    * 1000.0
                    / max(1e-9, float(group["peak_verifier_tokens_per_s"].iloc[0]))
                ),
            }
            for (batch_size, context_length), group in profiled.groupby(
                ["batch_size", "context_length"]
            )
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiled.to_csv(output_path, index=False)

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nWrote target profile to {output_path}")
    print(f"Wrote target profile summary to {summary_path}")
    print(
        "Fit: rmse={rmse:.2f} ms r2={r2:.4f}".format(
            rmse=summary["fit_rmse_ms"],
            r2=summary["fit_r2"],
        )
    )
    return profiled


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile target verification latency across batch size, gamma, and context length."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="results/target_profile.csv")
    parser.add_argument("--summary-output", default="results/target_profile_summary.json")
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--gammas", type=_parse_int_list, default=DEFAULT_GAMMAS)
    parser.add_argument("--context-lengths", type=_parse_int_list, default=DEFAULT_CONTEXT_LENGTHS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument(
        "--knee-threshold",
        type=float,
        default=0.90,
        help="Fraction of peak verifier throughput used to mark the roofline knee.",
    )
    parser.add_argument("--seed", type=int, help="Override simulation seed for reproducible contexts")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _profile_target(args)


if __name__ == "__main__":
    main()
