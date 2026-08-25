from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_specsim.dataset import load_samples
from edge_specsim.models import PromptSample

LIST_COLUMNS = {
    "history_token_ids",
    "context_token_ids",
    "proposed_token_ids",
    "committed_token_ids",
    "target_greedy_token_ids",
}


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parse_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _row_to_record(row: pd.Series) -> dict[str, object]:
    record: dict[str, object] = {}
    for key, value in row.items():
        parsed = _parse_value(value)
        if key in LIST_COLUMNS:
            record[key] = [] if parsed is None else parsed
        else:
            record[key] = parsed
    return record


def _decode_full_answer(trace_rows: list[dict[str, object]], model_name: str | None) -> str:
    committed_token_ids: list[int] = []
    fallback_text = "".join(
        str(row.get("committed_text", "") or "") for row in trace_rows
    ).strip()
    for row in trace_rows:
        token_ids = row.get("committed_token_ids")
        if isinstance(token_ids, list):
            committed_token_ids.extend(int(token_id) for token_id in token_ids)
    if not committed_token_ids or not model_name:
        return fallback_text
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        )
        return tokenizer.decode(committed_token_ids, skip_special_tokens=True).strip()
    except Exception:
        return fallback_text


def _render_csv_trace(trace: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append(f"sample_id: {trace['sample_id']}")
    lines.append(f"dataset_name: {trace['dataset_name']}")
    lines.append(f"prompt:\n{trace['prompt']}")
    lines.append("")
    lines.append(f"total rounds: {trace['rounds']}")
    lines.append(f"final generated tokens: {trace['final_generated_tokens']}")
    lines.append(f"full answer:\n{trace['full_answer']}")
    lines.append(f"completed: {trace['completed']}")
    lines.append("")
    for entry in trace["trace"]:
        lines.append(
            f"round {entry['round_id']} | batch {entry['verification_batch_id']}/{entry['verification_batch_size']} | "
            f"gamma={entry['gamma']} | accepted={entry['accepted_length']} | useful={entry['useful_tokens']}"
        )
        for key in (
            "context_token_ids",
            "proposed_token_ids",
            "committed_token_ids",
            "target_greedy_token_ids",
            "proposed_text",
            "committed_text",
            "target_greedy_text",
        ):
            if key in entry and entry[key] not in (None, [], ""):
                lines.append(f"  {key}: {entry[key]}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_csv_trace(df: pd.DataFrame, sample_id: str, model_name: str | None = None) -> dict[str, object]:
    sample_df = df[df["sample_id"] == sample_id].copy()
    if sample_df.empty:
        available = sorted(df["sample_id"].dropna().astype(str).unique().tolist())
        raise SystemExit(
            f"No rows found for sample_id={sample_id!r}. Available sample_ids: {available}"
        )
    sample_df["round_id"] = pd.to_numeric(sample_df["round_id"], errors="coerce")
    sample_df = sample_df.sort_values(["round_id", "verification_batch_id", "client_id"])
    first = sample_df.iloc[0]
    trace_rows = [_row_to_record(row) for _, row in sample_df.iterrows()]
    full_answer = _decode_full_answer(trace_rows, model_name)
    completed = bool(sample_df.iloc[-1].get("reached_eos", False)) or int(
        sample_df.iloc[-1]["generated_for_sample"]
    ) >= int(sample_df.iloc[-1]["max_new_tokens"])
    return {
        "sample_id": sample_id,
        "dataset_name": first["dataset_name"],
        "prompt": first["prompt"],
        "rounds": len(sample_df),
        "final_generated_tokens": int(sample_df.iloc[-1]["generated_for_sample"]),
        "full_answer": full_answer,
        "completed": completed,
        "trace": trace_rows,
    }


def _apply_overrides(
    cfg: dict,
    dataset_name: str | None,
    num_questions: int | None,
    split: str | None,
    math_subject: str | None,
    target_model: str | None,
    draft_model: str | None,
) -> dict:
    dataset_cfg = cfg.setdefault("dataset", {})
    if dataset_name is not None:
        dataset_cfg["name"] = dataset_name
    if num_questions is not None:
        dataset_cfg["num_questions"] = num_questions
    if split is not None:
        dataset_cfg["split"] = split
    if math_subject is not None:
        dataset_cfg["math_subject"] = math_subject

    model_cfg = cfg.setdefault("models", {})
    if target_model is not None:
        model_cfg["target"] = target_model
    if draft_model is not None:
        model_cfg["draft"] = draft_model
    return cfg


def _select_sample(
    samples: list[PromptSample],
    sample_id: str | None,
    sample_index: int,
) -> PromptSample:
    if sample_id is not None:
        for sample in samples:
            if sample.sample_id == sample_id:
                return sample
        available = [sample.sample_id for sample in samples]
        raise SystemExit(f"sample_id={sample_id!r} not found. Available: {available}")
    if sample_index < 0 or sample_index >= len(samples):
        raise SystemExit(
            f"sample_index={sample_index} out of range for {len(samples)} loaded samples"
        )
    return samples[sample_index]


def _decode_token(tokenizer: AutoTokenizer, token_id: int | None) -> str:
    if token_id is None:
        return ""
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _single_round_report(
    verifier: VLLMCandidateVerifier,
    draft: DraftResult,
    verification: VerificationResult,
) -> dict[str, object]:
    steps: list[dict[str, object]] = []
    inspected = len(verification.acceptance_probabilities)
    for offset, token_id in enumerate(draft.token_ids):
        draft_log_q = draft.log_q_proposed[offset]
        target_entry = (
            verification.target_logprobs[offset]
            if offset < len(verification.target_logprobs)
            else {}
        )
        target_log_p = float(target_entry.get(int(token_id), float("-inf")))
        accepted = offset < verification.accepted_length
        rejected_here = (
            verification.accepted_length < inspected
            and offset == verification.accepted_length
        )
        steps.append(
            {
                "position": offset + 1,
                "token_id": int(token_id),
                "token_text": _decode_token(verifier.tokenizer, int(token_id)),
                "draft_log_q": float(draft_log_q),
                "target_log_p": target_log_p,
                "accept_prob": (
                    float(verification.acceptance_probabilities[offset])
                    if offset < inspected
                    else None
                ),
                "accepted": accepted,
                "rejected_here": rejected_here,
                "target_greedy_token_id": (
                    int(verification.target_greedy_token_ids[offset])
                    if offset < len(verification.target_greedy_token_ids)
                    else None
                ),
                "target_greedy_text": (
                    _decode_token(
                        verifier.tokenizer,
                        int(verification.target_greedy_token_ids[offset]),
                    )
                    if offset < len(verification.target_greedy_token_ids)
                    else ""
                ),
            }
        )
    return {
        "accepted_length": verification.accepted_length,
        "inspected_tokens": inspected,
        "proposed_token_ids": [int(token_id) for token_id in draft.token_ids],
        "proposed_text": draft.text,
        "committed_token_ids": [int(token_id) for token_id in verification.committed_token_ids],
        "committed_text": verification.committed_text,
        "correction_token_id": verification.correction_token_id,
        "correction_text": _decode_token(verifier.tokenizer, verification.correction_token_id),
        "bonus_token_id": verification.bonus_token_id,
        "bonus_text": _decode_token(verifier.tokenizer, verification.bonus_token_id),
        "reached_eos": verification.reached_eos,
        "steps": steps,
    }


def _render_live_trace(trace: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append(f"sample_id: {trace['sample_id']}")
    lines.append(f"dataset_name: {trace['dataset_name']}")
    lines.append(f"mode: {trace['decoding_mode']}")
    lines.append(f"temperature: {trace['temperature']}")
    lines.append(f"gamma: {trace['gamma']}")
    lines.append(f"rounds_executed: {trace['rounds_executed']}")
    lines.append("")
    lines.append("prompt:")
    lines.append(str(trace["prompt"]))
    lines.append("")
    for round_trace in trace["rounds"]:
        lines.append(
            f"round {round_trace['round_id']} | accepted={round_trace['accepted_length']} "
            f"| inspected={round_trace['inspected_tokens']} | reached_eos={round_trace['reached_eos']}"
        )
        lines.append(f"  proposed_token_ids: {round_trace['proposed_token_ids']}")
        if round_trace["proposed_text"]:
            lines.append(f"  proposed_text: {round_trace['proposed_text']!r}")
        for step in round_trace["steps"]:
            lines.append(
                "  pos={position} token={token_id} {token_text!r} "
                "q={draft_log_q:.6f} p={target_log_p:.6f} "
                "accept_prob={accept_prob} accepted={accepted} rejected_here={rejected_here} "
                "target_greedy={target_greedy_token_id} {target_greedy_text!r}".format(
                    **step
                )
            )
        if round_trace["correction_token_id"] is not None:
            lines.append(
                f"  correction_token: {round_trace['correction_token_id']} "
                f"{round_trace['correction_text']!r}"
            )
        if round_trace["bonus_token_id"] is not None:
            lines.append(
                f"  bonus_token: {round_trace['bonus_token_id']} "
                f"{round_trace['bonus_text']!r}"
            )
        lines.append(f"  committed_token_ids: {round_trace['committed_token_ids']}")
        lines.append(f"  committed_text: {round_trace['committed_text']!r}")
        lines.append("")
    lines.append(f"final_token_ids: {trace['final_token_ids']}")
    lines.append(f"final_text: {trace['final_text']!r}")
    return "\n".join(lines).rstrip() + "\n"


async def _run_live_trace(args: argparse.Namespace) -> dict[str, object]:
    from edge_specsim.draft_worker import DraftRequest, DraftWorker
    from edge_specsim.models import DraftResult
    from edge_specsim.target_client import VLLMCandidateVerifier

    cfg = _load_config(args.config)
    cfg = _apply_overrides(
        cfg,
        dataset_name=args.dataset,
        num_questions=args.num_questions,
        split=args.split,
        math_subject=args.math_subject,
        target_model=args.target_model,
        draft_model=args.draft_model,
    )
    samples = load_samples(cfg["dataset"])
    sample = _select_sample(samples, args.sample_id, args.sample_index)
    model_cfg = cfg["models"]
    decoding_cfg = cfg.setdefault("decoding", {})
    decoding_mode = str(decoding_cfg.get("mode", "greedy")).lower()
    temperature = float(
        args.temperature
        if args.temperature is not None
        else decoding_cfg.get("temperature", 0.0)
    )
    gamma_default = int(
        max(
            cfg.get("controller", {}).get("gamma_choices", [cfg["simulation"].get("gamma_max", 1)])
        )
    )
    gamma = max(0, int(args.gamma if args.gamma is not None else gamma_default))
    max_rounds = max(1, int(args.max_rounds))

    verifier = VLLMCandidateVerifier(
        model_name=model_cfg["target"],
        gpu_memory_utilization=float(model_cfg.get("target_gpu_memory_utilization", 0.88)),
        max_model_len=int(model_cfg.get("target_max_model_len", 8192)),
        dtype=str(model_cfg.get("target_dtype", "auto")),
        enforce_eager=bool(model_cfg.get("target_enforce_eager", False)),
        enable_prefix_caching=bool(model_cfg.get("target_enable_prefix_caching", True)),
        sampling_temperature=temperature,
        seed=int(cfg["simulation"]["seed"]),
    )
    worker = DraftWorker(
        0,
        model_cfg["draft"],
        model_cfg["draft_devices"][0],
        enable_kv_cache=bool(model_cfg.get("draft_enable_kv_cache", True)),
    )

    client_id = 0
    history_token_ids: list[int] = []
    remaining = sample.max_new_tokens
    await worker.reset_client(client_id)
    verifier.set_confirmed_prefix(client_id, [])
    round_traces: list[dict[str, object]] = []

    for round_id in range(max_rounds):
        if remaining <= 0:
            break
        context_token_ids = verifier.build_context_token_ids(sample.prompt, history_token_ids)
        round_gamma = min(gamma, max(0, remaining - 1))
        if round_gamma > 0:
            draft = await worker.generate(
                DraftRequest(
                    client_id=client_id,
                    context_token_ids=context_token_ids,
                    gamma=round_gamma,
                    temperature=temperature,
                )
            )
        else:
            draft = DraftResult(client_id, [], [], "", 0.0, 0)

        verification = verifier.verify_batch_sync(
            [(context_token_ids, draft, remaining)]
        )[0]
        committed = list(verification.committed_token_ids)
        await worker.commit(client_id, committed)
        history_token_ids.extend(committed)
        verifier.set_confirmed_prefix(client_id, history_token_ids)
        remaining -= len(committed)
        round_trace = _single_round_report(verifier, draft, verification)
        round_trace["round_id"] = round_id
        round_traces.append(round_trace)
        if verification.reached_eos or not committed:
            break

    final_text = verifier.tokenizer.decode(history_token_ids, skip_special_tokens=True)
    return {
        "sample_id": sample.sample_id,
        "dataset_name": sample.dataset_name,
        "prompt": sample.prompt,
        "decoding_mode": decoding_mode,
        "temperature": temperature,
        "gamma": gamma,
        "rounds_executed": len(round_traces),
        "rounds": round_traces,
        "final_token_ids": history_token_ids,
        "final_text": final_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace one sample either from a saved CSV or by running live rejection-sampling verification."
    )
    parser.add_argument("--csv", help="Offline mode: path to simulation CSV output")
    parser.add_argument("--config", default="configs/default.yaml", help="Live mode config path")
    parser.add_argument("--sample-id", help="Sample id to inspect")
    parser.add_argument("--sample-index", type=int, default=0, help="Live mode fallback sample index")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Tokenizer/model name used to decode CSV traces")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human-readable text")
    parser.add_argument("--output", help="Write the trace to this file instead of stdout")
    parser.add_argument("--dataset", choices=["gsm8k", "math", "cnn_dailymail"])
    parser.add_argument("--num-questions", type=int)
    parser.add_argument("--split")
    parser.add_argument("--math-subject")
    parser.add_argument("--target-model")
    parser.add_argument("--draft-model")
    parser.add_argument("--gamma", type=int, help="Live mode: speculative length to trace")
    parser.add_argument("--max-rounds", type=int, default=4, help="Live mode: maximum rounds to execute")
    parser.add_argument("--temperature", type=float, help="Live mode: decoding temperature")
    args = parser.parse_args()

    if args.csv:
        if not args.sample_id:
            raise SystemExit("--sample-id is required in --csv mode")
        df = pd.read_csv(args.csv)
        trace = build_csv_trace(df, args.sample_id, args.model)
        rendered = (
            json.dumps(trace, indent=2, ensure_ascii=False)
            if args.json
            else _render_csv_trace(trace)
        )
    else:
        trace = asyncio.run(_run_live_trace(args))
        rendered = (
            json.dumps(trace, indent=2, ensure_ascii=False)
            if args.json
            else _render_live_trace(trace)
        )

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
