from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


LIST_COLUMNS = {
    "history_token_ids",
    "context_token_ids",
    "proposed_token_ids",
    "committed_token_ids",
    "target_greedy_token_ids",
}


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
        if key in LIST_COLUMNS:
            parsed = _parse_value(value)
            record[key] = [] if parsed is None else parsed
        else:
            parsed = _parse_value(value)
            if isinstance(parsed, float) and parsed.is_integer():
                parsed = int(parsed)
            record[key] = parsed
    return record


def _build_full_answer(trace_rows: list[dict[str, object]]) -> str:
    pieces: list[str] = []
    committed_token_ids: list[int] = []
    for row in trace_rows:
        token_ids = row.get("committed_token_ids")
        if isinstance(token_ids, list):
            committed_token_ids.extend(int(token_id) for token_id in token_ids)
        committed_text = row.get("committed_text")
        if isinstance(committed_text, str) and committed_text:
            pieces.append(committed_text)
    return "".join(pieces).strip()


def _decode_full_answer(trace_rows: list[dict[str, object]], model_name: str | None) -> str:
    committed_token_ids: list[int] = []
    fallback_text = _build_full_answer(trace_rows)
    for row in trace_rows:
        token_ids = row.get("committed_token_ids")
        if isinstance(token_ids, list):
            committed_token_ids.extend(int(token_id) for token_id in token_ids)
    if not committed_token_ids or not model_name:
        return fallback_text
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    except Exception:
        return fallback_text
    try:
        return tokenizer.decode(committed_token_ids, skip_special_tokens=True).strip()
    except Exception:
        return fallback_text


def _render_answer(trace: dict[str, object]) -> str:
    separator = "=" * 88
    lines: list[str] = [separator]
    lines.append(f"SAMPLE: {trace['sample_id']}")
    lines.append("")
    lines.append("PROMPT:")
    lines.append(str(trace["prompt"]).rstrip())
    lines.append("")
    lines.append("FULL ANSWER:")
    lines.append(str(trace["full_answer"]).rstrip())
    lines.append(separator)
    return "\n".join(lines).rstrip() + "\n"


def _render_trace(trace: dict[str, object]) -> str:
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


def build_trace(df: pd.DataFrame, sample_id: str, model_name: str | None = None) -> dict[str, object]:
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
    completed = bool(sample_df.iloc[-1].get("reached_eos", False)) or int(sample_df.iloc[-1]["generated_for_sample"]) >= int(sample_df.iloc[-1]["max_new_tokens"])
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the full speculative-decoding trace for one sample_id.")
    parser.add_argument("sample_id", help="The sample_id to inspect, for example gsm8k-test-1")
    parser.add_argument("--csv", default="results/rounds.csv", help="Path to the simulation CSV output")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Tokenizer/model name used to decode the final answer")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human-readable text")
    parser.add_argument("--trace", action="store_true", help="Print the detailed round-by-round trace instead of the compact answer view")
    parser.add_argument("--output", help="Write the trace to this file instead of stdout")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    trace = build_trace(df, args.sample_id, args.model)

    if args.json:
        rendered = json.dumps(trace, indent=2, ensure_ascii=False)
    elif args.trace:
        rendered = _render_trace(trace)
    else:
        rendered = _render_answer(trace)

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()