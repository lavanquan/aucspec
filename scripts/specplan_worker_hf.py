#!/usr/bin/env python3
"""SpecPlan JSONL worker using Hugging Face target and draft models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "legacy" / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from edge_specsim.hf_target_client import HFGreedyCandidateVerifier


PROTOCOL_VERSION = 1


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype {name!r}")


def _load_draft(model_name: str, device: str, dtype: str):
    token = _hf_token()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=_dtype(dtype),
        low_cpu_mem_usage=True,
        token=token,
    ).to(device)
    model.eval()
    return tokenizer, model


def _same_tokenizer(left: Any, right: Any) -> bool:
    return (
        len(left) == len(right)
        and left.eos_token_id == right.eos_token_id
        and left.bos_token_id == right.bos_token_id
        and left.get_vocab() == right.get_vocab()
    )


def _draft_greedy(
    tokenizer: Any,
    model: Any,
    device: str,
    context_token_ids: list[int],
    depth: int,
) -> list[int]:
    if depth <= 0:
        return []
    input_ids = torch.tensor(
        [context_token_ids],
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=depth,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return [
        int(token)
        for token in output[0, input_ids.shape[1] :].detach().cpu().tolist()
    ]


def _resource_label(role: str, device: str) -> str:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index
        index = 0 if index is None else index
        return f"{role}:{device}:{torch.cuda.get_device_name(index)}"
    return f"{role}:{device}"


def _respond(request_id: int, **payload: Any) -> None:
    response = {"ok": True, "request_id": request_id, **payload}
    print(json.dumps(response, separators=(",", ":")), flush=True)


def _error(request_id: int, exc: BaseException) -> None:
    response = {
        "ok": False,
        "request_id": request_id,
        "error": f"{type(exc).__name__}: {exc}",
    }
    print(json.dumps(response, separators=(",", ":")), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--draft-device", default="cuda:1")
    parser.add_argument("--target-dtype", default="bfloat16")
    parser.add_argument("--draft-dtype", default="bfloat16")
    args = parser.parse_args()

    if torch.cuda.device_count() < 2:
        raise RuntimeError("HF speculative worker requires two visible GPU GCDs")

    target = HFGreedyCandidateVerifier(
        args.target_model,
        device=args.target_device,
        dtype=args.target_dtype,
    )
    draft_tokenizer, draft_model = _load_draft(
        args.draft_model,
        args.draft_device,
        args.draft_dtype,
    )
    if not _same_tokenizer(target.tokenizer, draft_tokenizer):
        raise RuntimeError(
            "target and draft tokenizers/vocabularies are not identical"
        )

    active_resources = [
        _resource_label("target", args.target_device),
        _resource_label("draft", args.draft_device),
    ]

    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = -1
        try:
            request = json.loads(line)
            request_id = int(request["request_id"])
            op = request.get("op")

            if op == "hello":
                if int(request.get("protocol_version", -1)) != PROTOCOL_VERSION:
                    raise ValueError("unsupported protocol version")
                if request.get("target_model") != args.target_model:
                    raise ValueError("target model identity mismatch")
                if request.get("draft_model") != args.draft_model:
                    raise ValueError("draft model identity mismatch")
                _respond(
                    request_id,
                    protocol_version=PROTOCOL_VERSION,
                    target_model=args.target_model,
                    draft_model=args.draft_model,
                    capabilities=["build_context", "generate_cycle", "target_only"],
                    runtime="hf_causal_greedy",
                    active_resources=active_resources,
                )
                continue

            if op == "build_context":
                prompt = request.get("prompt")
                history = request.get("history_token_ids")
                if not isinstance(prompt, str):
                    raise ValueError("prompt must be a string")
                if not isinstance(history, list) or not all(
                    isinstance(token, int) for token in history
                ):
                    raise ValueError("history_token_ids must be an integer list")
                _respond(
                    request_id,
                    context_token_ids=target.build_context_token_ids(
                        prompt,
                        history,
                    ),
                )
                continue

            if op == "target_only":
                prompt = request.get("prompt")
                max_new_tokens = request.get("max_new_tokens")
                if not isinstance(prompt, str):
                    raise ValueError("prompt must be a string")
                if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
                    raise ValueError("max_new_tokens must be a non-negative integer")
                _respond(
                    request_id,
                    output_token_ids=target.generate_target_only_greedy(
                        prompt,
                        max_new_tokens,
                    ),
                )
                continue

            if op == "generate_cycle":
                context = request.get("context_token_ids")
                depth = request.get("requested_depth")
                remaining = request.get("remaining_tokens")
                if not isinstance(context, list) or not context or not all(
                    isinstance(token, int) for token in context
                ):
                    raise ValueError(
                        "context_token_ids must be a non-empty integer list"
                    )
                if not isinstance(depth, int) or depth <= 0:
                    raise ValueError("requested_depth must be positive")
                if not isinstance(remaining, int) or remaining <= 0:
                    raise ValueError("remaining_tokens must be positive")

                proposal_depth = min(depth, max(0, remaining - 1))
                draft = _draft_greedy(
                    draft_tokenizer,
                    draft_model,
                    args.draft_device,
                    context,
                    proposal_depth,
                )
                result = target.verify_sync(
                    context,
                    draft,
                    remaining,
                )
                _respond(
                    request_id,
                    proposed_tokens=len(draft),
                    accepted_prefix_tokens=result.accepted_length,
                    committed_token_ids=result.committed_token_ids,
                    reached_eos=result.reached_eos,
                    active_resources=active_resources,
                    target_greedy_token_ids=result.target_greedy_token_ids,
                    correction_token_id=result.correction_token_id,
                    bonus_token_id=result.bonus_token_id,
                )
                continue

            if op == "shutdown":
                _respond(request_id)
                return 0

            raise ValueError(f"unsupported op {op!r}")
        except BaseException as exc:
            _error(request_id, exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
