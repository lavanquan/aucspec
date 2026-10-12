from __future__ import annotations

import os
import time
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import VerificationResult


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported dtype {name!r}")
    return aliases[normalized]


def greedy_verify_choices(
    candidate_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    *,
    remaining_tokens: int,
    eos_token_id: int | None,
) -> tuple[int, list[int], int | None, int | None, bool]:
    """Apply fixed-depth greedy accept/correct semantics to target choices.

    target_token_ids contains one target next-token choice for every candidate
    position followed by exactly one bonus choice.
    """

    candidates = [int(token) for token in candidate_token_ids]
    choices = [int(token) for token in target_token_ids]
    if remaining_tokens <= 0:
        raise ValueError("remaining_tokens must be positive")
    if len(choices) != len(candidates) + 1:
        raise ValueError(
            "target_token_ids must contain one choice per candidate plus one bonus"
        )

    accepted = 0
    correction_token: int | None = None
    reached_eos = False

    for offset, proposed in enumerate(candidates):
        target = choices[offset]
        if proposed != target:
            correction_token = target
            break
        accepted += 1
        if eos_token_id is not None and proposed == int(eos_token_id):
            reached_eos = True
            break

    committed = list(candidates[:accepted])
    bonus_token: int | None = None
    if len(committed) < remaining_tokens and not reached_eos:
        if correction_token is not None:
            committed.append(correction_token)
        elif accepted == len(candidates):
            bonus_token = choices[len(candidates)]
            committed.append(bonus_token)

    if eos_token_id is not None and int(eos_token_id) in committed:
        eos_index = committed.index(int(eos_token_id))
        committed = committed[: eos_index + 1]
        reached_eos = True

    committed = committed[:remaining_tokens]
    return accepted, committed, correction_token, bonus_token, reached_eos


class HFGreedyCandidateVerifier:
    """Causal Hugging Face target verifier for greedy speculative decoding.

    For one request, the target runs one forward pass over context plus
    candidate tokens. Causal LM logits at position len(context)-1+j predict
    candidate token j; the final position predicts the bonus token.

    The implementation is intentionally single-request and correctness-first.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
    ) -> None:
        self.model_name = model_name
        self.device = device
        token = _hf_token()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.eos_token_id = self.tokenizer.eos_token_id
        model_dtype = _dtype_from_name(dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            token=token,
        ).to(device)
        self.model.eval()

    def build_context_token_ids(
        self,
        user_prompt: str,
        history_token_ids: Sequence[int],
    ) -> list[int]:
        messages = [{"role": "user", "content": user_prompt}]
        base_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return list(base_ids) + [int(token) for token in history_token_ids]

    @staticmethod
    def _greedy_ids_from_logits(
        logits: torch.Tensor,
        *,
        context_length: int,
        candidate_length: int,
    ) -> list[int]:
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError("expected logits with shape [1, sequence, vocab]")
        if context_length <= 0:
            raise ValueError("context must contain at least one token")
        expected_sequence = context_length + candidate_length
        if int(logits.shape[1]) != expected_sequence:
            raise ValueError(
                f"logit sequence length {logits.shape[1]} != input length "
                f"{expected_sequence}"
            )
        start = context_length - 1
        stop = start + candidate_length + 1
        selected = logits[0, start:stop, :]
        if int(selected.shape[0]) != candidate_length + 1:
            raise ValueError("unable to extract candidate plus bonus target choices")
        return [
            int(token)
            for token in torch.argmax(selected, dim=-1).detach().cpu().tolist()
        ]

    def verify_sync(
        self,
        context_token_ids: Sequence[int],
        candidate_token_ids: Sequence[int],
        remaining_tokens: int,
    ) -> VerificationResult:
        context = [int(token) for token in context_token_ids]
        candidates = [int(token) for token in candidate_token_ids]
        if not context:
            raise ValueError("target context cannot be empty")
        if remaining_tokens <= 0:
            raise ValueError("remaining_tokens must be positive")

        input_ids = torch.tensor(
            [context + candidates],
            dtype=torch.long,
            device=self.device,
        )
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
            )
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device=self.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        target_choices = self._greedy_ids_from_logits(
            output.logits,
            context_length=len(context),
            candidate_length=len(candidates),
        )
        (
            accepted,
            committed,
            correction_token,
            bonus_token,
            reached_eos,
        ) = greedy_verify_choices(
            candidates,
            target_choices,
            remaining_tokens=remaining_tokens,
            eos_token_id=self.eos_token_id,
        )

        inspected = min(len(candidates), accepted + (1 if accepted < len(candidates) else 0))
        return VerificationResult(
            accepted_length=accepted,
            committed_token_ids=committed,
            committed_text=self.tokenizer.decode(
                committed,
                skip_special_tokens=True,
            ),
            target_latency_ms=elapsed_ms,
            target_logprobs=[],
            acceptance_probabilities=[
                1.0 if candidates[index] == target_choices[index] else 0.0
                for index in range(inspected)
            ],
            target_greedy_token_ids=target_choices,
            reached_eos=reached_eos,
            finish_reason="eos" if reached_eos else None,
            correction_token_id=correction_token,
            bonus_token_id=bonus_token,
        )

    def generate_target_only_greedy(
        self,
        user_prompt: str,
        max_new_tokens: int,
    ) -> list[int]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        history: list[int] = []
        while len(history) < max_new_tokens:
            context = self.build_context_token_ids(user_prompt, history)
            result = self.verify_sync(
                context,
                (),
                max_new_tokens - len(history),
            )
            if not result.committed_token_ids:
                break
            history.extend(result.committed_token_ids)
            if result.reached_eos:
                break
        return history
