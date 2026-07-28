from __future__ import annotations

import math
import os
import random
import time
from typing import Any

import torch

from transformers import AutoTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


_original_config_register = CONFIG_MAPPING.register


def _register_config_with_aimv2_compat(model_type, config, exist_ok=False):
    if model_type == "aimv2":
        exist_ok = True
    return _original_config_register(model_type, config, exist_ok=exist_ok)


CONFIG_MAPPING.register = _register_config_with_aimv2_compat


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _format_hf_load_error(model_name: str, purpose: str, exc: Exception) -> OSError:
    message = str(exc)
    if "401" in message or "Unauthorized" in message or "expired" in message:
        detail = (
            f"Hugging Face authentication failed while loading the {purpose} for "
            f"{model_name!r}. The cached token may be expired. Run `huggingface-cli login` "
            "or set a fresh HF_TOKEN."
        )
    else:
        detail = (
            f"Unable to load the {purpose} for {model_name!r}. If this is a gated or "
            "private Hugging Face repo, set HF_TOKEN or run `huggingface-cli login`."
        )
    error = OSError(detail)
    raise error from exc

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from .models import DraftProposal, VerificationResult


class VLLMCandidateVerifier:
    """Batched candidate verification using vLLM prompt log-probabilities.

    For every request, the target receives context_token_ids + candidate_token_ids.
    Prompt log-probabilities expose the target distribution at every candidate
    position. Acceptance uses min(1, p/q) on the proposed token. If a proposal
    is rejected, the correction token is sampled from the residual distribution
    max(0, p - q). If all candidates are accepted, the next token is sampled from
    the target distribution at the following position.
    """

    def __init__(
        self,
        model_name: str,
        gpu_memory_utilization: float = 0.88,
        max_model_len: int = 8192,
        dtype: str = "auto",
        enforce_eager: bool = False,
        enable_prefix_caching: bool = True,
        sampling_temperature: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.model_name = model_name
        self.rng = random.Random(seed)
        token = _hf_token()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        except OSError as exc:
            raise _format_hf_load_error(model_name, "tokenizer", exc)
        self.vocab_size = len(self.tokenizer)
        self.eos_token_id = self.tokenizer.eos_token_id
        try:
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                dtype=dtype,
                enforce_eager=enforce_eager,
                generation_config="vllm",
                enable_prefix_caching=enable_prefix_caching,
                hf_token=token,
            )
        except OSError as exc:
            raise _format_hf_load_error(model_name, "vLLM model", exc)
        self.sampling = SamplingParams(
            temperature=sampling_temperature,
            top_p=1.0,
            max_tokens=1,
            prompt_logprobs=-1,
            logprobs=-1,
            detokenize=False,
            skip_special_tokens=False,
        )

    def build_context_token_ids(self, user_prompt: str, history_token_ids: list[int]) -> list[int]:
        messages = [{"role": "user", "content": user_prompt}]
        base_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return list(base_ids) + list(history_token_ids)

    @staticmethod
    def _logprob_value(value: Any) -> float:
        if hasattr(value, "logprob"):
            return float(value.logprob)
        if isinstance(value, (int, float)):
            return float(value)
        raise TypeError(f"Unsupported logprob value type: {type(value)!r}")

    @classmethod
    def _logprob_tensor(cls, entry: Any, vocab_size: int) -> torch.Tensor:
        if entry is None or not isinstance(entry, dict) or not entry:
            raise RuntimeError(
                "vLLM did not return prompt logprobs for a candidate position. "
                "Ensure prompt_logprobs is supported by the installed vLLM version."
            )
        logprobs = torch.full((vocab_size,), float("-inf"), dtype=torch.float32)
        for token_id, value in entry.items():
            logprobs[int(token_id)] = cls._logprob_value(value)
        return logprobs

    def _sample_from_logprobs(self, logprobs: torch.Tensor) -> int:
        finite_mask = torch.isfinite(logprobs)
        if not torch.any(finite_mask):
            return int(torch.argmax(logprobs).item())
        token_ids = torch.nonzero(finite_mask, as_tuple=False).flatten()
        weights = torch.exp(logprobs[finite_mask]).to(torch.float32)
        total = float(weights.sum().item())
        if not math.isfinite(total) or total <= 0.0:
            return int(token_ids[int(torch.argmax(weights).item())].item())
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.rng.randrange(1, 2**63 - 1))
        sample_index = int(torch.multinomial(weights, 1, generator=generator).item())
        return int(token_ids[sample_index].item())

    def _sample_from_weights(self, weights: torch.Tensor) -> int:
        positive_mask = weights > 0
        if not torch.any(positive_mask):
            return int(torch.argmax(weights).item())
        token_ids = torch.nonzero(positive_mask, as_tuple=False).flatten()
        positive_weights = weights[positive_mask].to(torch.float32)
        total = float(positive_weights.sum().item())
        if not math.isfinite(total) or total <= 0.0:
            return int(token_ids[int(torch.argmax(positive_weights).item())].item())
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.rng.randrange(1, 2**63 - 1))
        sample_index = int(torch.multinomial(positive_weights, 1, generator=generator).item())
        return int(token_ids[sample_index].item())

    def verify_batch_sync(
        self,
        requests: list[tuple[list[int], DraftProposal, int]],
    ) -> list[VerificationResult]:
        prompts = [
            TokensPrompt(prompt_token_ids=context_ids + draft.token_ids)
            for context_ids, draft, _ in requests
        ]
        start = time.perf_counter()
        outputs = self.llm.generate(prompts, self.sampling, use_tqdm=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_request_ms = elapsed_ms / max(1, len(requests))

        results: list[VerificationResult] = []
        for output, (context_ids, draft, remaining_tokens) in zip(outputs, requests):
            prompt_logprobs = output.prompt_logprobs
            if prompt_logprobs is None:
                raise RuntimeError("Missing prompt_logprobs in vLLM output")

            accepted = 0
            target_logprobs: list[torch.Tensor] = []
            acceptance_probabilities: list[float] = []
            target_greedy: list[int] = []
            correction_token: int | None = None
            candidate_start = len(context_ids)

            for offset, proposed_id in enumerate(draft.token_ids):
                target_entry = self._logprob_tensor(prompt_logprobs[candidate_start + offset], self.vocab_size)
                draft_entry = draft.draft_logprobs[offset]
                if isinstance(draft_entry, torch.Tensor):
                    draft_logprobs = draft_entry.to(dtype=torch.float32, device="cpu")
                else:
                    draft_logprobs = torch.as_tensor(draft_entry, dtype=torch.float32)
                target_logprobs.append(target_entry)
                target_id = int(torch.argmax(target_entry).item())
                target_greedy.append(target_id)

                log_p = float(target_entry[proposed_id].item())
                log_q = float(draft_logprobs[proposed_id].item())
                if math.isfinite(log_p) and math.isfinite(log_q):
                    acceptance_probability = 1.0 if log_p >= log_q else math.exp(log_p - log_q)
                elif math.isfinite(log_p) and not math.isfinite(log_q):
                    acceptance_probability = 1.0
                else:
                    acceptance_probability = 0.0
                acceptance_probabilities.append(acceptance_probability)
                if self.rng.random() > acceptance_probability:
                    residual = torch.exp(target_entry) - torch.exp(draft_logprobs)
                    residual.clamp_(min=0.0)
                    correction_token = self._sample_from_weights(residual)
                    break
                accepted += 1

            committed = list(draft.token_ids[:accepted])
            if len(committed) < remaining_tokens:
                if correction_token is not None:
                    committed.append(correction_token)
                elif output.outputs and output.outputs[0].logprobs:
                    bonus_entry = self._logprob_tensor(output.outputs[0].logprobs[0], self.vocab_size)
                    target_logprobs.append(bonus_entry)
                    bonus_id = self._sample_from_logprobs(bonus_entry)
                    target_greedy.append(int(torch.argmax(bonus_entry).item()))
                    committed.append(bonus_id)

            committed = committed[:remaining_tokens]
            reached_eos = self.eos_token_id is not None and self.eos_token_id in committed
            results.append(
                VerificationResult(
                    accepted_length=accepted,
                    committed_token_ids=committed,
                    committed_text=self.tokenizer.decode(committed, skip_special_tokens=True),
                    target_latency_ms=per_request_ms,
                    target_logprobs=target_logprobs,
                    acceptance_probabilities=acceptance_probabilities,
                    target_greedy_token_ids=target_greedy,
                    reached_eos=reached_eos,
                    finish_reason="eos" if reached_eos else None,
                )
            )
        return results
