from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


_original_config_register = CONFIG_MAPPING.register


def _register_config_with_aimv2_compat(model_type, config, exist_ok=False):
    if model_type == "aimv2":
        exist_ok = True
    return _original_config_register(model_type, config, exist_ok=exist_ok)


CONFIG_MAPPING.register = _register_config_with_aimv2_compat


def _ensure_vllm_subprocess_pythonpath() -> None:
    src_dir = str(Path(__file__).resolve().parents[1])
    existing = os.environ.get("PYTHONPATH")
    if existing:
        entries = existing.split(os.pathsep)
        if src_dir not in entries:
            os.environ["PYTHONPATH"] = os.pathsep.join([src_dir, *entries])
    else:
        os.environ["PYTHONPATH"] = src_dir


_ensure_vllm_subprocess_pythonpath()


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


def _token_id_upper_bound(tokenizer: AutoTokenizer) -> int:
    upper_bound = len(tokenizer)
    special_ids = [
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    ]
    for token_id in special_ids:
        if token_id is not None:
            upper_bound = max(upper_bound, int(token_id) + 1)
    added_vocab = tokenizer.get_added_vocab()
    if added_vocab:
        upper_bound = max(upper_bound, max(int(token_id) for token_id in added_vocab.values()) + 1)
    return upper_bound

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from .acceptance import (
    acceptance_probability,
    greedy_correction_token,
    sample_correction_token,
    should_accept,
)
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
        max_logprobs: int | None = None,
        dtype: str = "auto",
        enforce_eager: bool = False,
        enable_prefix_caching: bool = True,
        sampling_temperature: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.model_name = model_name
        self.rng = random.Random(seed)
        self._confirmed_prefix_hashes: dict[int, int] = {}
        token = _hf_token()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        except OSError as exc:
            raise _format_hf_load_error(model_name, "tokenizer", exc)
        try:
            self.config = AutoConfig.from_pretrained(model_name, token=token)
        except OSError as exc:
            raise _format_hf_load_error(model_name, "model config", exc)
        tokenizer_upper_bound = _token_id_upper_bound(self.tokenizer)
        config_vocab_size = int(getattr(self.config, "vocab_size", 0) or 0)
        self.vocab_size = max(config_vocab_size, tokenizer_upper_bound)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.greedy_only = sampling_temperature == 0.0
        requested_max_logprobs = max(
            1,
            int(
                max_logprobs
                if max_logprobs is not None
                else (1 if self.greedy_only else self.vocab_size)
            ),
        )
        try:
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_logprobs=requested_max_logprobs,
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
            prompt_logprobs=requested_max_logprobs,
            logprobs=requested_max_logprobs,
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
    def _logprob_dict(cls, entry: Any) -> dict[int, float]:
        if entry is None or not isinstance(entry, dict) or not entry:
            raise RuntimeError(
                "vLLM did not return prompt logprobs for a candidate position. "
                "Ensure prompt_logprobs is supported by the installed vLLM version."
            )
        return {int(token_id): cls._logprob_value(value) for token_id, value in entry.items()}

    @staticmethod
    def _best_token_id(logprob_dict: dict[int, float]) -> int:
        return max(logprob_dict.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _lookup_logprob(logprob_dict: dict[int, float], token_id: int) -> float:
        return float(logprob_dict.get(int(token_id), float("-inf")))

    @staticmethod
    def prefix_hash(token_ids: list[int]) -> int:
        return hash(tuple(token_ids))

    def set_confirmed_prefix(self, client_id: int, token_ids: list[int]) -> None:
        self._confirmed_prefix_hashes[client_id] = self.prefix_hash(token_ids)

    def confirmed_prefix_hash(self, client_id: int) -> int | None:
        return self._confirmed_prefix_hashes.get(client_id)

    def _generate_one_greedy_token(self, prompt_token_ids: list[int]) -> int | None:
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            logprobs=1,
            detokenize=False,
            skip_special_tokens=False,
        )
        output = self.llm.generate(
            TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling,
            use_tqdm=False,
        )[0]
        if not output.outputs or not output.outputs[0].token_ids:
            return None
        return int(output.outputs[0].token_ids[0])

    def _target_dense_logprobs(self, logprob_dict: dict[int, float]) -> torch.Tensor:
        dense = torch.full(
            (self.vocab_size,),
            float("-inf"),
            dtype=torch.float32,
        )
        for token_id, value in logprob_dict.items():
            token_index = int(token_id)
            if 0 <= token_index < self.vocab_size:
                dense[token_index] = float(value)
        return dense

    def _sample_target_token(self, logprob_dict: dict[int, float]) -> int:
        if not logprob_dict:
            raise ValueError("target logprobs must not be empty")
        distribution = torch.softmax(
            torch.tensor(
                [float(logprob) for _, logprob in sorted(logprob_dict.items())],
                dtype=torch.float32,
            ),
            dim=0,
        )
        support = [token_id for token_id, _ in sorted(logprob_dict.items())]
        draw = self.rng.random()
        cumulative = 0.0
        selected_token = support[-1]
        for token_id, probability in zip(support, distribution.tolist()):
            cumulative += float(probability)
            selected_token = token_id
            if draw <= cumulative:
                return int(token_id)
        return int(selected_token)

    def generate_target_only_greedy(self, user_prompt: str, max_new_tokens: int) -> list[int]:
        history_token_ids: list[int] = []
        remaining_tokens = max_new_tokens
        while remaining_tokens > 0:
            context_token_ids = self.build_context_token_ids(user_prompt, history_token_ids)
            empty_draft = DraftProposal(
                client_id=0,
                token_ids=[],
                draft_logprobs=[],
                text="",
                latency_ms=0.0,
                worker_id=-1,
                distribution_payload="delta_proposal",
            )
            verification = self.verify_batch_sync(
                [(context_token_ids, empty_draft, remaining_tokens)]
            )[0]
            committed = verification.committed_token_ids
            if not committed:
                break
            history_token_ids.extend(committed)
            remaining_tokens -= len(committed)
            if verification.reached_eos:
                break
        return history_token_ids

    def verify_batch_sync(
        self,
        requests: list[tuple[list[int], DraftProposal, int]],
    ) -> list[VerificationResult]:
        start = time.perf_counter()
        prompts = [
            TokensPrompt(prompt_token_ids=list(context_ids) + list(draft.token_ids))
            for context_ids, draft, _ in requests
        ]
        outputs = self.llm.generate(
            prompts,
            self.sampling,
            use_tqdm=False,
        )
        results: list[VerificationResult] = []
        for (context_ids, draft, remaining_tokens), output in zip(requests, outputs):
            if not draft.supports_lossless_rejection_sampling(self.vocab_size):
                raise ValueError(
                    "Draft proposal payload is not sufficient for lossless rejection-sampling "
                    "verification. Use distribution_payload='delta_proposal' for greedy draft "
                    "or distribution_payload='full_vocab_logprobs' for sampling draft."
                )
            accepted = 0
            target_logprobs: list[dict[int, float]] = []
            acceptance_probabilities: list[float] = []
            target_greedy: list[int] = []
            correction_token: int | None = None
            bonus_token: int | None = None
            confirmed_prefix = list(context_ids)
            prompt_logprobs = getattr(output, "prompt_logprobs", None)
            candidate_start = len(context_ids)

            for offset, proposed_id in enumerate(draft.token_ids):
                if prompt_logprobs is None:
                    raise RuntimeError(
                        "vLLM did not return prompt_logprobs needed for rejection sampling "
                        "verification."
                    )
                target_entry = self._logprob_dict(prompt_logprobs[candidate_start + offset])
                if (not self.greedy_only) and int(proposed_id) not in target_entry:
                    raise RuntimeError(
                        "Target prompt_logprobs did not include the proposed token. "
                        "Increase target max_logprobs to cover the draft support."
                    )
                target_logprobs.append(target_entry)
                target_best_token = self._best_token_id(target_entry)
                target_greedy.append(target_best_token)

                if self.greedy_only:
                    accept_prob = 1.0 if int(proposed_id) == int(target_best_token) else 0.0
                    acceptance_probabilities.append(accept_prob)
                    if accept_prob < 1.0:
                        correction_token = int(target_best_token)
                        break
                else:
                    target_log_p = self._lookup_logprob(target_entry, int(proposed_id))
                    draft_log_q = draft.log_q_proposed[offset]
                    accept_prob = acceptance_probability(target_log_p, draft_log_q)
                    acceptance_probabilities.append(accept_prob)
                    if not should_accept(self.rng.random(), accept_prob):
                        correction_token = sample_correction_token(
                            target_entry,
                            draft.distribution_dict(offset, self.vocab_size),
                            self.rng.random(),
                        )
                        break

                accepted += 1
                confirmed_prefix.append(int(proposed_id))
                if self.eos_token_id is not None and int(proposed_id) == self.eos_token_id:
                    break

            committed = list(draft.token_ids[:accepted])
            reached_eos = self.eos_token_id is not None and self.eos_token_id in committed
            if len(committed) < remaining_tokens and not reached_eos:
                if correction_token is not None:
                    committed.append(correction_token)
                else:
                    if not output.outputs or not output.outputs[0].token_ids:
                        bonus_id = None
                    else:
                        bonus_id = int(output.outputs[0].token_ids[0])
                    if bonus_id is not None:
                        if self.greedy_only:
                            target_greedy.append(bonus_id)
                        else:
                            bonus_entry = self._logprob_dict(output.outputs[0].logprobs[0])
                            target_logprobs.append(bonus_entry)
                            target_greedy.append(greedy_correction_token(bonus_entry))
                        committed.append(bonus_id)
                        bonus_token = bonus_id

            if self.eos_token_id is not None and self.eos_token_id in committed:
                eos_index = committed.index(self.eos_token_id)
                committed = committed[: eos_index + 1]
                reached_eos = True

            committed = committed[:remaining_tokens]
            results.append(
                VerificationResult(
                    accepted_length=accepted,
                    committed_token_ids=committed,
                    committed_text=self.tokenizer.decode(committed, skip_special_tokens=True),
                    target_latency_ms=0.0,
                    target_logprobs=target_logprobs,
                    acceptance_probabilities=acceptance_probabilities,
                    target_greedy_token_ids=target_greedy,
                    reached_eos=reached_eos,
                    finish_reason="eos" if reached_eos else None,
                    correction_token_id=correction_token,
                    bonus_token_id=bonus_token,
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_request_ms = elapsed_ms / max(1, len(results))
        for result in results:
            result.target_latency_ms = per_request_ms

        return results
