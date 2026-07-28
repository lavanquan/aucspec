from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .models import DraftProposal


@dataclass
class DraftRequest:
    client_id: int
    context_token_ids: list[int]
    gamma: int
    speed_multiplier: float
    temperature: float = 0.0


@dataclass
class DraftKVState:
    """Persistent draft-side state for one virtual client.

    The cache always represents the *confirmed* prefix only. Proposal temporarily
    advances it; after target verification the cache is rolled back and then
    advanced with the actually committed target-approved tokens.
    """

    cache: DynamicCache
    confirmed_length: int
    next_logits: torch.Tensor


class DraftWorker:
    """One persistent Qwen draft model pinned to one GPU.

    Clients are affinitized to workers by the simulator. Each client receives an
    independent KV cache stored on that worker's GPU. Model weights are loaded once.
    """

    def __init__(self, worker_id: int, model_name: str, device: str) -> None:
        self.worker_id = worker_id
        self.device = device
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                token=token,
            ).to(device)
        except OSError as exc:
            message = str(exc)
            if "401" in message or "Unauthorized" in message or "expired" in message:
                raise OSError(
                    f"Hugging Face authentication failed while loading draft model "
                    f"{model_name!r}. The cached token may be expired. Run "
                    "`huggingface-cli login` or set a fresh HF_TOKEN."
                ) from exc
            raise OSError(
                f"Unable to load draft model {model_name!r}. If this is a gated or "
                "private Hugging Face repo, set HF_TOKEN or run `huggingface-cli login`."
            ) from exc
        self.model.eval()
        self.lock = asyncio.Lock()
        self.client_states: dict[int, DraftKVState] = {}

    async def reset_client(self, client_id: int) -> None:
        async with self.lock:
            self.client_states.pop(client_id, None)
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()

    async def generate(self, request: DraftRequest) -> DraftProposal:
        async with self.lock:
            return await asyncio.to_thread(self._generate_sync, request)

    async def commit(self, client_id: int, committed_token_ids: list[int]) -> None:
        """Synchronize the draft cache with target-approved tokens."""
        if not committed_token_ids:
            return
        async with self.lock:
            await asyncio.to_thread(self._commit_sync, client_id, committed_token_ids)

    def _initialize_state(self, request: DraftRequest) -> DraftKVState:
        if not request.context_token_ids:
            raise ValueError("Draft context cannot be empty")
        input_ids = torch.tensor(
            [request.context_token_ids], dtype=torch.long, device=self.device
        )
        cache = DynamicCache()
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        state = DraftKVState(
            cache=output.past_key_values,
            confirmed_length=len(request.context_token_ids),
            next_logits=output.logits[:, -1, :].detach(),
        )
        self.client_states[request.client_id] = state
        return state

    def _ensure_state(self, request: DraftRequest) -> DraftKVState:
        state = self.client_states.get(request.client_id)
        # A mismatch indicates a new sample/reset or an external state change.
        if state is None or state.confirmed_length != len(request.context_token_ids):
            self.client_states.pop(request.client_id, None)
            state = self._initialize_state(request)
        return state

    def _generate_sync(self, request: DraftRequest) -> DraftProposal:
        if request.gamma <= 0:
            return DraftProposal(request.client_id, [], [], "", 0.0, self.worker_id)

        start = time.perf_counter()
        state = self._ensure_state(request)
        base_length = state.confirmed_length
        proposed: list[int] = []
        draft_logprobs: list[torch.Tensor] = []
        logits = state.next_logits

        with torch.inference_mode():
            for _ in range(request.gamma):
                if request.temperature <= 0.0:
                    token_id = int(torch.argmax(logits, dim=-1).item())
                    logprobs = torch.full(
                        (logits.shape[-1],),
                        float("-inf"),
                        dtype=torch.float32,
                    )
                    logprobs[token_id] = 0.0
                else:
                    scaled_logits = logits / request.temperature
                    logprobs = torch.log_softmax(scaled_logits, dim=-1).squeeze(0).detach().cpu()
                    probs = torch.softmax(scaled_logits, dim=-1)
                    token_id = int(torch.multinomial(probs, num_samples=1).item())
                proposed.append(token_id)
                draft_logprobs.append(logprobs)
                token = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                output = self.model(
                    input_ids=token,
                    past_key_values=state.cache,
                    use_cache=True,
                    return_dict=True,
                )
                state.cache = output.past_key_values
                logits = output.logits[:, -1, :].detach()

        # Proposal is tentative. Roll the cache back to the confirmed prefix.
        state.cache.crop(base_length)
        state.confirmed_length = base_length
        # next_logits at the confirmed prefix must remain unchanged until commit.
        # We intentionally do not assign the tentative `logits` here.

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return DraftProposal(
            client_id=request.client_id,
            token_ids=proposed,
            draft_logprobs=draft_logprobs,
            text=self.tokenizer.decode(proposed, skip_special_tokens=True),
            latency_ms=elapsed_ms * request.speed_multiplier,
            worker_id=self.worker_id,
        )

    def _commit_sync(self, client_id: int, committed_token_ids: list[int]) -> None:
        state = self.client_states.get(client_id)
        if state is None:
            # The next generate call will initialize from the complete context.
            return
        input_ids = torch.tensor(
            [committed_token_ids], dtype=torch.long, device=self.device
        )
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                past_key_values=state.cache,
                use_cache=True,
                return_dict=True,
            )
        state.cache = output.past_key_values
        state.confirmed_length += len(committed_token_ids)
        state.next_logits = output.logits[:, -1, :].detach()

    def cache_stats(self) -> dict[str, int]:
        return {
            "worker_id": self.worker_id,
            "cached_clients": len(self.client_states),
            "cached_tokens": sum(s.confirmed_length for s in self.client_states.values()),
        }
