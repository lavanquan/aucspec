from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .models import DraftResult, VerificationResult
from .target_client import VLLMCandidateVerifier


@dataclass
class VerificationRequest:
    context_token_ids: list[int]
    draft: DraftResult
    remaining_tokens: int
    future: asyncio.Future[tuple[int, int, VerificationResult]]


class VerificationBatcher:
    """Forms a real vLLM batch of context+candidate sequences."""

    def __init__(
        self,
        target: VLLMCandidateVerifier,
        max_batch_size: int = 16,
        batch_wait_ms: float = 3.0,
    ) -> None:
        self.target = target
        self.max_batch_size = max_batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.queue: asyncio.Queue[VerificationRequest | None] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None
        self.batch_counter = 0

    async def start(self) -> None:
        if self.task is None:
            self.task = asyncio.create_task(self._loop())

    async def close(self) -> None:
        await self.queue.put(None)
        if self.task is not None:
            await self.task
            self.task = None

    async def submit(
        self,
        context_token_ids: list[int],
        draft: DraftResult,
        remaining_tokens: int,
    ) -> tuple[int, int, VerificationResult]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[int, int, VerificationResult]] = loop.create_future()
        await self.queue.put(
            VerificationRequest(context_token_ids, draft, remaining_tokens, future)
        )
        return await future

    async def _loop(self) -> None:
        while True:
            first = await self.queue.get()
            if first is None:
                return
            batch = [first]
            deadline = asyncio.get_running_loop().time() + self.batch_wait_s
            while len(batch) < self.max_batch_size:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    await self._process(batch)
                    return
                batch.append(item)
            await self._process(batch)

    async def _process(self, batch: list[VerificationRequest]) -> None:
        batch_id = self.batch_counter
        self.batch_counter += 1
        payload = [
            (req.context_token_ids, req.draft, req.remaining_tokens)
            for req in batch
        ]
        try:
            results = await asyncio.to_thread(self.target.verify_batch_sync, payload)
        except BaseException as exc:
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(exc)
            return

        batch_size = len(batch)
        for request, result in zip(batch, results):
            request.future.set_result((batch_id, batch_size, result))
