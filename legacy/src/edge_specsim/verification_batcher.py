from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

from .models import DraftResult, VerificationResult
from .roofline_budget import RooflineBudgetModel
from .virtual_causality import VirtualCoordinator
from .verification_queue import (
    VerificationBatchMetadata,
    VerificationQueue,
    VerificationRequest,
)

if TYPE_CHECKING:  # keeps the batcher importable without torch/vLLM (CPU tests)
    from .target_client import VLLMCandidateVerifier

# How the shared verifier clock is advanced after a batch.
#   amortized_per_request : V1 (legacy, reproduces earlier results). The clock
#                           advances by max(target_latency_ms) = wall time / B.
#   batch_wall            : V1-batchwall. The whole batch occupies the single
#                           shared verifier for its full measured wall time.
SERVICE_TIME_MODELS = ("amortized_per_request", "batch_wall")


class VerificationBatcher:
    """Forms a real vLLM batch of context+candidate sequences."""

    def __init__(
        self,
        target: VLLMCandidateVerifier,
        max_batch_size: int = 16,
        batch_wait_ms: float = 3.0,
        verify_token_budget: int = 32,
        scheduler_name: str = "fcfs",
        utility_lambda: float = 0.0,
        scheduler_seed: int = 0,
        roofline_budget_model: RooflineBudgetModel | None = None,
        target_replication_factor: int = 1,
        service_time_model: str = "amortized_per_request",
        coordinator: VirtualCoordinator | None = None,
        causal: bool = False,
        on_batch_complete: Callable[[VerificationBatchMetadata], None] | None = None,
    ) -> None:
        if causal and coordinator is None:
            raise ValueError("causal batching requires a VirtualCoordinator")
        if causal and roofline_budget_model is not None:
            raise ValueError("causal virtual time is not supported with roofline_knee budgets (D3: the budget "
                             "signal reads real-order queue state)")
        self.coordinator = coordinator
        self.causal = bool(causal)
        self.on_batch_complete = on_batch_complete
        self._request_counter: dict[int, int] = {}
        if service_time_model not in SERVICE_TIME_MODELS:
            raise ValueError(
                f"unknown verifier service_time_model {service_time_model!r}; "
                f"expected one of {SERVICE_TIME_MODELS}"
            )
        self.service_time_model = service_time_model
        self.target = target
        self.max_batch_size = max_batch_size
        self.batch_wait_ms = batch_wait_ms
        self.verify_token_budget = max(1, verify_token_budget)
        self.scheduler_name = scheduler_name
        self.utility_lambda = utility_lambda
        self.roofline_budget_model = roofline_budget_model
        self.target_replication_factor = max(1, int(target_replication_factor))
        self.queue: asyncio.Queue[VerificationRequest | None] = asyncio.Queue()
        self.server_queue = VerificationQueue(random_seed=scheduler_seed)
        self.task: asyncio.Task[None] | None = None
        self.batch_counter = 0
        self.server_available_at_ms = 0.0

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
        client_id: int,
        context_token_ids: list[int],
        draft: DraftResult,
        remaining_tokens: int,
        server_arrival_ms: float,
        weight: float = 1.0,
        deadline: float | None = None,
        expected_acceptance_rate: float = 0.5,
        expected_acceptance_profile: list[float] | None = None,
        server_queue_price: float = 0.0,
        theta_f_ms_per_token: float = 0.0,
    ) -> tuple[VerificationBatchMetadata, VerificationResult]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[VerificationBatchMetadata, VerificationResult]] = (
            loop.create_future()
        )
        request_id = self._request_counter.get(client_id, 0)
        self._request_counter[client_id] = request_id + 1
        if self.coordinator is not None:
            self.coordinator.note_submit(client_id, request_id, server_arrival_ms)
        await self.queue.put(
            VerificationRequest(
                client_id=client_id,
                arrival_time=server_arrival_ms,
                gamma=len(draft.token_ids),
                candidate_token_ids=list(draft.token_ids),
                weight=weight,
                deadline=deadline,
                expected_acceptance_rate=expected_acceptance_rate,
                expected_acceptance_profile=(
                    list(expected_acceptance_profile)
                    if expected_acceptance_profile is not None
                    else []
                ),
                context_token_ids=context_token_ids,
                draft=draft,
                remaining_tokens=remaining_tokens,
                future=future,
                # CAPACITY_AWARE_FRAMEWORK_CODEX_IMPLEMENTATION.md Section 6:
                # only consumed by the capacity_knapsack value_fn; harmless
                # (default 0.0) for every legacy scheduler.
                server_queue_price=server_queue_price,
                theta_f_ms_per_token=theta_f_ms_per_token,
                request_id=request_id,
            )
        )
        return await future

    def estimate_verify_token_budget(
        self,
        context_length: int,
        incoming_requests: int = 1,
    ) -> int:
        return self.estimate_verify_budget_signal(
            context_length=context_length,
            incoming_requests=incoming_requests,
        )[0]

    def estimate_verify_budget_signal(
        self,
        context_length: int,
        incoming_requests: int = 1,
    ) -> tuple[int, str, float]:
        if self.roofline_budget_model is None:
            return self.verify_token_budget, "memory_bound", 0.0
        eligible = self.server_queue.eligible_requests(self.batch_wait_ms)
        profiled_batch_size = min(
            self.max_batch_size,
            max(1, len(eligible) + max(0, int(incoming_requests))),
        )
        queued_verifier_tokens = sum(
            request.verifier_token_cost for request in eligible
        ) + max(0, int(incoming_requests))
        signal = self.roofline_budget_model.signal_for(
            batch_size=profiled_batch_size,
            context_length=max(1, int(context_length)),
            queued_verifier_tokens=queued_verifier_tokens,
        )
        return (
            max(
                1,
                min(
                    self.verify_token_budget * self.target_replication_factor,
                    signal.budget_tokens * self.target_replication_factor,
                ),
            ),
            signal.regime,
            float(signal.theta_f_ms_per_token) / float(self.target_replication_factor),
        )

    def _head_cutoff(self) -> float | None:
        """Cutoff of the current head, after absorbing everything already submitted."""
        self._absorb_submitted()
        if not self.server_queue:
            return None
        return self.server_queue.waiting[0].arrival_time + self.batch_wait_ms

    def _absorb_submitted(self) -> bool:
        """Move submitted-but-undrained requests into the server queue; True if the close sentinel was seen."""
        closed = False
        while True:
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return closed
            if item is None:
                self._close_seen = True
                closed = True
                continue
            self.server_queue.push(item)

    async def _loop_causal(self) -> None:
        """D1: batches are formed only when no request with arrival <= cutoff can still appear."""
        coord = self.coordinator
        assert coord is not None
        coord._head_cutoff_fn = self._head_cutoff
        self._close_seen = False

        def ready() -> bool:
            cutoff = self._head_cutoff()
            if cutoff is None:
                return False
            return coord.cutoff_satisfied(cutoff)

        while True:
            self._absorb_submitted()
            if not self.server_queue:
                if self._close_seen:
                    return
                item = await self.queue.get()
                if item is None:
                    self._close_seen = True
                else:
                    self.server_queue.push(item)
                continue
            if not self._close_seen:
                await coord.wait_for_cutoff(ready)
            effective_budget = self._effective_verify_token_budget()
            batch = self.server_queue.pop_batch(
                scheduler_name=self.scheduler_name,
                max_batch_size=self.max_batch_size,
                batch_wait_ms=self.batch_wait_ms,
                verify_token_budget=effective_budget,
                utility_lambda=self.utility_lambda,
            )
            await self._process(batch, effective_budget)

    async def _loop(self) -> None:
        if self.causal:
            await self._loop_causal()
            return
        while True:
            item = await self.queue.get()
            if item is None:
                while self.server_queue:
                    effective_budget = self._effective_verify_token_budget()
                    await self._process(
                        self.server_queue.pop_batch(
                            scheduler_name=self.scheduler_name,
                            max_batch_size=self.max_batch_size,
                            batch_wait_ms=self.batch_wait_ms,
                            verify_token_budget=effective_budget,
                            utility_lambda=self.utility_lambda,
                        ),
                        effective_budget,
                    )
                return
            self.server_queue.push(item)
            await self._drain_queue()
            while self.server_queue:
                effective_budget = self._effective_verify_token_budget()
                batch = self.server_queue.pop_batch(
                    scheduler_name=self.scheduler_name,
                    max_batch_size=self.max_batch_size,
                    batch_wait_ms=self.batch_wait_ms,
                    verify_token_budget=effective_budget,
                    utility_lambda=self.utility_lambda,
                )
                cutoff_ms = batch[0].arrival_time + self.batch_wait_ms
                if len(batch) < self.max_batch_size and not self.server_queue.has_request_past_cutoff(
                    cutoff_ms
                ):
                    self.server_queue.extend(batch)
                    await asyncio.sleep(0)
                    await self._drain_queue()
                    effective_budget = self._effective_verify_token_budget()
                    batch = self.server_queue.pop_batch(
                        scheduler_name=self.scheduler_name,
                        max_batch_size=self.max_batch_size,
                        batch_wait_ms=self.batch_wait_ms,
                        verify_token_budget=effective_budget,
                        utility_lambda=self.utility_lambda,
                    )
                await self._process(batch, effective_budget)

    async def _drain_queue(self) -> None:
        while True:
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if item is None:
                await self.queue.put(None)
                return
            self.server_queue.push(item)

    def _effective_verify_token_budget(self) -> int:
        if self.roofline_budget_model is None:
            return self.verify_token_budget
        eligible = self.server_queue.eligible_requests(self.batch_wait_ms)
        if not eligible:
            return self.verify_token_budget
        profiled_batch_size = min(self.max_batch_size, len(eligible))
        profiled_context_length = max(
            1,
            max(len(request.context_token_ids) for request in eligible),
        )
        dynamic_budget = self.roofline_budget_model.budget_for(
            batch_size=profiled_batch_size,
            context_length=profiled_context_length,
        )
        return max(
            1,
            min(
                self.verify_token_budget * self.target_replication_factor,
                dynamic_budget * self.target_replication_factor,
            ),
        )

    @staticmethod
    def _member_id(request: VerificationRequest) -> int:
        return getattr(request, "request_id", 0)

    async def _process(
        self,
        batch: list[VerificationRequest],
        effective_budget: int,
    ) -> None:
        batch_ready_ms = batch[0].arrival_time + self.batch_wait_ms
        batch_token_cost = sum(request.verifier_token_cost for request in batch)
        # CAPACITY_AWARE_FRAMEWORK_CODEX_IMPLEMENTATION.md Section 6.3: this
        # reads the flag set by the pop_batch() call that produced `batch`
        # immediately before this _process() call (synchronous, no
        # interleaving pop_batch between them -- see _loop()'s call sites).
        forced_service = self.server_queue.last_batch_forced_service
        sum_capacity_value = sum(request.capacity_dpp_value for request in batch)
        context_lengths = [len(request.context_token_ids) for request in batch]
        batch_context_tokens_sum = sum(context_lengths)
        batch_context_tokens_max = max(context_lengths, default=0)
        # Tokens the target actually processes: stateless generate over
        # context + draft tokens for every request (logical cost is gamma+1).
        batch_processed_tokens = batch_context_tokens_sum + sum(
            len(request.candidate_token_ids) for request in batch
        )
        batch_metadata = VerificationBatchMetadata(
            batch_id=self.batch_counter,
            batch_size=len(batch),
            batch_token_cost=batch_token_cost,
            batch_token_budget=effective_budget,
            scheduler_name=self.scheduler_name,
            measured_batch_service_ms=0.0,
            modeled_batch_service_ms=0.0,
            compute_bound_batch_service_ms=0.0,
            theta_f_ms_per_token=0.0,
            batch_ready_ms=batch_ready_ms,
            batch_start_ms=max(batch_ready_ms, self.server_available_at_ms),
            verify_finish_ms=0.0,
            forced_service=forced_service,
            sum_capacity_value=sum_capacity_value,
            service_time_model=self.service_time_model,
            batch_context_tokens_sum=batch_context_tokens_sum,
            batch_context_tokens_max=batch_context_tokens_max,
            batch_processed_tokens=batch_processed_tokens,
        )
        self.batch_counter += 1
        members = [(request.client_id, self._member_id(request)) for request in batch]
        if self.coordinator is not None:
            self.coordinator.batch_formed(
                batch_metadata.batch_id, batch[0].arrival_time + self.batch_wait_ms, members,
                batch_metadata.batch_start_ms)
        payload = [
            (req.context_token_ids, req.draft, req.remaining_tokens)
            for req in batch
        ]
        batch_service_start = asyncio.get_running_loop().time()
        try:
            results = await asyncio.to_thread(self.target.verify_batch_sync, payload)
        except BaseException as exc:
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(exc)
            return
        measured_batch_service_ms = (
            asyncio.get_running_loop().time() - batch_service_start
        ) * 1000.0

        # Full measured wall time of the batch. A verifier that does not stamp
        # it (e.g. a bare mock) falls back to the event-loop measurement.
        batch_wall_ms = max(
            (result.target_batch_wall_ms for result in results),
            default=0.0,
        )
        if batch_wall_ms <= 0.0:
            batch_wall_ms = measured_batch_service_ms
        amortized_request_ms = batch_wall_ms / max(1, len(batch))

        if self.roofline_budget_model is None:
            if self.service_time_model == "batch_wall":
                modeled_batch_service_ms = batch_wall_ms
            else:
                modeled_batch_service_ms = max(
                    (result.target_latency_ms for result in results),
                    default=0.0,
                )
            compute_bound_batch_service_ms = modeled_batch_service_ms
            theta_f_ms_per_token = (
                compute_bound_batch_service_ms / max(1, batch_token_cost)
                if batch_token_cost > 0
                else 0.0
            )
        else:
            profiled_context_length = max(
                1,
                max(len(request.context_token_ids) for request in batch),
            )
            signal = self.roofline_budget_model.signal_for(
                batch_size=len(batch),
                context_length=profiled_context_length,
                queued_verifier_tokens=batch_token_cost,
            )
            modeled_batch_service_ms = signal.service_time_ms(batch_token_cost)
            modeled_batch_service_ms /= float(self.target_replication_factor)
            theta_f_ms_per_token = (
                float(signal.theta_f_ms_per_token) / float(self.target_replication_factor)
            )
            compute_bound_batch_service_ms = theta_f_ms_per_token * batch_token_cost
        batch_metadata = VerificationBatchMetadata(
            batch_id=batch_metadata.batch_id,
            batch_size=batch_metadata.batch_size,
            batch_token_cost=batch_metadata.batch_token_cost,
            batch_token_budget=effective_budget,
            scheduler_name=batch_metadata.scheduler_name,
            measured_batch_service_ms=measured_batch_service_ms,
            modeled_batch_service_ms=modeled_batch_service_ms,
            compute_bound_batch_service_ms=compute_bound_batch_service_ms,
            theta_f_ms_per_token=theta_f_ms_per_token,
            batch_ready_ms=batch_metadata.batch_ready_ms,
            batch_start_ms=batch_metadata.batch_start_ms,
            verify_finish_ms=batch_metadata.batch_start_ms + modeled_batch_service_ms,
            forced_service=batch_metadata.forced_service,
            sum_capacity_value=batch_metadata.sum_capacity_value,
            service_time_model=self.service_time_model,
            batch_wall_ms=batch_wall_ms,
            amortized_request_ms=amortized_request_ms,
            batch_context_tokens_sum=batch_context_tokens_sum,
            batch_context_tokens_max=batch_context_tokens_max,
            batch_processed_tokens=batch_processed_tokens,
        )
        self.server_available_at_ms = batch_metadata.verify_finish_ms
        if self.coordinator is not None:
            self.coordinator.batch_done(
                batch_metadata.batch_id, members, batch_metadata.batch_start_ms, batch_metadata.verify_finish_ms)
        if self.on_batch_complete is not None:
            # D2: the verifier publishes its state to the controller in batch order, before any client wakes.
            self.on_batch_complete(batch_metadata)
        for request, result in zip(batch, results):
            request.future.set_result((batch_metadata, result))
