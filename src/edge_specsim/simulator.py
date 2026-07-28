from __future__ import annotations

import asyncio
import csv
import random
import time
from pathlib import Path

import yaml

from .controller import OnlineController
from .dataset import load_samples, shard_round_robin
from .draft_worker import DraftRequest, DraftWorker
from .models import ClientProfile, DraftResult
from .target_client import VLLMCandidateVerifier
from .verification_batcher import VerificationBatcher


class EdgeSpecSimulator:
    def __init__(
        self,
        config_path: str,
        dataset_name: str | None = None,
        num_questions: int | None = None,
        split: str | None = None,
        math_subject: str | None = None,
        target_model: str | None = None,
        draft_model: str | None = None,
        num_clients: int | None = None,
        detailed_log: bool = False,
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as handle:
            self.cfg = yaml.safe_load(handle)

        dataset_cfg = self.cfg.setdefault("dataset", {})
        if dataset_name is not None:
            dataset_cfg["name"] = dataset_name
        if num_questions is not None:
            dataset_cfg["num_questions"] = num_questions
        if split is not None:
            dataset_cfg["split"] = split
        if math_subject is not None:
            dataset_cfg["math_subject"] = math_subject

        random.seed(self.cfg["simulation"]["seed"])
        sim = self.cfg["simulation"]
        self.sampling_temperature = float(sim.get("temperature", 0.0))
        model_cfg = self.cfg["models"]
        if target_model is not None:
            model_cfg["target"] = target_model
        if draft_model is not None:
            model_cfg["draft"] = draft_model
        # Initialize the target first so vLLM reserves GPU 0.
        self.target = VLLMCandidateVerifier(
            model_name=model_cfg["target"],
            gpu_memory_utilization=float(model_cfg.get("target_gpu_memory_utilization", 0.88)),
            max_model_len=int(model_cfg.get("target_max_model_len", 8192)),
            dtype=str(model_cfg.get("target_dtype", "auto")),
            enforce_eager=bool(model_cfg.get("target_enforce_eager", False)),
            enable_prefix_caching=bool(model_cfg.get("target_enable_prefix_caching", True)),
            sampling_temperature=self.sampling_temperature,
            seed=int(sim["seed"]),
        )
        self.workers = [
            DraftWorker(i, model_cfg["draft"], device)
            for i, device in enumerate(model_cfg["draft_devices"])
        ]
        batching = self.cfg.get("verification_batching", {})
        self.verifier = VerificationBatcher(
            self.target,
            max_batch_size=int(batching.get("max_batch_size", 16)),
            batch_wait_ms=float(batching.get("batch_wait_ms", 3.0)),
        )
        self.controller = OnlineController(
            gamma_max=sim["gamma_max"],
            v=sim["controller_v"],
            min_tps=sim["min_interactivity_tps"],
            server_price_scale=sim["server_price_scale"],
        )
        self.records: list[dict[str, object]] = []
        self.detailed_log = detailed_log
        self.total_rounds = 0
        samples = load_samples(self.cfg["dataset"])
        self.clients = self._make_clients(samples, num_clients=num_clients)

    def _make_clients(self, samples, num_clients: int | None = None) -> list[ClientProfile]:
        c = self.cfg["clients"]
        configured_num_clients = int(self.cfg["simulation"]["num_clients"])
        num_clients = int(num_clients) if num_clients is not None else configured_num_clients
        if num_clients <= 0:
            raise ValueError("simulation.num_clients must be greater than zero")
        shards = shard_round_robin(samples, num_clients)
        clients: list[ClientProfile] = []
        for i in range(num_clients):
            clients.append(
                ClientProfile(
                    client_id=i,
                    prompt_queue=shards[i],
                    rtt_ms=random.uniform(*c["rtt_ms_range"]),
                    uplink_mbps=random.uniform(*c["uplink_mbps_range"]),
                    draft_speed_multiplier=random.uniform(*c["draft_speed_multiplier_range"]),
                )
            )
        return clients

    async def run(self) -> None:
        await self.verifier.start()
        try:
            await asyncio.gather(*(self._run_client(client) for client in self.clients))
        finally:
            await self.verifier.close()
        self._write_results()

    async def _run_client(self, client: ClientProfile) -> None:
        while client.start_next_sample():
            assert client.current_sample is not None
            worker = self.workers[client.client_id % len(self.workers)]
            await worker.reset_client(client.client_id)
            sample_round = 0
            while client.remaining_tokens > 0:
                reached_eos = await self._run_round(client, sample_round)
                sample_round += 1
                if reached_eos:
                    break
            client.samples_completed += 1
            print(
                f"client={client.client_id:02d} completed sample={client.current_sample.sample_id} "
                f"tokens={client.generated_for_sample} rounds={sample_round}"
            )

    async def _run_round(self, client: ClientProfile, round_id: int) -> bool:
        assert client.current_sample is not None
        self.total_rounds += 1
        remaining = client.remaining_tokens
        selected_gamma = self.controller.choose_gamma(client, self.total_rounds)
        gamma = min(selected_gamma, max(0, remaining - 1))
        client.gamma = gamma
        context_token_ids = self.target.build_context_token_ids(
            client.current_sample.prompt, client.history_token_ids
        )
        round_start = time.perf_counter()

        if gamma > 0:
            worker = self.workers[client.client_id % len(self.workers)]
            draft = await worker.generate(
                DraftRequest(
                    client_id=client.client_id,
                    context_token_ids=context_token_ids,
                    gamma=gamma,
                    speed_multiplier=client.draft_speed_multiplier,
                    temperature=self.sampling_temperature,
                )
            )
        else:
            draft = DraftResult(client.client_id, [], [], "", 0.0, -1)

        upload_ms = 0.0
        if gamma > 0:
            payload_bits = (gamma * 4 + 64) * 8
            upload_ms = payload_bits / (client.uplink_mbps * 1_000_000) * 1000.0

        await asyncio.sleep((client.rtt_ms / 2.0 + upload_ms) / 1000.0)
        batch_id, verification_batch_size, verification = await self.verifier.submit(
            context_token_ids, draft, remaining
        )
        await asyncio.sleep(client.rtt_ms / 2000.0)

        committed = verification.committed_token_ids
        if draft.worker_id >= 0:
            await self.workers[draft.worker_id].commit(client.client_id, committed)
        client.history_token_ids.extend(committed)
        useful = len(committed)
        client.generated_for_sample += useful
        if gamma > 0:
            client.alpha_successes += verification.accepted_length
            if verification.accepted_length < gamma:
                client.alpha_failures += 1
        client.accepted_tokens += verification.accepted_length
        client.useful_tokens += useful
        client.rounds_completed += 1

        round_latency_ms = (time.perf_counter() - round_start) * 1000.0
        self.controller.update(
            client,
            useful_tokens=useful,
            round_latency_ms=round_latency_ms,
            draft_latency_ms=draft.latency_ms,
            target_latency_ms=verification.target_latency_ms,
        )

        record: dict[str, object] = {
            "client_id": client.client_id,
            "sample_id": client.current_sample.sample_id,
            "dataset_name": client.current_sample.dataset_name,
            "prompt": client.current_sample.prompt,
            "round_id": round_id,
            "verification_batch_id": batch_id,
            "verification_batch_size": verification_batch_size,
            "worker_id": draft.worker_id,
            "gamma": gamma,
            "accepted_length": verification.accepted_length,
            "useful_tokens": useful,
            "generated_for_sample": client.generated_for_sample,
            "max_new_tokens": client.current_sample.max_new_tokens,
            "reached_eos": verification.reached_eos,
            "finish_reason": verification.finish_reason,
            "alpha_hat": client.alpha_hat,
            "z_queue": client.z_queue,
            "server_queue": self.controller.server_queue,
            "rtt_ms": client.rtt_ms,
            "upload_ms": upload_ms,
            "draft_latency_ms": draft.latency_ms,
            "target_latency_ms": verification.target_latency_ms,
            "round_latency_ms": round_latency_ms,
            "committed_text": verification.committed_text,
        }
        if self.detailed_log:
            record.update(
                {
                    "history_token_ids": list(client.history_token_ids),
                    "context_token_ids": list(context_token_ids),
                    "proposed_token_ids": list(draft.token_ids),
                    "proposed_text": draft.text,
                    "committed_token_ids": list(committed),
                    "target_greedy_token_ids": verification.target_greedy_token_ids,
                    "target_greedy_text": self.target.tokenizer.decode(
                        verification.target_greedy_token_ids,
                        skip_special_tokens=True,
                    ),
                    "draft_cached_clients": (
                        self.workers[draft.worker_id].cache_stats()["cached_clients"]
                        if draft.worker_id >= 0
                        else 0
                    ),
                    "draft_cached_tokens": (
                        self.workers[draft.worker_id].cache_stats()["cached_tokens"]
                        if draft.worker_id >= 0
                        else 0
                    ),
                }
            )
        self.records.append(record)
        if self.detailed_log:
            print(
                f"client={client.client_id:02d} sample={client.current_sample.sample_id} "
                f"round={round_id} batch={batch_id}/{verification_batch_size} gamma={gamma} "
                f"accepted={verification.accepted_length} useful={useful} "
                f"proposed={draft.token_ids} committed={committed} "
                f"progress={client.generated_for_sample}/{client.current_sample.max_new_tokens}"
            )
        else:
            print(
                f"client={client.client_id:02d} sample={client.current_sample.sample_id} "
                f"round={round_id} gamma={gamma} accepted={verification.accepted_length} "
                f"useful={useful} progress={client.generated_for_sample}/{client.current_sample.max_new_tokens}"
            )
        return verification.reached_eos or useful == 0

    def _write_results(self) -> None:
        output = Path(self.cfg["simulation"]["output_csv"])
        output.parent.mkdir(parents=True, exist_ok=True)
        if not self.records:
            return
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.records[0].keys()))
            writer.writeheader()
            writer.writerows(self.records)
        print(f"Wrote {len(self.records)} records to {output}")
