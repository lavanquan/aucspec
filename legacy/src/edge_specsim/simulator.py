from __future__ import annotations

import asyncio
import csv
import json
import math
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd
import yaml

from .controller import OnlineController
from .dataset import load_samples, shard_round_robin
from .draft_worker import DraftRequest, DraftWorker
from .draft_timing import (
    draft_queue_wait_ms,
    normalize_draft_execution_mode,
    resolve_effective_draft_latency_ms,
)
from .channel import current_snr_db, spectral_efficiency_from_snr_db
from .models import ClientProfile, DraftResult
from .network import (
    estimate_downlink_bytes,
    estimate_downlink_bytes_from_token_count,
    estimate_uplink_bytes,
    estimate_uplink_bytes_from_gamma,
    paper_square_root_allocation_weight,
)
from .metrics import summarize_round_csv
from .roofline_budget import RooflineBudgetModel
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
        self.config_path = str(config_path)
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
        controller_cfg = self.cfg.setdefault("controller", {})
        self.learning_cfg = controller_cfg.setdefault("learning", {})
        control_cfg = self.cfg.setdefault("control", {})
        experiment_cfg = self.cfg.setdefault("experiment", {})
        gamma_choices = controller_cfg.get("gamma_choices")
        if gamma_choices is None:
            gamma_max = int(sim.get("gamma_max", 0))
            gamma_choices = list(range(gamma_max + 1))
            controller_cfg["gamma_choices"] = gamma_choices
        gamma_choices = sorted({int(choice) for choice in gamma_choices})
        if not gamma_choices:
            raise ValueError("controller.gamma_choices must not be empty")
        if gamma_choices[0] < 0:
            raise ValueError("controller.gamma_choices cannot contain negative values")
        if 0 not in gamma_choices:
            raise ValueError("controller.gamma_choices must include 0 for target-only mode")
        decoding_cfg = self.cfg.setdefault("decoding", {})
        decoding_mode = str(decoding_cfg.get("mode", "greedy")).lower()
        decoding_temperature = float(
            decoding_cfg.get("temperature", sim.get("temperature", 0.0))
        )
        if decoding_mode not in {"greedy", "sampling"}:
            raise ValueError(
                "decoding.mode must be one of: greedy, sampling"
            )
        if decoding_mode == "greedy" and decoding_temperature != 0.0:
            raise ValueError(
                "Greedy decoding requires "
                "`decoding.temperature: 0.0`."
            )
        if decoding_mode == "sampling" and decoding_temperature <= 0.0:
            raise ValueError(
                "Sampling decoding requires `decoding.temperature > 0.0`."
            )
        self.sampling_temperature = decoding_temperature
        model_cfg = self.cfg["models"]
        if target_model is not None:
            model_cfg["target"] = target_model
        if draft_model is not None:
            model_cfg["draft"] = draft_model
        draft_devices = list(model_cfg.get("draft_devices", []))
        if not draft_devices:
            raise ValueError("models.draft_devices must contain at least one draft device")
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
            DraftWorker(
                i,
                model_cfg["draft"],
                device,
                enable_kv_cache=bool(model_cfg.get("draft_enable_kv_cache", True)),
            )
            for i, device in enumerate(draft_devices)
        ]
        batching = self.cfg.get("verification_batching", {})
        self.batch_wait_ms = float(batching.get("batch_wait_ms", 3.0))
        batch_scheduler = str(batching.get("scheduler", "fcfs")).lower()
        batch_utility_lambda = float(batching.get("utility_lambda", 0.0))
        target_replication_factor = int(batching.get("target_replication_factor", 1))
        budget_mode = str(batching.get("budget_mode", "fixed")).lower()
        verify_token_budget = int(
            batching.get(
                "verify_token_budget",
                int(batching.get("max_batch_size", 16)) * (max(gamma_choices) + 1),
            )
        )
        roofline_budget_model = None
        if budget_mode == "roofline_knee":
            roofline_summary_path = batching.get(
                "roofline_summary_path",
                "results/target_profile_summary.json",
            )
            roofline_budget_model = RooflineBudgetModel.from_summary_json(roofline_summary_path)
        self.verifier = VerificationBatcher(
            self.target,
            max_batch_size=int(batching.get("max_batch_size", 16)),
            batch_wait_ms=self.batch_wait_ms,
            verify_token_budget=verify_token_budget,
            scheduler_name=batch_scheduler,
            utility_lambda=batch_utility_lambda,
            scheduler_seed=int(sim["seed"]),
            roofline_budget_model=roofline_budget_model,
            target_replication_factor=target_replication_factor,
        )
        proposal_distribution_payload = (
            "delta_proposal" if self.sampling_temperature <= 0.0 else "full_vocab_logprobs"
        )
        self.controller = OnlineController(
            gamma_choices=gamma_choices,
            V=float(controller_cfg.get("V", controller_cfg.get("v", sim["controller_v"]))),
            min_tps=float(controller_cfg.get("min_interactivity_tps", sim["min_interactivity_tps"])),
            server_price_scale=float(
                controller_cfg.get("server_price_scale", sim["server_price_scale"])
            ),
            verifier_price_lambda=float(controller_cfg.get("verifier_price_lambda", 1.0)),
            draft_latency_weight=float(controller_cfg.get("draft_latency_weight", 0.0)),
            uplink_latency_weight=float(controller_cfg.get("uplink_latency_weight", 0.0)),
            policy=str(controller_cfg.get("policy", "adaptive_ucb")).lower(),
            fixed_gamma=int(controller_cfg.get("fixed_gamma", 0)),
            seed=int(sim["seed"]),
            slot_ms=float(control_cfg.get("slot_ms", 100.0)),
            min_exploration_rounds=int(controller_cfg.get("min_exploration_rounds", 0)),
            exploration_gamma=int(controller_cfg.get("exploration_gamma", 1)),
            exploration_epsilon=float(controller_cfg.get("exploration_epsilon", 0.0)),
            use_virtual_queues=bool(controller_cfg.get("use_virtual_queues", True)),
            proposal_distribution_payload=proposal_distribution_payload,
            proposal_vocab_size=self.target.vocab_size,
        )
        self.records: list[dict[str, object]] = []
        self.detailed_log = detailed_log
        self.total_rounds = 0
        self.wall_clock_start = time.perf_counter()
        self.virtual_system_time_ms = 0.0
        self.virtual_target_available_at_ms = 0.0
        self.draft_execution_mode = normalize_draft_execution_mode(
            sim.get("draft_execution_mode", "simulation")
        )
        self.warmup_seconds = max(0.0, float(experiment_cfg.get("warmup_seconds", 0.0)))
        measurement_seconds = experiment_cfg.get("measurement_seconds")
        self.measurement_seconds = (
            None if measurement_seconds is None else max(0.0, float(measurement_seconds))
        )
        self.measurement_start_ms = self.warmup_seconds * 1000.0
        self.measurement_end_ms = (
            None
            if self.measurement_seconds is None
            else self.measurement_start_ms + self.measurement_seconds * 1000.0
        )
        # CAPACITY_AUC_NSTAR_IMPLEMENTATION.md Section 4: nested-population
        # mode for the N* capacity search. When enabled, every candidate
        # concurrency N is a prefix of the SAME frozen catalog: client i
        # always gets prompt block [i*Q : (i+1)*Q] (not a round-robin shard
        # that shifts with N), and per-client params are already
        # index-deterministic because the RNG is seeded once and consumed
        # in client-index order. We therefore load population_n_max * Q
        # prompts regardless of the requested num_clients.
        self._nested_population = bool(sim.get("nested_population", False))
        self._nested_qpc = int(sim.get("questions_per_client", 5))
        self._nested_n_max = int(sim.get("population_n_max", sim["num_clients"]))
        self.population_fingerprint = ""
        if self._nested_population:
            self.cfg["dataset"] = dict(self.cfg["dataset"])
            self.cfg["dataset"]["num_questions"] = self._nested_n_max * self._nested_qpc
        samples = load_samples(self.cfg["dataset"])
        self.samples = list(samples)
        self.clients = self._make_clients(self.samples, num_clients=num_clients)

    @staticmethod
    def _prefix_hash(token_ids: list[int]) -> int:
        return hash(tuple(token_ids))

    @staticmethod
    def _sample_range(cfg: dict, key: str, default: list[float]) -> float:
        values = cfg.get(key, default)
        return random.uniform(*values)

    def _sample_client_class(self, client_cfg: dict) -> tuple[str, dict]:
        class_cfgs = client_cfg.get("client_classes")
        if not class_cfgs:
            return "default", {}
        names = list(class_cfgs.keys())
        probabilities = [float(class_cfgs[name].get("probability", 0.0)) for name in names]
        total_probability = sum(probabilities)
        if total_probability <= 0.0:
            raise ValueError("clients.client_classes probabilities must sum to a positive value")
        normalized = [probability / total_probability for probability in probabilities]
        selected_name = random.choices(names, weights=normalized, k=1)[0]
        return selected_name, class_cfgs[selected_name]

    @staticmethod
    def _transfer_time_ms(payload_bytes: float, rate_mbps: float, packet_loss: float) -> float:
        effective_bandwidth = max(1e-6, rate_mbps) * max(1e-6, 1.0 - packet_loss)
        payload_bits = payload_bytes * 8.0
        return payload_bits / (effective_bandwidth * 1_000_000) * 1000.0

    def _active_clients(self) -> list[ClientProfile]:
        return [
            client
            for client in self.clients
            if client.current_sample is not None and client.remaining_tokens > 0
        ]

    def _allocate_shared_bandwidth(
        self,
        client: ClientProfile,
        direction: str,
        time_ms: float,
        payload_bytes: int,
    ) -> tuple[float, float, float, float, str]:
        wireless_cfg = self.cfg["clients"].get("shared_wireless", {})
        active_clients = self._active_clients()
        if not active_clients:
            active_clients = [client]
        baseline = str(
            wireless_cfg.get(
                f"{direction}_allocator",
                wireless_cfg.get("allocator", "proportional"),
            )
        ).lower()

        total_bandwidth_mhz = float(
            wireless_cfg.get(
                f"total_{direction}_bandwidth_mhz",
                wireless_cfg.get("total_bandwidth_mhz", 20.0),
            )
        )
        raw_scores: dict[int, float] = {}
        for active_client in active_clients:
            demand_bytes = self._estimate_active_payload_bytes(
                active_client,
                direction,
                current_client=client,
                current_payload_bytes=payload_bytes,
            )
            raw_scores[active_client.client_id] = self._bandwidth_allocation_score(
                active_client,
                baseline,
                direction,
                demand_bytes,
                time_ms,
            )
        total_raw = sum(max(1e-6, score) for score in raw_scores.values())
        client_bandwidth_mhz = (
            total_bandwidth_mhz
            * max(1e-6, raw_scores.get(client.client_id, 1.0))
            / total_raw
        )
        snr_db = self._current_snr_db(client, direction, time_ms)
        spectral_efficiency = self._spectral_efficiency_from_snr_db(snr_db)
        rate_mbps = client_bandwidth_mhz * spectral_efficiency
        return client_bandwidth_mhz, rate_mbps, snr_db, spectral_efficiency, baseline

    def _estimate_active_payload_bytes(
        self,
        client: ClientProfile,
        direction: str,
        current_client: ClientProfile,
        current_payload_bytes: int,
    ) -> int:
        if client.client_id == current_client.client_id:
            return current_payload_bytes
        if direction == "uplink":
            return estimate_uplink_bytes_from_gamma(
                max(0, client.gamma),
                distribution_payload=self.controller.proposal_distribution_payload,
                vocab_size=self.controller.proposal_vocab_size,
            )
        token_count = max(1, min(client.gamma + 1, client.remaining_tokens))
        return estimate_downlink_bytes_from_token_count(token_count)

    def _bandwidth_allocation_score(
        self,
        client: ClientProfile,
        baseline: str,
        direction: str,
        payload_bytes: int,
        time_ms: float,
    ) -> float:
        weight = max(1e-6, client.wireless_weight)
        if baseline == "equal":
            return 1.0
        if baseline == "proportional":
            return weight
        if baseline == "queue-weighted":
            queue_signal = client.z_queue if direction == "downlink" else client.device_queue
            return weight * max(1.0, 1.0 + queue_signal)
        if baseline == "square-root":
            if direction == "uplink":
                spectral_efficiency = self._spectral_efficiency_from_snr_db(
                    self._current_snr_db(client, direction, time_ms)
                )
                payload_bits_per_token = 0.0
                if client.gamma > 0:
                    payload_bits_per_token = (8.0 * max(0.0, float(payload_bytes))) / float(
                        client.gamma
                    )
                paper_weight = paper_square_root_allocation_weight(
                    queue_price=client.device_queue,
                    payload_bits_per_token=payload_bits_per_token,
                    gamma=client.gamma,
                    spectral_efficiency=spectral_efficiency,
                )
                if paper_weight > 0.0:
                    return paper_weight
            return weight * math.sqrt(max(1.0, payload_bytes))
        raise ValueError(
            "shared_wireless allocator must be one of: "
            "equal, proportional, queue-weighted, square-root"
        )

    @staticmethod
    def _spectral_efficiency_from_snr_db(snr_db: float) -> float:
        return spectral_efficiency_from_snr_db(snr_db)

    @staticmethod
    def _current_snr_db(client: ClientProfile, direction: str, time_ms: float) -> float:
        return current_snr_db(client, direction, time_ms)

    def _advance_virtual_timeline(
        self,
        client: ClientProfile,
        draft_latency_ms: float,
        upload_ms: float,
        download_ms: float,
        server_arrival_ms: float,
        batch_ready_ms: float,
        batch_start_ms: float,
        verify_finish_ms: float,
    ) -> dict[str, float]:
        draft_start_ms = client.virtual_time_ms
        draft_finish_ms = draft_start_ms + draft_latency_ms
        uplink_start_ms = draft_finish_ms
        expected_server_arrival_ms = uplink_start_ms + client.rtt_ms / 2.0 + upload_ms
        assert abs(server_arrival_ms - expected_server_arrival_ms) <= 1e-6
        assert batch_ready_ms + 1e-6 >= server_arrival_ms
        assert batch_start_ms + 1e-6 >= batch_ready_ms
        assert verify_finish_ms + 1e-6 >= batch_start_ms
        self.virtual_target_available_at_ms = verify_finish_ms
        client_receive_ms = verify_finish_ms + client.rtt_ms / 2.0 + download_ms
        client.virtual_start_time_ms = draft_start_ms
        client.virtual_time_ms = client_receive_ms
        self.virtual_system_time_ms = max(self.virtual_system_time_ms, client_receive_ms)
        return {
            "draft_start_ms": draft_start_ms,
            "draft_finish_ms": draft_finish_ms,
            "uplink_start_ms": uplink_start_ms,
            "server_arrival_ms": server_arrival_ms,
            "batch_ready_ms": batch_ready_ms,
            "batch_start_ms": batch_start_ms,
            "verify_finish_ms": verify_finish_ms,
            "client_receive_ms": client_receive_ms,
            "virtual_round_latency_ms": client_receive_ms - draft_start_ms,
        }

    def _make_clients(self, samples, num_clients: int | None = None) -> list[ClientProfile]:
        c = self.cfg["clients"]
        configured_num_clients = int(self.cfg["simulation"]["num_clients"])
        num_clients = int(num_clients) if num_clients is not None else configured_num_clients
        if num_clients <= 0:
            raise ValueError("simulation.num_clients must be greater than zero")

        # eq. (formulation) heterogeneous acceptance diagnostic: client_classes may each
        # declare their own dataset override (client_classes.<name>.dataset), so different
        # groups of clients get genuinely different real measured acceptance rates instead
        # of all sharing one global sample pool.
        assigned_names: list[str] = []
        assigned_cfgs: list[dict] = []
        for i in range(num_clients):
            name, cfg = self._sample_client_class(c)
            assigned_names.append(name)
            assigned_cfgs.append(cfg)

        shards: list = [None] * num_clients

        if self._nested_population:
            # Section 4: fixed per-client prompt block, so the client set
            # for N is a strict prefix of the client set for any N' > N.
            # Homogeneous single-dataset only (pilot / Exp1 "start
            # homogeneous"). Per-client params are already index-deterministic
            # (RNG seeded once, consumed in client-index order below), so
            # only the prompt assignment needs to change here.
            from collections import deque as _deque
            import hashlib as _hashlib
            import json as _json

            qpc = self._nested_qpc
            for i in range(num_clients):
                shards[i] = _deque(samples[i * qpc : (i + 1) * qpc])
            fp_payload = _json.dumps(
                {
                    "seed": int(self.cfg["simulation"]["seed"]),
                    "n_max": self._nested_n_max,
                    "qpc": qpc,
                    "clients": [
                        {
                            "i": i,
                            "class": assigned_names[i],
                            "prompt_ids": [str(s.sample_id) for s in list(shards[i])],
                        }
                        for i in range(num_clients)
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            self.population_fingerprint = _hashlib.sha256(
                fp_payload.encode("utf-8")
            ).hexdigest()[:16]
        else:
            default_indices = [
                i for i in range(num_clients) if not assigned_cfgs[i].get("dataset")
            ]
            if default_indices:
                default_shards = shard_round_robin(samples, len(default_indices))
                for slot, i in enumerate(default_indices):
                    shards[i] = default_shards[slot]

            override_groups: dict[str, list[int]] = {}
            for i, cfg in enumerate(assigned_cfgs):
                override = cfg.get("dataset")
                if override:
                    override_groups.setdefault(assigned_names[i], []).append(i)
            for class_name, indices in override_groups.items():
                class_dataset_cfg = dict(self.cfg["dataset"])
                class_dataset_cfg.update(assigned_cfgs[indices[0]]["dataset"])
                class_samples = load_samples(class_dataset_cfg)
                self.samples.extend(class_samples)
                class_shards = shard_round_robin(class_samples, len(indices))
                for slot, i in enumerate(indices):
                    shards[i] = class_shards[slot]

        clients: list[ClientProfile] = []
        for i in range(num_clients):
            draft_device = self.workers[i % len(self.workers)].device
            client_class_name, class_cfg = assigned_names[i], assigned_cfgs[i]
            draft_slowdown = float(class_cfg.get("draft_slowdown", 1.0))
            base_draft_tps = self._sample_range(
                c, "draft_tokens_per_second_range", [20.0, 60.0]
            )
            clients.append(
                ClientProfile(
                    client_id=i,
                    prompt_queue=shards[i],
                    draft_model_id=str(self.cfg["models"]["draft"]),
                    draft_device_class=client_class_name,
                    draft_tokens_per_second=base_draft_tps / max(1e-6, draft_slowdown),
                    draft_fixed_latency_ms=self._sample_range(
                        class_cfg, "draft_fixed_latency_ms_range", c.get("draft_fixed_latency_ms_range", [5.0, 20.0])
                    ),
                    rtt_ms=self._sample_range(class_cfg, "rtt_ms", c["rtt_ms_range"]),
                    uplink_mbps=self._sample_range(class_cfg, "uplink_mbps", c["uplink_mbps_range"]),
                    downlink_mbps=self._sample_range(
                        class_cfg,
                        "downlink_mbps",
                        c.get("downlink_mbps_range", c["uplink_mbps_range"]),
                    ),
                    packet_loss=self._sample_range(
                        class_cfg, "packet_loss", c.get("packet_loss_range", [0.0, 0.0])
                    ),
                    channel_model=str(
                        class_cfg.get(
                            "channel_model",
                            c.get("channel_model", "iid_block_fading"),
                        )
                    ).lower(),
                    uplink_base_snr_db=self._sample_range(
                        class_cfg,
                        "uplink_base_snr_db",
                        c.get("uplink_base_snr_db_range", [8.0, 18.0]),
                    ),
                    downlink_base_snr_db=self._sample_range(
                        class_cfg,
                        "downlink_base_snr_db",
                        c.get("downlink_base_snr_db_range", [10.0, 20.0]),
                    ),
                    uplink_snr_jitter_db=self._sample_range(
                        class_cfg,
                        "uplink_snr_jitter_db",
                        c.get("uplink_snr_jitter_db_range", [1.0, 4.0]),
                    ),
                    downlink_snr_jitter_db=self._sample_range(
                        class_cfg,
                        "downlink_snr_jitter_db",
                        c.get("downlink_snr_jitter_db_range", [1.0, 4.0]),
                    ),
                    snr_period_ms=self._sample_range(
                        class_cfg,
                        "snr_period_ms",
                        c.get("snr_period_ms_range", [500.0, 3000.0]),
                    ),
                    snr_block_duration_ms=self._sample_range(
                        class_cfg,
                        "snr_block_duration_ms",
                        c.get("snr_block_duration_ms_range", [100.0, 100.0]),
                    ),
                    snr_phase_rad=random.uniform(0.0, 2.0 * math.pi),
                    channel_seed=int(self.cfg["simulation"]["seed"]),
                    wireless_weight=float(class_cfg.get("wireless_weight", 1.0)),
                    diagnostic_priority_multiplier=float(
                        c.get("diagnostic_priority_multiplier", {}).get(
                            client_class_name, 1.0
                        )
                    ),
                    alpha_prior_success=float(self.learning_cfg.get("alpha_prior_success", 1.0)),
                    alpha_prior_failure=float(self.learning_cfg.get("alpha_prior_failure", 1.0)),
                    alpha_ucb_scale=float(
                        self.learning_cfg.get(
                            "ucb_coefficient",
                            self.learning_cfg.get("alpha_ucb_scale", 1.0),
                        )
                    ),
                    alpha_discount=float(self.learning_cfg.get("discount", 1.0)),
                    alpha_learning_mode=str(
                        self.learning_cfg.get("mode", "bernstein_censored")
                    ).lower().replace("discounted_ucb", "bernstein_censored_discounted"),
                    alpha_profile_mode=str(
                        self.learning_cfg.get("profile_mode", "position")
                    ).lower(),
                    alpha_window_size=int(self.learning_cfg.get("window_size", 0)),
                    alpha_confidence_delta=float(
                        self.learning_cfg.get("confidence_delta", 0.05)
                    ),
                    alpha_min_value=float(self.learning_cfg.get("alpha_min", 1e-6)),
                    alpha_max_value=float(self.learning_cfg.get("alpha_max", 1.0 - 1e-6)),
                    draft_speed_multiplier=draft_slowdown,
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
            self.target.set_confirmed_prefix(client.client_id, [])
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
        context_token_ids = self.target.build_context_token_ids(
            client.current_sample.prompt, client.history_token_ids
        )
        (
            estimated_verifier_budget,
            estimated_verifier_regime,
            estimated_theta_f_ms_per_token,
        ) = self.verifier.estimate_verify_budget_signal(
            context_length=len(context_token_ids),
            incoming_requests=1,
        )
        self.controller.set_verifier_budget_signal(estimated_verifier_budget)
        self.controller.set_verifier_regime_signal(estimated_verifier_regime)
        self.controller.set_verifier_theta_f_signal(estimated_theta_f_ms_per_token)
        self.controller.set_active_client_signal(len(self._active_clients()))
        remaining = client.remaining_tokens
        selected_gamma = self.controller.choose_gamma(client, self.total_rounds)
        gamma = min(selected_gamma, max(0, remaining - 1))
        client.gamma = gamma
        worker = self.workers[client.client_id % len(self.workers)]
        round_start = time.perf_counter()

        if gamma > 0:
            draft_confirmed_length = await worker.confirmed_length(client.client_id)
            if draft_confirmed_length is not None:
                assert draft_confirmed_length == len(context_token_ids)
            draft_wall_start = time.perf_counter()
            draft = await worker.generate(
                DraftRequest(
                    client_id=client.client_id,
                    context_token_ids=context_token_ids,
                    gamma=gamma,
                    temperature=self.sampling_temperature,
                )
            )
            draft_wall_latency_ms = (time.perf_counter() - draft_wall_start) * 1000.0
        else:
            draft = DraftResult(
                client.client_id,
                [],
                [],
                "",
                0.0,
                -1,
                distribution_payload="delta_proposal",
            )
            draft_wall_latency_ms = 0.0

        upload_ms = 0.0
        download_ms = 0.0
        upload_bytes = 0
        download_bytes = 0
        uplink_bandwidth_mhz = 0.0
        downlink_bandwidth_mhz = 0.0
        uplink_rate_mbps = client.uplink_mbps
        downlink_rate_mbps = client.downlink_mbps
        uplink_allocator = "proportional"
        downlink_allocator = "proportional"
        uplink_snr_db = client.uplink_base_snr_db
        downlink_snr_db = client.downlink_base_snr_db
        uplink_spectral_efficiency = 0.0
        downlink_spectral_efficiency = 0.0
        draft_edge_latency_ms = resolve_effective_draft_latency_ms(
            mode=self.draft_execution_mode,
            measured_gpu_latency_ms=draft.latency_ms,
            measured_wall_latency_ms=draft_wall_latency_ms,
            fixed_latency_ms=client.draft_fixed_latency_ms,
            speed_multiplier=client.draft_speed_multiplier,
        )
        draft_wait_ms = draft_queue_wait_ms(
            measured_gpu_latency_ms=draft.latency_ms,
            measured_wall_latency_ms=draft_wall_latency_ms,
        )
        if gamma > 0:
            upload_bytes = estimate_uplink_bytes(draft, vocab_size=self.target.vocab_size)
            (
                uplink_bandwidth_mhz,
                uplink_rate_mbps,
                uplink_snr_db,
                uplink_spectral_efficiency,
                uplink_allocator,
            ) = self._allocate_shared_bandwidth(
                client,
                "uplink",
                client.virtual_time_ms + draft_edge_latency_ms,
                upload_bytes,
            )
            upload_ms = self._transfer_time_ms(
                upload_bytes, uplink_rate_mbps, client.packet_loss
            )

        draft_start_ms = client.virtual_time_ms
        draft_finish_ms = draft_start_ms + draft_edge_latency_ms
        server_arrival_ms = draft_finish_ms + client.rtt_ms / 2.0 + upload_ms
        batch_metadata, verification = await self.verifier.submit(
            client_id=client.client_id,
            context_token_ids=context_token_ids,
            draft=draft,
            remaining_tokens=remaining,
            server_arrival_ms=server_arrival_ms,
            # AUC_ACHIEVABLE_REGION_DIAGNOSTIC.md eq. in Section 4:
            # w_i = m_{c(i)} * (V + Z_i). diagnostic_priority_multiplier
            # defaults to 1.0 (no-op) unless clients.diagnostic_priority_multiplier
            # is set in config, so this does not change any prior experiment.
            weight=client.diagnostic_priority_multiplier
            * (self.controller.V + (client.z_queue if self.controller.use_virtual_queues else 0.0)),
            deadline=None,
            expected_acceptance_rate=client.alpha_hat,
            expected_acceptance_profile=client.positional_acceptance_profile(
                gamma,
                use_ucb=False,
            ),
        )
        batch_id = batch_metadata.batch_id
        verification_batch_size = batch_metadata.batch_size

        committed = verification.committed_token_ids
        await worker.commit(client.client_id, committed)
        client.history_token_ids.extend(committed)
        self.target.set_confirmed_prefix(client.client_id, client.history_token_ids)
        committed_length = await worker.confirmed_length(client.client_id)
        if committed_length is not None:
            assert committed_length == len(
                self.target.build_context_token_ids(
                    client.current_sample.prompt, client.history_token_ids
                )
            )
        target_prefix_hash = self.target.confirmed_prefix_hash(client.client_id)
        assert target_prefix_hash == self._prefix_hash(client.history_token_ids)
        useful = verification.useful_token_count
        if useful > 0:
            download_bytes = estimate_downlink_bytes(verification)
            (
                downlink_bandwidth_mhz,
                downlink_rate_mbps,
                downlink_snr_db,
                downlink_spectral_efficiency,
                downlink_allocator,
            ) = self._allocate_shared_bandwidth(
                client,
                "downlink",
                batch_metadata.verify_finish_ms,
                download_bytes,
            )
            download_ms = self._transfer_time_ms(
                download_bytes, downlink_rate_mbps, client.packet_loss
            )
        client.generated_for_sample += useful
        client.generated_tokens += useful
        if gamma > 0:
            client.observe_acceptance_feedback(
                proposed_tokens=gamma,
                accepted_length=verification.accepted_length,
                inspected_tokens=len(verification.acceptance_probabilities),
            )
            client.speculative_rounds += 1
        client.refresh_learning_stats(self.total_rounds)
        client.accepted_tokens += verification.accepted_length
        client.useful_tokens += useful
        client.rounds_completed += 1

        wall_round_latency_ms = (time.perf_counter() - round_start) * 1000.0
        virtual_timing = self._advance_virtual_timeline(
            client=client,
            draft_latency_ms=draft_edge_latency_ms,
            upload_ms=upload_ms,
            download_ms=download_ms,
            server_arrival_ms=server_arrival_ms,
            batch_ready_ms=batch_metadata.batch_ready_ms,
            batch_start_ms=batch_metadata.batch_start_ms,
            verify_finish_ms=batch_metadata.verify_finish_ms,
        )
        lambda_price = self.controller.observe_server_batch(
            batch_id=batch_id,
            verify_finish_ms=batch_metadata.verify_finish_ms,
            theta_f_ms_per_token=batch_metadata.theta_f_ms_per_token,
            verifier_token_cost=batch_metadata.batch_token_cost,
        )
        round_latency_ms = virtual_timing["virtual_round_latency_ms"]
        edge_finish_ms = virtual_timing["uplink_start_ms"] + upload_ms
        tau_d_ms = self.controller._draft_time_ms_per_token(client)
        kappa_bits_per_token = self.controller._uplink_kappa_bits_per_token()
        kappa_over_r_ms = self.controller._network_time_ms_for_kappa_bits(
            client=client,
            kappa_bits_per_token=kappa_bits_per_token,
            rate_mbps=uplink_rate_mbps,
        )
        paper_edge_service_ms = gamma * (tau_d_ms + kappa_over_r_ms)
        slot_observation = self.controller.update(
            client,
            useful_tokens=useful,
            client_receive_ms=virtual_timing["client_receive_ms"],
            edge_finish_ms=edge_finish_ms,
            gamma=gamma,
            tau_d_ms=tau_d_ms,
            kappa_over_r_ms=kappa_over_r_ms,
        )

        record: dict[str, object] = {
            "client_id": client.client_id,
            "sample_id": client.current_sample.sample_id,
            "dataset_name": client.current_sample.dataset_name,
            "prompt": client.current_sample.prompt,
            "sample_start_time_ms": client.sample_start_time_ms,
            "experiment_warmup_end_ms": self.measurement_start_ms,
            "experiment_measurement_end_ms": self.measurement_end_ms,
            "round_id": round_id,
            "verification_batch_id": batch_id,
            "verification_batch_size": verification_batch_size,
            "verification_batch_token_cost": batch_metadata.batch_token_cost,
            "verification_batch_token_budget": batch_metadata.batch_token_budget,
            "estimated_verifier_budget": estimated_verifier_budget,
            "estimated_verifier_regime": estimated_verifier_regime,
            "verification_scheduler": batch_metadata.scheduler_name,
            "verification_batch_service_ms": batch_metadata.modeled_batch_service_ms,
            "verification_batch_measured_service_ms": batch_metadata.measured_batch_service_ms,
            "controller_policy": self.controller.policy,
            "controller_V": self.controller.V,
            "population_fingerprint": self.population_fingerprint,
            "n_active": len(self.clients),
            "x_requirement": self.controller.min_tps,
            "control_slot_ms": self.controller.slot_ms,
            "control_slot_index": slot_observation.slot_index,
            "slot_useful_tokens": slot_observation.slot_useful_tokens,
            "worker_id": draft.worker_id,
            "selected_gamma": selected_gamma,
            "gamma": gamma,
            "accepted_length": verification.accepted_length,
            "committed_token_count": verification.useful_token_count,
            "committed_correction_token": verification.correction_token_id is not None,
            "committed_bonus_token": verification.bonus_token_id is not None,
            "acceptance_observed_tokens": len(verification.acceptance_probabilities),
            "acceptance_right_censored": (
                gamma > 0
                and len(verification.acceptance_probabilities) == gamma
                and verification.accepted_length == gamma
            ),
            "useful_tokens": useful,
            "generated_for_sample": client.generated_for_sample,
            "max_new_tokens": client.current_sample.max_new_tokens,
            "reached_eos": verification.reached_eos,
            "finish_reason": verification.finish_reason,
            "alpha_hat": client.alpha_hat,
            "alpha_ucb": client.alpha_ucb,
            "speculative_rounds": client.speculative_rounds,
            "controller_forced_exploration": client.last_forced_exploration,
            "controller_exploration_reason": client.last_exploration_reason,
            "alpha_observed_tokens": client.alpha_observed_tokens,
            "alpha_censored_rounds": client.alpha_censored_rounds,
            "alpha_failure_events": client.alpha_failure_events,
            "z_queue": client.z_queue,
            "device_queue": client.device_queue,
            "server_queue": self.controller.server_queue,
            "lambda_price": lambda_price,
            "rtt_ms": client.rtt_ms,
            "uplink_bandwidth_mhz": uplink_bandwidth_mhz,
            "downlink_bandwidth_mhz": downlink_bandwidth_mhz,
            "uplink_allocator": uplink_allocator,
            "downlink_allocator": downlink_allocator,
            "uplink_rate_mbps": uplink_rate_mbps,
            "downlink_rate_mbps": downlink_rate_mbps,
            "upload_bytes": upload_bytes,
            "upload_ms": upload_ms,
            "edge_service_ms": paper_edge_service_ms,
            "download_bytes": download_bytes,
            "download_ms": download_ms,
            "downlink_mbps": client.downlink_mbps,
            "uplink_base_snr_db": client.uplink_base_snr_db,
            "downlink_base_snr_db": client.downlink_base_snr_db,
            "uplink_snr_db": uplink_snr_db,
            "downlink_snr_db": downlink_snr_db,
            "uplink_snr_jitter_db": client.uplink_snr_jitter_db,
            "downlink_snr_jitter_db": client.downlink_snr_jitter_db,
            "snr_period_ms": client.snr_period_ms,
            "snr_block_duration_ms": client.snr_block_duration_ms,
            "channel_model": client.channel_model,
            "uplink_spectral_efficiency": uplink_spectral_efficiency,
            "downlink_spectral_efficiency": downlink_spectral_efficiency,
            "wireless_weight": client.wireless_weight,
            "packet_loss": client.packet_loss,
            "draft_model_id": client.draft_model_id,
            "draft_execution_mode": self.draft_execution_mode,
            "draft_device_class": client.draft_device_class,
            "draft_tokens_per_second": client.draft_tokens_per_second,
            "draft_fixed_latency_ms": client.draft_fixed_latency_ms,
            "draft_measured_latency_ms": draft.latency_ms,
            "draft_measured_wall_latency_ms": draft_wall_latency_ms,
            "draft_queue_wait_ms": draft_wait_ms,
            "draft_latency_ms": draft_edge_latency_ms,
            "target_latency_ms": verification.target_latency_ms,
            "round_latency_ms": round_latency_ms,
            "wall_round_latency_ms": wall_round_latency_ms,
            "wall_clock_time_ms": (time.perf_counter() - self.wall_clock_start) * 1000.0,
            "virtual_system_time_ms": self.virtual_system_time_ms,
            "virtual_target_available_at_ms": self.virtual_target_available_at_ms,
            "committed_text": verification.committed_text,
        }
        record.update(virtual_timing)
        client_receive_ms = float(record["client_receive_ms"])
        in_measurement_window = client_receive_ms >= self.measurement_start_ms
        if self.measurement_end_ms is not None:
            in_measurement_window = (
                in_measurement_window and client_receive_ms <= self.measurement_end_ms
            )
        record["in_warmup_window"] = client_receive_ms < self.measurement_start_ms
        record["in_measurement_window"] = in_measurement_window
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
        output_stem = output.stem
        artifacts_dir = output.parent / f"{output_stem}_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.records[0].keys()))
            writer.writeheader()
            writer.writerows(self.records)
        df = pd.DataFrame(self.records)
        rounds_parquet = artifacts_dir / "rounds.parquet"
        events_jsonl = artifacts_dir / "events.jsonl"
        run_config_json = artifacts_dir / "run_config.json"
        summary_json = artifacts_dir / "summary.json"
        try:
            df.to_parquet(rounds_parquet, index=False)
        except Exception as exc:
            with (artifacts_dir / "rounds_parquet_error.txt").open("w", encoding="utf-8") as handle:
                handle.write(f"{type(exc).__name__}: {exc}\n")
        self._write_events_jsonl(events_jsonl, df)
        self._write_run_config_json(run_config_json, output_csv=output, artifacts_dir=artifacts_dir)
        self._write_summary_json(summary_json, df, output_csv=output, artifacts_dir=artifacts_dir)
        print(f"Wrote {len(self.records)} records to {output}")

    @staticmethod
    def _package_version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    @staticmethod
    def _git_commit() -> str | None:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                .strip()
            )
        except Exception:
            return None

    def _write_run_config_json(
        self,
        path: Path,
        *,
        output_csv: Path,
        artifacts_dir: Path,
    ) -> None:
        payload = {
            "run_id": f"{output_csv.stem}-{int(time.time() * 1000)}",
            "git_commit": self._git_commit(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_path": self.config_path,
            "output_csv": str(output_csv),
            "artifacts_dir": str(artifacts_dir),
            "models": self.cfg.get("models", {}),
            "dataset": self.cfg.get("dataset", {}),
            "dataset_prompt_ids": [sample.sample_id for sample in self.samples],
            "seed": int(self.cfg["simulation"]["seed"]),
            "simulation": self.cfg.get("simulation", {}),
            "controller": self.cfg.get("controller", {}),
            "control": self.cfg.get("control", {}),
            "verification_batching": self.cfg.get("verification_batching", {}),
            "clients": self.cfg.get("clients", {}),
            "experiment": self.cfg.get("experiment", {}),
            "hardware": {
                "platform": platform.platform(),
                "python": sys.version,
                "draft_devices": list(self.cfg.get("models", {}).get("draft_devices", [])),
                "target_replication_factor": int(
                    self.cfg.get("verification_batching", {}).get("target_replication_factor", 1)
                ),
            },
            "software_versions": {
                "python": platform.python_version(),
                "pandas": self._package_version("pandas"),
                "pyarrow": self._package_version("pyarrow"),
                "torch": self._package_version("torch"),
                "transformers": self._package_version("transformers"),
                "vllm": self._package_version("vllm"),
            },
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _event_wall_time_ms(
        self,
        record: dict[str, object],
        event_virtual_ms: float,
    ) -> float:
        wall_end_ms = float(record.get("wall_clock_time_ms", 0.0))
        wall_round_latency_ms = max(0.0, float(record.get("wall_round_latency_ms", 0.0)))
        virtual_start_ms = float(record.get("draft_start_ms", 0.0))
        virtual_round_latency_ms = max(0.0, float(record.get("round_latency_ms", 0.0)))
        wall_start_ms = wall_end_ms - wall_round_latency_ms
        if virtual_round_latency_ms <= 1e-9:
            return wall_end_ms
        progress = (event_virtual_ms - virtual_start_ms) / virtual_round_latency_ms
        progress = min(1.0, max(0.0, progress))
        return wall_start_ms + progress * wall_round_latency_ms

    def _write_events_jsonl(self, path: Path, df: pd.DataFrame) -> None:
        event_specs = [
            ("draft_start", "draft_start_ms"),
            ("draft_finish", "draft_finish_ms"),
            ("uplink_start", "uplink_start_ms"),
            ("uplink_finish", None),
            ("server_arrival", "server_arrival_ms"),
            ("batch_start", "batch_start_ms"),
            ("batch_finish", "verify_finish_ms"),
            ("downlink_finish", "client_receive_ms"),
            ("request_complete", "client_receive_ms"),
        ]
        with path.open("w", encoding="utf-8") as handle:
            for record in df.to_dict(orient="records"):
                rtt_half_ms = float(record.get("rtt_ms", 0.0)) / 2.0
                upload_ms = float(record.get("upload_ms", 0.0))
                for event_type, column in event_specs:
                    if event_type == "uplink_finish":
                        event_virtual_ms = float(record.get("uplink_start_ms", 0.0)) + upload_ms
                    else:
                        value = record.get(column) if column is not None else None
                        if value is None or (isinstance(value, float) and math.isnan(value)):
                            continue
                        event_virtual_ms = float(value)
                    event = {
                        "timestamp_virtual_ms": event_virtual_ms,
                        "timestamp_wall_ms": self._event_wall_time_ms(record, event_virtual_ms),
                        "event_type": event_type,
                        "client_id": int(record["client_id"]),
                        "session_id": str(record["sample_id"]),
                        "prompt_id": str(record["sample_id"]),
                        "round_id": int(record["round_id"]),
                        "batch_id": int(record["verification_batch_id"]),
                    }
                    if event_type == "server_arrival":
                        event["uplink_finish_ms"] = event_virtual_ms - rtt_half_ms
                    if event_type == "downlink_finish":
                        event["downlink_start_ms"] = float(record.get("verify_finish_ms", 0.0)) + rtt_half_ms
                    handle.write(json.dumps(event) + "\n")

    def _write_summary_json(
        self,
        path: Path,
        df: pd.DataFrame,
        *,
        output_csv: Path,
        artifacts_dir: Path,
    ) -> None:
        summary = summarize_round_csv(df)
        per_sample = summary.pop("per_sample")
        per_client = summary.pop("per_client")
        if isinstance(per_client, pd.DataFrame) and not per_client.empty:
            sorted_interactivity = sorted(per_client["interactivity_s_per_token"].astype(float).tolist())
            sorted_service_rates = sorted(
                per_client["goodput_tps"].astype(float).tolist()
            )
            summary.update(
                {
                    "per_client_interactivity_s_per_token": per_client[
                        "interactivity_s_per_token"
                    ].astype(float).tolist(),
                    "minimum_interactivity_s_per_token": float(min(sorted_interactivity)),
                    "p10_interactivity_s_per_token": float(
                        pd.Series(sorted_interactivity, dtype=float).quantile(0.10)
                    ),
                    "median_interactivity_s_per_token": float(
                        pd.Series(sorted_interactivity, dtype=float).median()
                    ),
                    "minimum_interactivity_tps": float(min(sorted_service_rates)),
                    "p10_interactivity_tps": float(
                        pd.Series(sorted_service_rates, dtype=float).quantile(0.10)
                    ),
                    "median_interactivity_tps": float(
                        pd.Series(sorted_service_rates, dtype=float).median()
                    ),
                }
            )
        summary.update(
            {
                "run_id": f"{output_csv.stem}",
                "output_csv": str(output_csv),
                "artifacts_dir": str(artifacts_dir),
                "verification_tokens_per_useful_token": (
                    float(df["verification_batch_token_cost"].sum()) / max(1.0, float(df["useful_tokens"].sum()))
                ),
                "target_utilization": float(summary.get("server_utilization", 0.0)),
                "mean_server_queue": float(df["server_queue"].mean()),
                "p95_server_queue": float(df["server_queue"].quantile(0.95)),
                "mean_gamma": float(df["gamma"].mean()),
                "acceptance_rate": float(summary.get("acceptance_rate", 0.0)),
                "per_client_rows": len(per_client) if isinstance(per_client, pd.DataFrame) else 0,
                "per_sample_rows": len(per_sample) if isinstance(per_sample, pd.DataFrame) else 0,
            }
        )
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
