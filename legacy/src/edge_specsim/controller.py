from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .expectations import expected_useful_tokens, expected_useful_tokens_from_profile
from .models import ClientProfile
from .network import estimate_uplink_bits_per_token, estimate_uplink_bytes_from_gamma


@dataclass(frozen=True)
class SlotObservation:
    slot_index: int
    slot_useful_tokens: int


def paper_index_gamma(
    alpha_hat: float,
    c_value: float,
    gamma_max: int,
) -> int:
    clipped_gamma_max = max(0, int(gamma_max))
    if clipped_gamma_max <= 0:
        return 0
    alpha = min(1.0 - 1e-9, max(1e-9, float(alpha_hat)))
    if c_value <= 0.0:
        return clipped_gamma_max
    if c_value >= 1.0:
        return 0
    gamma = math.ceil(math.log(float(c_value)) / math.log(alpha)) - 1
    return max(0, min(clipped_gamma_max, int(gamma)))


class OnlineController:
    def __init__(
        self,
        gamma_choices: list[int],
        V: float,
        min_tps: float,
        server_price_scale: float,
        verifier_price_lambda: float = 1.0,
        draft_latency_weight: float = 0.0,
        uplink_latency_weight: float = 0.0,
        policy: str = "adaptive_ucb",
        fixed_gamma: int = 0,
        seed: int = 0,
        slot_ms: float = 100.0,
        min_exploration_rounds: int = 0,
        exploration_gamma: int = 1,
        exploration_epsilon: float = 0.0,
        use_virtual_queues: bool = True,
        proposal_distribution_payload: str = "delta_proposal",
        proposal_vocab_size: int | None = None,
        # --- Phase 3 baselines added on top of the original policy set ---
        # (sim/controllers/{goodspeed,turbospec,fixed_slo,gelato}.py's
        # from-scratch formulations, ported here so they can drive this
        # same real-GPU async simulator instead of only sim/'s closed-form
        # one; "adaptive_index" above already *is* the AUC-controller, see
        # docs/model_choices.md-adjacent commit notes).
        fixed_slo_ms: float = 200.0,
        gelato_v: float | None = None,
        gelato_draft_power_mw: float = 500.0,
        gelato_uplink_power_mw: float = 200.0,
        gelato_energy_budget_mj: float = 50.0,
    ) -> None:
        if not gamma_choices:
            raise ValueError("controller.gamma_choices must contain at least one action")
        normalized_choices = sorted({int(choice) for choice in gamma_choices})
        if normalized_choices[0] < 0:
            raise ValueError("controller.gamma_choices cannot contain negative values")
        if 0 not in normalized_choices:
            raise ValueError("controller.gamma_choices must include 0 for target-only mode")
        self.gamma_choices = normalized_choices
        self.V = float(V)
        self.min_tps = min_tps
        self.server_price_scale = server_price_scale
        self.verifier_price_lambda = verifier_price_lambda
        self.draft_latency_weight = draft_latency_weight
        self.uplink_latency_weight = uplink_latency_weight
        self.policy = str(policy).lower()
        self.fixed_gamma = int(fixed_gamma)
        self.rng = random.Random(seed)
        self.slot_ms = float(slot_ms)
        self.min_exploration_rounds = max(0, int(min_exploration_rounds))
        self.exploration_epsilon = min(1.0, max(0.0, float(exploration_epsilon)))
        self.use_virtual_queues = bool(use_virtual_queues)
        self.proposal_distribution_payload = str(proposal_distribution_payload)
        self.proposal_vocab_size = (
            None if proposal_vocab_size is None else max(1, int(proposal_vocab_size))
        )
        if self.slot_ms <= 0.0:
            raise ValueError("control.slot_ms must be greater than zero")
        self.server_queue = 0.0
        self.server_price = 0.0
        self.current_verifier_budget_tokens = max(1, max(self.gamma_choices) + 1)
        self.current_verifier_regime = "memory_bound"
        self.current_verifier_theta_f_ms_per_token = 1.0
        self.current_active_clients = 1
        self._slot_completions: dict[int, dict[int, int]] = {}
        self._slot_state: dict[int, tuple[int, float, int]] = {}
        self._server_slot_state: tuple[int, float, float] | None = None
        self._device_slot_state: dict[int, tuple[int, float, float]] = {}
        self._seen_server_batches: set[int] = set()
        self.fixed_slo_ms = float(fixed_slo_ms)
        self.gelato_v = float(V) if gelato_v is None else float(gelato_v)
        self.gelato_draft_power_mw = float(gelato_draft_power_mw)
        self.gelato_uplink_power_mw = float(gelato_uplink_power_mw)
        self.gelato_energy_budget_mj = float(gelato_energy_budget_mj)
        self._gelato_energy_queue: dict[int, float] = {}
        self._gelato_last_energy_mj: dict[int, float] = {}
        self._goodspeed_epsilon_x = 1e-6  # tokens/s floor so 1/x_hat never blows up before any tokens land
        valid_policies = {
            "target_only",
            "fixed_gamma",
            "load_only",
            "random_gamma",
            "oracle_gamma",
            "adaptive_queue",
            "adaptive_ucb",
            "adaptive_index",
            "goodspeed",
            "turbospec",
            "fixed_slo",
            "gelato",
        }
        if self.policy not in valid_policies:
            raise ValueError(
                "controller.policy must be one of: "
                "target_only, fixed_gamma, load_only, random_gamma, oracle_gamma, "
                "adaptive_queue, adaptive_ucb, adaptive_index, "
                "goodspeed, turbospec, fixed_slo, gelato"
            )
        if self.fixed_gamma not in self.gamma_choices:
            raise ValueError("controller.fixed_gamma must be included in controller.gamma_choices")
        self._positive_gamma_choices = [choice for choice in self.gamma_choices if choice > 0]
        if self._positive_gamma_choices:
            self.exploration_gamma = self._nearest_positive_gamma(exploration_gamma)
        else:
            self.exploration_gamma = 0

    def _nearest_positive_gamma(self, gamma: int) -> int:
        if not self._positive_gamma_choices:
            return 0
        requested_gamma = max(1, int(gamma))
        for candidate in self._positive_gamma_choices:
            if candidate >= requested_gamma:
                return candidate
        return self._positive_gamma_choices[-1]

    @staticmethod
    def _draft_time_ms(client: ClientProfile, gamma: int) -> float:
        if gamma <= 0:
            return 0.0
        per_token_ms = 1000.0 / max(1e-6, client.draft_tokens_per_second)
        return client.draft_fixed_latency_ms + gamma * per_token_ms

    @staticmethod
    def _draft_time_ms_per_token(client: ClientProfile) -> float:
        return 1000.0 / max(1e-6, client.draft_tokens_per_second)

    @staticmethod
    def _uplink_time_ms_for_payload(
        client: ClientProfile,
        gamma: int,
        distribution_payload: str,
        proposal_vocab_size: int | None,
    ) -> float:
        if gamma <= 0:
            return 0.0
        payload_bytes = estimate_uplink_bytes_from_gamma(
            gamma,
            distribution_payload=distribution_payload,
            vocab_size=proposal_vocab_size,
        )
        effective_uplink_mbps = max(1e-6, client.uplink_mbps) * max(1e-6, 1.0 - client.packet_loss)
        return payload_bytes * 8.0 / (effective_uplink_mbps * 1_000_000.0) * 1000.0

    def _uplink_time_ms(self, client: ClientProfile, gamma: int) -> float:
        return self._uplink_time_ms_for_payload(
            client,
            gamma,
            self.proposal_distribution_payload,
            self.proposal_vocab_size,
        )

    def _uplink_time_ms_per_token(
        self,
        client: ClientProfile,
        rate_mbps: float | None = None,
    ) -> float:
        kappa_bits_per_token = self._uplink_kappa_bits_per_token()
        return self._network_time_ms_for_kappa_bits(
            client=client,
            kappa_bits_per_token=kappa_bits_per_token,
            rate_mbps=rate_mbps,
        )

    def _uplink_kappa_bits_per_token(self) -> float:
        return estimate_uplink_bits_per_token(
            distribution_payload=self.proposal_distribution_payload,
            vocab_size=self.proposal_vocab_size,
        )

    @staticmethod
    def _network_time_ms_for_kappa_bits(
        client: ClientProfile,
        kappa_bits_per_token: float,
        rate_mbps: float | None = None,
    ) -> float:
        effective_uplink_mbps = max(
            1e-6,
            float(client.uplink_mbps if rate_mbps is None else rate_mbps),
        ) * max(1e-6, 1.0 - client.packet_loss)
        return float(kappa_bits_per_token) / (effective_uplink_mbps * 1_000_000.0) * 1000.0

    def set_verifier_budget_signal(self, budget_tokens: int | float | None) -> None:
        if budget_tokens is None:
            return
        self.current_verifier_budget_tokens = max(1, int(float(budget_tokens)))
        self._refresh_server_price()

    def set_verifier_regime_signal(self, regime: str | None) -> None:
        if regime is None:
            return
        normalized = str(regime).strip().lower()
        if normalized not in {"memory_bound", "roofline_knee", "compute_bound"}:
            return
        self.current_verifier_regime = normalized

    def set_verifier_theta_f_signal(self, theta_f_ms_per_token: float | None) -> None:
        if theta_f_ms_per_token is None:
            return
        self.current_verifier_theta_f_ms_per_token = max(
            0.0,
            float(theta_f_ms_per_token),
        )

    def set_active_client_signal(self, active_clients: int | None) -> None:
        if active_clients is None:
            return
        self.current_active_clients = max(1, int(active_clients))

    def _refresh_server_price(self) -> float:
        if not self.use_virtual_queues:
            self.server_price = 0.0
            return self.server_price
        self.server_price = float(self.server_queue)
        return self.server_price

    def refresh_server_price(self) -> float:
        return self._refresh_server_price()

    def _verifier_regime_multiplier(self) -> float:
        if self.current_verifier_regime == "memory_bound":
            return 0.5
        if self.current_verifier_regime == "compute_bound":
            return 2.0
        return 1.0

    def score_gamma(
        self,
        client: ClientProfile,
        gamma: int,
        acceptance_profile: list[float],
    ) -> float:
        queue_weight = client.z_queue if self.use_virtual_queues else 0.0
        benefit = (self.V + queue_weight) * expected_useful_tokens_from_profile(
            acceptance_profile
        )
        lambda_price = self.server_price
        verifier_price = (
            self.verifier_price_lambda
            * lambda_price
            * (gamma + 1)
        )
        draft_cost = self.draft_latency_weight * self._draft_time_ms(client, gamma)
        uplink_cost = self.uplink_latency_weight * self._uplink_time_ms(client, gamma)
        return benefit - verifier_price - draft_cost - uplink_cost

    def _argmax_gamma(self, client: ClientProfile, use_ucb: bool) -> int:
        gamma = 0
        best_score = self.score_gamma(client, 0, [])
        for candidate in self.gamma_choices[1:]:
            acceptance_profile = client.positional_acceptance_profile(candidate, use_ucb)
            score = self.score_gamma(client, candidate, acceptance_profile)
            if score >= best_score:
                gamma = candidate
                best_score = score
        return gamma

    def _queue_policy_gamma(self, client: ClientProfile, use_ucb: bool) -> int:
        gamma = 0
        current_profile: list[float] = []
        current_expected = expected_useful_tokens_from_profile(current_profile)
        for candidate in self.gamma_choices[1:]:
            acceptance_profile = client.positional_acceptance_profile(candidate, use_ucb)
            candidate_expected = expected_useful_tokens_from_profile(acceptance_profile)
            queue_weight = client.z_queue if self.use_virtual_queues else 0.0
            marginal_benefit = (self.V + queue_weight) * (
                candidate_expected - current_expected
            )
            verifier_cost = self.verifier_price_lambda * self.server_price
            draft_cost = self.draft_latency_weight * (
                self._draft_time_ms(client, candidate) - self._draft_time_ms(client, gamma)
            )
            uplink_cost = self.uplink_latency_weight * (
                self._uplink_time_ms(client, candidate) - self._uplink_time_ms(client, gamma)
            )
            queue_cost = 0.01 * client.device_queue if self.use_virtual_queues else 0.0
            if marginal_benefit > verifier_cost + draft_cost + uplink_cost + queue_cost:
                gamma = candidate
                current_profile = acceptance_profile
                current_expected = candidate_expected
        return gamma

    def _scalar_acceptance(self, client: ClientProfile, use_ucb: bool) -> float:
        alpha = client.alpha_ucb if use_ucb else client.alpha_hat
        return min(1.0 - 1e-9, max(1e-9, float(alpha)))

    def _paper_index_cost_ratio(self, client: ClientProfile, use_ucb: bool) -> float:
        alpha = self._scalar_acceptance(client, use_ucb)
        del alpha  # acceptance is consumed separately by the closed-form gamma map
        lambda_price = self.server_queue if self.use_virtual_queues else 0.0
        mu_price = client.device_queue if self.use_virtual_queues else 0.0
        queue_weight = client.z_queue if self.use_virtual_queues else 0.0
        benefit_scale = max(1e-9, self.V + queue_weight)
        tau_d_ms = self._draft_time_ms_per_token(client)
        kappa_bits_per_token = self._uplink_kappa_bits_per_token()
        r_i_mbps = max(1e-6, client.uplink_mbps)
        kappa_over_r_ms = self._network_time_ms_for_kappa_bits(
            client=client,
            kappa_bits_per_token=kappa_bits_per_token,
            rate_mbps=r_i_mbps,
        )
        marginal_cost = (
            lambda_price * self.current_verifier_theta_f_ms_per_token
            + mu_price * (tau_d_ms + kappa_over_r_ms)
        )
        return float(marginal_cost) / benefit_scale

    def _index_policy_gamma(self, client: ClientProfile, use_ucb: bool) -> int:
        if not self.gamma_choices:
            return 0
        alpha = self._scalar_acceptance(client, use_ucb)
        c_value = self._paper_index_cost_ratio(client, use_ucb)
        gamma = paper_index_gamma(
            alpha_hat=alpha,
            c_value=c_value,
            gamma_max=max(self.gamma_choices),
        )
        feasible = [choice for choice in self.gamma_choices if choice <= gamma]
        if feasible:
            return feasible[-1]
        return 0

    def _goodspeed_cost_ratio(self, client: ClientProfile) -> float:
        """Same eq.(13) stopping-rule shape as _paper_index_cost_ratio, but
        the benefit weight is 1/x_hat_i(t) (proportional-fair marginal
        utility d(log x_i)/dx_i) instead of (V + Z_i(t)) -- see
        sim/controllers/goodspeed.py's docstring for the derivation. Server/
        device queue prices (lambda, mu) are unchanged: GoodSpeed still
        reacts to real scarcity, it just weighs the benefit side by
        fairness instead of a Lyapunov backlog target.
        """
        x_hat = client.useful_tokens / max(self._goodspeed_epsilon_x, client.virtual_time_ms / 1000.0)
        benefit_scale = 1.0 / max(self._goodspeed_epsilon_x, x_hat)
        lambda_price = self.server_queue if self.use_virtual_queues else 0.0
        mu_price = client.device_queue if self.use_virtual_queues else 0.0
        tau_d_ms = self._draft_time_ms_per_token(client)
        kappa_bits_per_token = self._uplink_kappa_bits_per_token()
        kappa_over_r_ms = self._network_time_ms_for_kappa_bits(
            client=client,
            kappa_bits_per_token=kappa_bits_per_token,
            rate_mbps=max(1e-6, client.uplink_mbps),
        )
        marginal_cost = (
            lambda_price * self.current_verifier_theta_f_ms_per_token
            + mu_price * (tau_d_ms + kappa_over_r_ms)
        )
        return float(marginal_cost) / benefit_scale

    def _goodspeed_gamma(self, client: ClientProfile) -> int:
        if not self.gamma_choices:
            return 0
        alpha = self._scalar_acceptance(client, use_ucb=False)
        c_value = self._goodspeed_cost_ratio(client)
        gamma = paper_index_gamma(alpha_hat=alpha, c_value=c_value, gamma_max=max(self.gamma_choices))
        feasible = [choice for choice in self.gamma_choices if choice <= gamma]
        return feasible[-1] if feasible else 0

    def _round_time_estimate_ms(self, client: ClientProfile, gamma: int) -> float:
        """Round-trip estimate used by turbospec/fixed_slo/gelato: draft +
        uplink + RTT + a compute-bound verify-latency proxy from the
        current_verifier_theta_f_ms_per_token signal (eq. 3's theta_f term;
        this proxy ignores the memory-bound term since it does not depend
        on this client's own gamma)."""
        verify_estimate_ms = self.current_verifier_theta_f_ms_per_token * (gamma + 1)
        return (
            self._draft_time_ms(client, gamma)
            + self._uplink_time_ms(client, gamma)
            + client.rtt_ms
            + verify_estimate_ms
        )

    def _turbospec_goodput(self, client: ClientProfile, gamma: int) -> float:
        profile = client.positional_acceptance_profile(gamma, use_ucb=False)
        expected_tokens = expected_useful_tokens_from_profile(profile) if gamma > 0 else 1.0
        round_time_ms = max(1e-6, self._round_time_estimate_ms(client, gamma))
        return expected_tokens / round_time_ms

    def _turbospec_gamma(self, client: ClientProfile) -> int:
        """ArgMaxGoodput (arXiv:2406.14066): search gamma maximizing
        expected_tokens(gamma)/round_time(gamma), using client.alpha_hat (a
        plain moving-average-like estimate, not the UCB-optimistic
        alpha_ucb) -- TurboSpec makes no optimism/regret guarantee."""
        best_gamma = 0
        best_goodput = self._turbospec_goodput(client, 0)
        for candidate in self.gamma_choices[1:]:
            goodput = self._turbospec_goodput(client, candidate)
            if goodput > best_goodput:
                best_goodput = goodput
                best_gamma = candidate
        return best_gamma

    def _fixed_slo_gamma(self, client: ClientProfile) -> int:
        """Largest gamma whose round-time estimate still meets
        self.fixed_slo_ms (uniform across clients here -- a simplification
        TASKS.md T3.4 explicitly allows; per-client SLOs would just need a
        dict keyed by client_id instead of one scalar)."""
        best_gamma = 0
        for candidate in self.gamma_choices:
            if self._round_time_estimate_ms(client, candidate) <= self.fixed_slo_ms:
                best_gamma = candidate
        return best_gamma

    def _gelato_energy_mj(self, client: ClientProfile, gamma: int) -> float:
        draft_ms = self._draft_time_ms(client, gamma)
        uplink_ms = self._uplink_time_ms(client, gamma)
        return self.gelato_draft_power_mw * draft_ms + self.gelato_uplink_power_mw * uplink_ms

    def _gelato_gamma(self, client: ClientProfile) -> int:
        """Per-device independent drift-plus-penalty (GELATO, arXiv:2605.10124):
        exhaustive search over gamma maximizing
        V*throughput(gamma) - Q_energy*energy(gamma). Energy is tracked in
        relative units (power_mw * time_ms); update the energy virtual
        queue (eq. 7) using the PREVIOUS round's decision before deciding
        this round's gamma, since this controller has no separate call site
        for it (see sim/controllers/gelato.py for the from-scratch version
        with its own explicit update_queue() call)."""
        cid = client.client_id
        last_energy = self._gelato_last_energy_mj.get(cid, 0.0)
        queue = max(0.0, self._gelato_energy_queue.get(cid, 0.0) + last_energy - self.gelato_energy_budget_mj)
        self._gelato_energy_queue[cid] = queue

        best_gamma = 0
        best_utility = self.gelato_v * self._turbospec_goodput(client, 0) - queue * self._gelato_energy_mj(client, 0)
        for candidate in self.gamma_choices[1:]:
            throughput = self._turbospec_goodput(client, candidate)
            energy = self._gelato_energy_mj(client, candidate)
            utility = self.gelato_v * throughput - queue * energy
            if utility > best_utility:
                best_utility = utility
                best_gamma = candidate
        self._gelato_last_energy_mj[cid] = self._gelato_energy_mj(client, best_gamma)
        return best_gamma

    def _load_only_gamma(self) -> int:
        if not self.gamma_choices:
            return 0
        per_client_budget = max(
            0,
            int(self.current_verifier_budget_tokens // max(1, self.current_active_clients)) - 1,
        )
        feasible = [choice for choice in self.gamma_choices if choice <= per_client_budget]
        if feasible:
            return feasible[-1]
        return 0

    def _apply_exploration(self, client: ClientProfile, gamma: int) -> int:
        client.last_forced_exploration = False
        client.last_exploration_reason = "policy"
        if self.policy not in {"adaptive_queue", "adaptive_ucb", "adaptive_index"}:
            return gamma
        if not self._positive_gamma_choices:
            return gamma
        if gamma > 0:
            return gamma
        if client.speculative_rounds < self.min_exploration_rounds:
            client.last_forced_exploration = True
            client.last_exploration_reason = "bootstrap"
            return self.exploration_gamma
        if self.exploration_epsilon > 0.0 and self.rng.random() < self.exploration_epsilon:
            client.last_forced_exploration = True
            client.last_exploration_reason = "epsilon"
            return self.exploration_gamma
        return gamma

    def choose_gamma(self, client: ClientProfile, total_rounds: int) -> int:
        client.refresh_learning_stats(total_rounds)

        if self.policy == "target_only":
            gamma = 0
        elif self.policy == "fixed_gamma":
            gamma = self.fixed_gamma
        elif self.policy == "load_only":
            gamma = self._load_only_gamma()
        elif self.policy == "random_gamma":
            gamma = self.rng.choice(self.gamma_choices)
        elif self.policy == "oracle_gamma":
            gamma = self._argmax_gamma(client, use_ucb=False)
        elif self.policy == "adaptive_queue":
            gamma = self._queue_policy_gamma(client, use_ucb=False)
        elif self.policy == "adaptive_ucb":
            gamma = self._argmax_gamma(client, use_ucb=True)
        elif self.policy == "adaptive_index":
            gamma = self._index_policy_gamma(client, use_ucb=True)
        elif self.policy == "goodspeed":
            gamma = self._goodspeed_gamma(client)
        elif self.policy == "turbospec":
            gamma = self._turbospec_gamma(client)
        elif self.policy == "fixed_slo":
            gamma = self._fixed_slo_gamma(client)
        elif self.policy == "gelato":
            gamma = self._gelato_gamma(client)
        else:
            raise RuntimeError(f"Unsupported controller policy: {self.policy}")
        gamma = self._apply_exploration(client, gamma)
        client.last_selected_gamma = gamma
        return gamma

    def slot_index(self, time_ms: float) -> int:
        return max(0, int(time_ms // self.slot_ms))

    def observe_completion(
        self,
        client: ClientProfile,
        client_receive_ms: float,
        useful_tokens: int,
    ) -> SlotObservation:
        if not self.use_virtual_queues:
            client.z_queue = 0.0
            slot_index = self.slot_index(client_receive_ms)
            slot_totals = self._slot_completions.setdefault(client.client_id, {})
            slot_totals[slot_index] = slot_totals.get(slot_index, 0) + useful_tokens
            return SlotObservation(
                slot_index=slot_index,
                slot_useful_tokens=slot_totals[slot_index],
            )
        slot_index = self.slot_index(client_receive_ms)
        slot_totals = self._slot_completions.setdefault(client.client_id, {})
        slot_totals[slot_index] = slot_totals.get(slot_index, 0) + useful_tokens
        slot_seconds = self.slot_ms / 1000.0

        state = self._slot_state.get(client.client_id)
        if state is None:
            current_slot = slot_index
            slot_base_queue = client.z_queue
            slot_useful_tokens = 0
        else:
            current_slot, slot_base_queue, slot_useful_tokens = state

        if slot_index != current_slot:
            finalized_queue = max(
                0.0,
                slot_base_queue + self.min_tps * slot_seconds - slot_useful_tokens,
            )
            for _ in range(current_slot + 1, slot_index):
                finalized_queue = max(0.0, finalized_queue + self.min_tps * slot_seconds)
            current_slot = slot_index
            slot_base_queue = finalized_queue
            slot_useful_tokens = 0

        slot_useful_tokens += useful_tokens
        client.z_queue = max(
            0.0,
            slot_base_queue + self.min_tps * slot_seconds - slot_useful_tokens,
        )
        self._slot_state[client.client_id] = (
            current_slot,
            slot_base_queue,
            slot_useful_tokens,
        )
        return SlotObservation(
            slot_index=slot_index,
            slot_useful_tokens=slot_totals[slot_index],
        )

    def observe_server_batch(
        self,
        batch_id: int,
        verify_finish_ms: float,
        theta_f_ms_per_token: float,
        verifier_token_cost: int,
    ) -> float:
        if not self.use_virtual_queues:
            self.server_queue = 0.0
            self.server_price = 0.0
            self._seen_server_batches.add(batch_id)
            return self.server_price
        if batch_id in self._seen_server_batches:
            return self.server_price
        self._seen_server_batches.add(batch_id)

        slot_index = self.slot_index(verify_finish_ms)
        slot_budget_ms = self.slot_ms
        state = self._server_slot_state
        if state is None:
            current_slot = slot_index
            slot_base_queue = self.server_queue
            slot_service_ms = 0.0
        else:
            current_slot, slot_base_queue, slot_service_ms = state

        if slot_index != current_slot:
            finalized_queue = max(0.0, slot_base_queue + slot_service_ms - slot_budget_ms)
            for _ in range(current_slot + 1, slot_index):
                finalized_queue = max(0.0, finalized_queue - slot_budget_ms)
            current_slot = slot_index
            slot_base_queue = finalized_queue
            slot_service_ms = 0.0

        gamma_batch_service_ms = max(0.0, float(theta_f_ms_per_token)) * max(
            0,
            int(verifier_token_cost),
        )
        slot_service_ms += gamma_batch_service_ms
        self.server_queue = max(0.0, slot_base_queue + slot_service_ms - slot_budget_ms)
        self._server_slot_state = (current_slot, slot_base_queue, slot_service_ms)
        return self._refresh_server_price()

    def observe_device_service(
        self,
        client_id: int,
        edge_finish_ms: float,
        gamma: int,
        tau_d_ms: float,
        kappa_over_r_ms: float,
    ) -> float:
        if not self.use_virtual_queues:
            return 0.0
        slot_index = self.slot_index(edge_finish_ms)
        slot_budget_ms = self.slot_ms
        state = self._device_slot_state.get(client_id)
        if state is None:
            current_slot = slot_index
            slot_base_queue = 0.0
            slot_service_ms = 0.0
        else:
            current_slot, slot_base_queue, slot_service_ms = state

        if slot_index != current_slot:
            finalized_queue = max(0.0, slot_base_queue + slot_service_ms - slot_budget_ms)
            for _ in range(current_slot + 1, slot_index):
                finalized_queue = max(0.0, finalized_queue - slot_budget_ms)
            current_slot = slot_index
            slot_base_queue = finalized_queue
            slot_service_ms = 0.0

        device_round_service_ms = max(0, int(gamma)) * (
            max(0.0, float(tau_d_ms)) + max(0.0, float(kappa_over_r_ms))
        )
        slot_service_ms += device_round_service_ms
        device_queue = max(0.0, slot_base_queue + slot_service_ms - slot_budget_ms)
        self._device_slot_state[client_id] = (current_slot, slot_base_queue, slot_service_ms)
        return device_queue

    def update(
        self,
        client: ClientProfile,
        useful_tokens: int,
        client_receive_ms: float,
        edge_finish_ms: float,
        gamma: int,
        tau_d_ms: float,
        kappa_over_r_ms: float,
    ) -> SlotObservation:
        client.device_queue = self.observe_device_service(
            client_id=client.client_id,
            edge_finish_ms=edge_finish_ms,
            gamma=gamma,
            tau_d_ms=tau_d_ms,
            kappa_over_r_ms=kappa_over_r_ms,
        )
        if not self.use_virtual_queues:
            client.device_queue = 0.0
        return self.observe_completion(client, client_receive_ms, useful_tokens)
