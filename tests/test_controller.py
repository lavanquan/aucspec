from __future__ import annotations

from collections import deque

from edge_specsim.controller import OnlineController, paper_index_gamma
from edge_specsim.models import ClientProfile


def _make_client() -> ClientProfile:
    return ClientProfile(
        client_id=0,
        prompt_queue=deque(),
        draft_model_id="draft",
        draft_device_class="medium",
        draft_tokens_per_second=32.0,
        draft_fixed_latency_ms=8.0,
        rtt_ms=40.0,
        uplink_mbps=20.0,
        downlink_mbps=40.0,
        uplink_base_snr_db=12.0,
        downlink_base_snr_db=14.0,
        uplink_snr_jitter_db=1.5,
        downlink_snr_jitter_db=1.5,
        snr_period_ms=1000.0,
        snr_phase_rad=0.0,
        wireless_weight=1.0,
        packet_loss=0.0,
    )


def test_adaptive_ucb_forces_positive_gamma_during_bootstrap() -> None:
    client = _make_client()
    controller = OnlineController(
        gamma_choices=[0, 1, 2],
        V=0.0,
        min_tps=1.0,
        server_price_scale=0.0,
        verifier_price_lambda=1.0,
        draft_latency_weight=0.0,
        uplink_latency_weight=0.0,
        policy="adaptive_ucb",
        fixed_gamma=0,
        seed=0,
        slot_ms=100.0,
        min_exploration_rounds=3,
        exploration_gamma=1,
        exploration_epsilon=0.0,
    )
    controller.server_queue = 1e6
    controller.refresh_server_price()

    gamma = controller.choose_gamma(client, total_rounds=1)

    assert gamma == 1
    assert client.last_forced_exploration is True
    assert client.last_exploration_reason == "bootstrap"


def test_adaptive_ucb_can_stay_target_only_after_bootstrap_if_policy_prefers_zero() -> None:
    client = _make_client()
    client.speculative_rounds = 3
    controller = OnlineController(
        gamma_choices=[0, 1, 2],
        V=0.0,
        min_tps=1.0,
        server_price_scale=0.0,
        verifier_price_lambda=1.0,
        draft_latency_weight=0.0,
        uplink_latency_weight=0.0,
        policy="adaptive_ucb",
        fixed_gamma=0,
        seed=0,
        slot_ms=100.0,
        min_exploration_rounds=3,
        exploration_gamma=1,
        exploration_epsilon=0.0,
    )
    controller.server_queue = 1e6
    controller.refresh_server_price()

    gamma = controller.choose_gamma(client, total_rounds=10)

    assert gamma == 0
    assert client.last_forced_exploration is False
    assert client.last_exploration_reason == "policy"


def test_virtual_queues_can_be_disabled() -> None:
    client = _make_client()
    controller = OnlineController(
        gamma_choices=[0, 1, 2],
        V=10.0,
        min_tps=2.0,
        server_price_scale=0.0,
        policy="adaptive_queue",
        use_virtual_queues=False,
    )

    server_queue = controller.observe_server_batch(
        batch_id=7,
        verify_finish_ms=50.0,
        theta_f_ms_per_token=20.0,
        verifier_token_cost=4,
    )
    slot = controller.update(
        client,
        useful_tokens=3,
        client_receive_ms=120.0,
        edge_finish_ms=90.0,
        gamma=2,
        tau_d_ms=20.0,
        kappa_over_r_ms=15.0,
    )

    assert server_queue == 0.0
    assert controller.server_queue == 0.0
    assert controller.server_price == 0.0
    assert client.z_queue == 0.0
    assert client.device_queue == 0.0
    assert slot.slot_index == 1
    assert slot.slot_useful_tokens == 3


def test_adaptive_index_prefers_larger_gamma_when_prices_are_low() -> None:
    client = _make_client()
    client.alpha_ucb = 0.8
    controller = OnlineController(
        gamma_choices=[0, 1, 2, 4],
        V=100.0,
        min_tps=1.0,
        server_price_scale=0.02,
        policy="adaptive_index",
        use_virtual_queues=True,
    )
    controller.server_queue = 0.0
    controller.refresh_server_price()
    client.z_queue = 0.0
    client.device_queue = 0.0

    gamma = controller.choose_gamma(client, total_rounds=10)

    assert gamma == 4


def test_adaptive_index_prefers_zero_gamma_when_prices_are_high() -> None:
    client = _make_client()
    client.alpha_ucb = 0.8
    controller = OnlineController(
        gamma_choices=[0, 1, 2, 4],
        V=1.0,
        min_tps=1.0,
        server_price_scale=0.02,
        policy="adaptive_index",
        use_virtual_queues=True,
    )
    controller.server_queue = 1e6
    controller.refresh_server_price()
    client.device_queue = 1e6

    gamma = controller.choose_gamma(client, total_rounds=10)

    assert gamma == 0


def test_paper_index_gamma_matches_closed_form_clamping() -> None:
    assert paper_index_gamma(alpha_hat=0.8, c_value=0.0, gamma_max=8) == 8
    assert paper_index_gamma(alpha_hat=0.8, c_value=1.2, gamma_max=8) == 0
    assert paper_index_gamma(alpha_hat=0.8, c_value=0.8**3, gamma_max=8) == 2


def test_adaptive_index_depends_on_lambda_and_mu_prices() -> None:
    client = _make_client()
    client.alpha_ucb = 0.8
    controller = OnlineController(
        gamma_choices=[0, 1, 2, 4],
        V=10.0,
        min_tps=1.0,
        server_price_scale=1.0,
        policy="adaptive_index",
        use_virtual_queues=True,
    )
    controller.server_queue = 1.0
    controller.refresh_server_price()
    client.device_queue = 0.0
    gamma_low_price = controller.choose_gamma(client, total_rounds=10)

    controller.server_queue = 100.0
    controller.refresh_server_price()
    client.device_queue = 0.0
    gamma_high_lambda = controller.choose_gamma(client, total_rounds=10)

    controller.server_queue = 1.0
    controller.refresh_server_price()
    client.device_queue = 100.0
    gamma_high_mu = controller.choose_gamma(client, total_rounds=10)

    assert gamma_low_price >= gamma_high_lambda
    assert gamma_low_price >= gamma_high_mu


def test_verifier_knee_signal_no_longer_changes_lambda_price_directly() -> None:
    client = _make_client()
    client.alpha_ucb = 0.8
    controller = OnlineController(
        gamma_choices=[0, 1, 2, 4],
        V=10.0,
        min_tps=1.0,
        server_price_scale=1.0,
        policy="adaptive_index",
        use_virtual_queues=True,
    )
    controller.server_queue = 10.0
    client.device_queue = 0.0
    controller.set_verifier_budget_signal(64)
    price_large_knee = controller.refresh_server_price()
    gamma_large_knee = controller.choose_gamma(client, total_rounds=10)

    controller.set_verifier_budget_signal(2)
    controller.set_verifier_regime_signal("compute_bound")
    price_small_knee = controller.refresh_server_price()
    gamma_small_knee = controller.choose_gamma(client, total_rounds=10)

    assert price_small_knee == price_large_knee
    assert gamma_large_knee == gamma_small_knee


def test_adaptive_index_uses_theta_f_signal_in_lambda_term() -> None:
    client = _make_client()
    client.alpha_ucb = 0.8
    controller = OnlineController(
        gamma_choices=[0, 1, 2, 4],
        V=10.0,
        min_tps=1.0,
        server_price_scale=1.0,
        policy="adaptive_index",
        use_virtual_queues=True,
    )
    controller.server_queue = 10.0
    controller.refresh_server_price()
    client.device_queue = 0.0

    controller.set_verifier_theta_f_signal(0.1)
    gamma_low_theta_f = controller.choose_gamma(client, total_rounds=10)

    controller.set_verifier_theta_f_signal(100.0)
    gamma_high_theta_f = controller.choose_gamma(client, total_rounds=10)

    assert gamma_low_theta_f >= gamma_high_theta_f


def test_server_queue_matches_theta_f_gamma_minus_slot_budget() -> None:
    controller = OnlineController(
        gamma_choices=[0, 1],
        V=1.0,
        min_tps=1.0,
        server_price_scale=1.0,
        policy="adaptive_index",
        use_virtual_queues=True,
        slot_ms=100.0,
    )

    server_queue = controller.observe_server_batch(
        batch_id=1,
        verify_finish_ms=50.0,
        theta_f_ms_per_token=10.0,
        verifier_token_cost=18,
    )

    assert controller.server_queue == 80.0
    assert server_queue == 80.0


def test_device_queue_matches_gamma_tau_plus_kappa_over_r_minus_slot_budget() -> None:
    controller = OnlineController(
        gamma_choices=[0, 1],
        V=1.0,
        min_tps=1.0,
        server_price_scale=1.0,
        policy="adaptive_index",
        use_virtual_queues=True,
        slot_ms=100.0,
    )

    device_queue = controller.observe_device_service(
        client_id=0,
        edge_finish_ms=50.0,
        gamma=4,
        tau_d_ms=20.0,
        kappa_over_r_ms=15.0,
    )

    assert device_queue == 40.0
