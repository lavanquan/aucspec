from __future__ import annotations

from collections import deque

from edge_specsim.models import ClientProfile


def _make_client(profile_mode: str = "position") -> ClientProfile:
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
        alpha_profile_mode=profile_mode,
    )


def test_prefix_level_profile_repeats_scalar_alpha() -> None:
    client = _make_client(profile_mode="prefix")
    client.alpha_hat = 0.7
    client.alpha_ucb = 0.9
    client.alpha_position_hat = {1: 0.2, 2: 0.1, 3: 0.05}
    client.alpha_position_ucb = {1: 0.3, 2: 0.2, 3: 0.1}

    assert client.positional_acceptance_profile(3, use_ucb=False) == [0.7, 0.7, 0.7]
    assert client.positional_acceptance_profile(3, use_ucb=True) == [0.9, 0.9, 0.9]


def test_position_level_profile_uses_per_position_tables() -> None:
    client = _make_client(profile_mode="position")
    client.alpha_position_hat = {1: 0.8, 2: 0.6, 3: 0.4}
    client.alpha_position_ucb = {1: 0.85, 2: 0.65, 3: 0.45}

    assert client.positional_acceptance_profile(3, use_ucb=False) == [0.8, 0.6, 0.4]
    assert client.positional_acceptance_profile(3, use_ucb=True) == [0.85, 0.65, 0.45]


def test_censored_feedback_counts_successes_without_spurious_failure() -> None:
    client = _make_client(profile_mode="position")
    client.alpha_learning_mode = "bernstein_censored"

    client.observe_acceptance_feedback(
        proposed_tokens=4,
        accepted_length=4,
        inspected_tokens=4,
    )
    client.refresh_learning_stats(total_rounds=1)

    assert client.alpha_successes == 4.0
    assert client.alpha_failures == 0.0
    assert client.alpha_censored_rounds == 1
    assert client.alpha_hat > 0.99
    assert client.alpha_ucb >= client.alpha_hat


def test_rejection_feedback_adds_single_failure_at_first_rejected_position() -> None:
    client = _make_client(profile_mode="position")
    client.alpha_learning_mode = "bernstein_censored"

    client.observe_acceptance_feedback(
        proposed_tokens=4,
        accepted_length=2,
        inspected_tokens=3,
    )
    client.refresh_learning_stats(total_rounds=5)

    assert client.alpha_successes == 2.0
    assert client.alpha_failures == 1.0
    assert client.alpha_failure_events == 1
    assert client.alpha_position_successes[1] == 1.0
    assert client.alpha_position_successes[2] == 1.0
    assert client.alpha_position_failures[3] == 1.0
    assert client.alpha_position_ucb[3] >= client.alpha_position_hat[3]
