from __future__ import annotations

import math

import pandas as pd

from edge_specsim.metrics import (
    compute_client_interactivity,
    compute_sample_interactivity,
    compute_system_metrics,
    proportional_fairness_utility,
)


def test_sample_interactivity_uses_end_to_end_virtual_time() -> None:
    df = pd.DataFrame(
        [
            {
                "client_id": 0,
                "sample_id": "s0",
                "useful_tokens": 2,
                "accepted_length": 1,
                "round_id": 0,
                "draft_start_ms": 100.0,
                "sample_start_time_ms": 100.0,
                "client_receive_ms": 220.0,
            },
            {
                "client_id": 0,
                "sample_id": "s0",
                "useful_tokens": 1,
                "accepted_length": 1,
                "round_id": 1,
                "draft_start_ms": 220.0,
                "sample_start_time_ms": 100.0,
                "client_receive_ms": 340.0,
            },
        ]
    )

    per_sample = compute_sample_interactivity(df)

    assert len(per_sample) == 1
    assert per_sample.loc[0, "useful_tokens"] == 3
    assert per_sample.loc[0, "e2e_ms"] == 240.0
    assert math.isclose(per_sample.loc[0, "interactivity_s_per_token"], 0.08)


def test_client_interactivity_sums_sample_end_to_end_time() -> None:
    df = pd.DataFrame(
        [
            {
                "client_id": 1,
                "sample_id": "a",
                "useful_tokens": 2,
                "accepted_length": 1,
                "round_id": 0,
                "draft_start_ms": 0.0,
                "sample_start_time_ms": 0.0,
                "client_receive_ms": 100.0,
            },
            {
                "client_id": 1,
                "sample_id": "b",
                "useful_tokens": 3,
                "accepted_length": 2,
                "round_id": 0,
                "draft_start_ms": 150.0,
                "sample_start_time_ms": 150.0,
                "client_receive_ms": 450.0,
            },
        ]
    )

    per_client = compute_client_interactivity(df)

    assert len(per_client) == 1
    assert per_client.loc[0, "useful_tokens"] == 5
    assert per_client.loc[0, "active_e2e_ms"] == 400.0
    assert math.isclose(per_client.loc[0, "interactivity_s_per_token"], 0.08)


def test_system_metrics_include_goodput_ttft_and_utilization() -> None:
    df = pd.DataFrame(
        [
            {
                "client_id": 0,
                "sample_id": "s0",
                "useful_tokens": 2,
                "accepted_length": 1,
                "gamma": 1,
                "round_id": 0,
                "draft_start_ms": 0.0,
                "sample_start_time_ms": 0.0,
                "client_receive_ms": 100.0,
                "round_latency_ms": 100.0,
                "virtual_system_time_ms": 300.0,
                "verification_batch_id": 10,
                "verification_batch_service_ms": 40.0,
                "edge_service_ms": 30.0,
                "worker_id": 0,
            },
            {
                "client_id": 0,
                "sample_id": "s0",
                "useful_tokens": 1,
                "accepted_length": 1,
                "gamma": 1,
                "round_id": 1,
                "draft_start_ms": 100.0,
                "sample_start_time_ms": 0.0,
                "client_receive_ms": 300.0,
                "round_latency_ms": 200.0,
                "virtual_system_time_ms": 300.0,
                "verification_batch_id": 11,
                "verification_batch_service_ms": 60.0,
                "edge_service_ms": 50.0,
                "worker_id": 0,
            },
        ]
    )

    summary = compute_system_metrics(df)

    assert math.isclose(summary["goodput_tps"], 10.0)
    assert math.isclose(summary["accepted_token_goodput_tps"], 6.6666666667, rel_tol=1e-6)
    assert math.isclose(summary["mean_ttft_ms"], 100.0)
    assert math.isclose(summary["mean_inter_token_latency_ms"], 100.0)
    assert math.isclose(summary["server_utilization"], 100.0 / 300.0)
    assert math.isclose(summary["draft_gpu_utilization_mean"], 80.0 / 300.0)
    assert math.isclose(summary["acceptance_rate"], 1.0)
    assert math.isclose(summary["straggler_ratio"], 1.0)
    assert math.isclose(summary["min_client_interactivity_s_per_token"], 0.1)
    assert math.isclose(summary["max_client_interactivity_s_per_token"], 0.1)
    assert math.isclose(summary["min_client_service_rate_tps"], 10.0)
    assert math.isclose(summary["max_client_service_rate_tps"], 10.0)
    assert math.isclose(summary["interactivity_jain_fairness"], 1.0)
    assert math.isclose(
        summary["interactivity_proportional_fairness_utility"],
        proportional_fairness_utility([0.1]),
    )


def test_fairness_metrics_track_interactivity_across_clients() -> None:
    df = pd.DataFrame(
        [
            {
                "client_id": 0,
                "sample_id": "a",
                "useful_tokens": 2,
                "accepted_length": 1,
                "gamma": 1,
                "round_id": 0,
                "draft_start_ms": 0.0,
                "sample_start_time_ms": 0.0,
                "client_receive_ms": 100.0,
                "round_latency_ms": 100.0,
                "virtual_system_time_ms": 400.0,
                "verification_batch_id": 1,
                "verification_batch_service_ms": 50.0,
                "edge_service_ms": 25.0,
                "worker_id": 0,
            },
            {
                "client_id": 1,
                "sample_id": "b",
                "useful_tokens": 2,
                "accepted_length": 1,
                "gamma": 1,
                "round_id": 0,
                "draft_start_ms": 0.0,
                "sample_start_time_ms": 0.0,
                "client_receive_ms": 200.0,
                "round_latency_ms": 200.0,
                "virtual_system_time_ms": 400.0,
                "verification_batch_id": 2,
                "verification_batch_service_ms": 50.0,
                "edge_service_ms": 25.0,
                "worker_id": 1,
            },
        ]
    )

    summary = compute_system_metrics(df)

    expected_x = [0.05, 0.1]
    expected_service_rates = [20.0, 10.0]
    expected_jain = (sum(expected_x) ** 2) / (len(expected_x) * sum(x * x for x in expected_x))
    assert math.isclose(summary["min_client_interactivity_s_per_token"], 0.05)
    assert math.isclose(summary["max_client_interactivity_s_per_token"], 0.1)
    assert math.isclose(summary["min_client_service_rate_tps"], min(expected_service_rates))
    assert math.isclose(summary["max_client_service_rate_tps"], max(expected_service_rates))
    assert math.isclose(summary["interactivity_jain_fairness"], expected_jain)
    assert math.isclose(
        summary["interactivity_proportional_fairness_utility"],
        proportional_fairness_utility(expected_x),
    )
