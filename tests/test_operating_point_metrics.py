from __future__ import annotations

import math

import pandas as pd

from edge_specsim.metrics import measurement_complete_samples, measurement_rounds, summarize_round_csv


def test_measurement_window_filters_rounds_and_complete_samples() -> None:
    df = pd.DataFrame(
        [
            {
                "client_id": 0,
                "sample_id": "warmup",
                "useful_tokens": 1,
                "accepted_length": 1,
                "gamma": 1,
                "round_id": 0,
                "draft_start_ms": 0.0,
                "sample_start_time_ms": 0.0,
                "client_receive_ms": 20_000.0,
                "round_latency_ms": 20_000.0,
                "virtual_system_time_ms": 20_000.0,
                "verification_batch_id": 1,
                "verification_batch_service_ms": 100.0,
                "edge_service_ms": 50.0,
                "worker_id": 0,
                "experiment_warmup_end_ms": 30_000.0,
                "experiment_measurement_end_ms": 330_000.0,
                "in_measurement_window": False,
            },
            {
                "client_id": 0,
                "sample_id": "measured",
                "useful_tokens": 2,
                "accepted_length": 1,
                "gamma": 1,
                "round_id": 0,
                "draft_start_ms": 40_000.0,
                "sample_start_time_ms": 40_000.0,
                "client_receive_ms": 80_000.0,
                "round_latency_ms": 40_000.0,
                "virtual_system_time_ms": 80_000.0,
                "verification_batch_id": 2,
                "verification_batch_service_ms": 120.0,
                "edge_service_ms": 60.0,
                "worker_id": 0,
                "experiment_warmup_end_ms": 30_000.0,
                "experiment_measurement_end_ms": 330_000.0,
                "in_measurement_window": True,
            },
            {
                "client_id": 1,
                "sample_id": "crosses_end",
                "useful_tokens": 2,
                "accepted_length": 1,
                "gamma": 1,
                "round_id": 0,
                "draft_start_ms": 320_000.0,
                "sample_start_time_ms": 320_000.0,
                "client_receive_ms": 340_000.0,
                "round_latency_ms": 20_000.0,
                "virtual_system_time_ms": 340_000.0,
                "verification_batch_id": 3,
                "verification_batch_service_ms": 80.0,
                "edge_service_ms": 40.0,
                "worker_id": 1,
                "experiment_warmup_end_ms": 30_000.0,
                "experiment_measurement_end_ms": 330_000.0,
                "in_measurement_window": False,
            },
        ]
    )

    measured_rounds = measurement_rounds(df)
    measured_samples = measurement_complete_samples(df)
    summary = summarize_round_csv(df)

    assert len(measured_rounds) == 1
    assert measured_rounds.iloc[0]["sample_id"] == "measured"
    assert len(measured_samples) == 1
    assert measured_samples.iloc[0]["sample_id"] == "measured"
    assert summary["measurement_useful_rounds"] == 1
    assert summary["measurement_complete_samples"] == 1
    assert math.isclose(summary["goodput_tps"], 2.0 / 300.0)
