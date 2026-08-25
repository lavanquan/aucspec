from __future__ import annotations

from collections import deque

from edge_specsim.channel import current_snr_db
from edge_specsim.models import ClientProfile


def _make_client(channel_model: str = "iid_block_fading") -> ClientProfile:
    return ClientProfile(
        client_id=3,
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
        downlink_snr_jitter_db=2.0,
        snr_period_ms=1000.0,
        snr_phase_rad=0.0,
        wireless_weight=1.0,
        packet_loss=0.0,
        channel_model=channel_model,
        snr_block_duration_ms=100.0,
        channel_seed=42,
    )


def test_iid_block_fading_is_constant_within_block() -> None:
    client = _make_client("iid_block_fading")

    snr_start = current_snr_db(client, "uplink", 10.0)
    snr_middle = current_snr_db(client, "uplink", 90.0)

    assert snr_start == snr_middle


def test_iid_block_fading_resamples_across_blocks_and_is_deterministic() -> None:
    client = _make_client("iid_block_fading")

    block0 = current_snr_db(client, "downlink", 10.0)
    block1 = current_snr_db(client, "downlink", 110.0)
    block1_repeat = current_snr_db(client, "downlink", 110.0)

    assert block0 != block1
    assert block1 == block1_repeat


def test_sinusoidal_model_varies_with_time_inside_period() -> None:
    client = _make_client("sinusoidal")

    snr_a = current_snr_db(client, "uplink", 0.0)
    snr_b = current_snr_db(client, "uplink", 250.0)

    assert snr_a != snr_b
