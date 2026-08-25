from __future__ import annotations

import math
import random

from .models import ClientProfile


def spectral_efficiency_from_snr_db(snr_db: float) -> float:
    snr_linear = 10 ** (snr_db / 10.0)
    return max(0.1, math.log2(1.0 + snr_linear))


def block_fading_snr_db(client: ClientProfile, direction: str, time_ms: float) -> float:
    base_snr_db = (
        client.uplink_base_snr_db
        if direction == "uplink"
        else client.downlink_base_snr_db
    )
    jitter_db = (
        client.uplink_snr_jitter_db
        if direction == "uplink"
        else client.downlink_snr_jitter_db
    )
    block_duration_ms = max(1.0, client.snr_block_duration_ms)
    block_index = int(max(0.0, float(time_ms)) // block_duration_ms)
    direction_offset = 0 if direction == "uplink" else 1
    seed = (
        int(client.channel_seed)
        + 10_007 * int(client.client_id)
        + 1_000_003 * direction_offset
        + 17_179_869 * block_index
    )
    rng = random.Random(seed)
    return base_snr_db + rng.uniform(-jitter_db, jitter_db)


def sinusoidal_snr_db(client: ClientProfile, direction: str, time_ms: float) -> float:
    base_snr_db = (
        client.uplink_base_snr_db
        if direction == "uplink"
        else client.downlink_base_snr_db
    )
    jitter_db = (
        client.uplink_snr_jitter_db
        if direction == "uplink"
        else client.downlink_snr_jitter_db
    )
    period_ms = max(1.0, client.snr_period_ms)
    phase = (2.0 * math.pi * time_ms / period_ms) + client.snr_phase_rad
    return base_snr_db + jitter_db * math.sin(phase)


def current_snr_db(client: ClientProfile, direction: str, time_ms: float) -> float:
    channel_model = str(client.channel_model).strip().lower()
    if channel_model == "iid_block_fading":
        return block_fading_snr_db(client, direction, time_ms)
    if channel_model == "sinusoidal":
        return sinusoidal_snr_db(client, direction, time_ms)
    raise ValueError(
        "clients.channel_model must be one of: iid_block_fading, sinusoidal"
    )
