"""Shannon uplink rate and square-root waterfilling bandwidth allocation.

See docs/notation.md eq. (1) and CLAUDE.md section 2, muc 1.6-iii. TASKS.md T1.1.
"""

from __future__ import annotations

import math
from typing import Sequence


def snr_linear_from_db(snr_db: float) -> float:
    """Convert SNR in dB to linear scale."""
    return 10.0 ** (float(snr_db) / 10.0)


def shannon_rate(bandwidth_hz: float, snr_linear: float) -> float:
    """r_i(t) = w_i(t) * log2(1 + SNR_i(t)), eq. (1).

    bandwidth_hz is the bandwidth w_i(t) allocated to stream i (Hz);
    snr_linear is SNR_i(t) on a linear (not dB) scale. Returns bits/s.
    """
    if bandwidth_hz < 0.0:
        raise ValueError("bandwidth_hz must be non-negative")
    if snr_linear < 0.0:
        raise ValueError("snr_linear must be non-negative")
    if bandwidth_hz == 0.0:
        return 0.0
    return float(bandwidth_hz) * math.log2(1.0 + float(snr_linear))


def spectral_efficiency(snr_linear: float) -> float:
    """g_i(t) = log2(1 + SNR_i(t)), bits/s per Hz -- the per-Hz rate used by
    the waterfilling allocation below (eq. 1 with w_i(t) factored out)."""
    if snr_linear < 0.0:
        raise ValueError("snr_linear must be non-negative")
    return math.log2(1.0 + float(snr_linear))


def sqrt_waterfilling(costs: Sequence[float], total_bandwidth: float) -> list[float]:
    """Square-root waterfilling bandwidth allocation (Cauchy-Schwarz), muc 1.6(iii).

    Given per-stream cost c_i = mu_i * kappa * gamma_i / g_i (time-price times
    bits-per-token divided by spectral efficiency), the allocation minimizing
    sum_i mu_i * kappa * gamma_i / (w_i * g_i) subject to sum_i w_i <= W is

        w_i* = W * sqrt(c_i) / sum_j sqrt(c_j).

    Streams with cost 0 (e.g. gamma_i = 0, nothing to send) receive 0
    bandwidth. If every cost is 0, bandwidth is split evenly so the return
    value is always well defined.
    """
    if total_bandwidth < 0.0:
        raise ValueError("total_bandwidth must be non-negative")
    n = len(costs)
    if n == 0:
        return []
    sqrt_costs = [math.sqrt(max(0.0, float(c))) for c in costs]
    denom = sum(sqrt_costs)
    if denom <= 0.0:
        even_share = float(total_bandwidth) / n
        return [even_share] * n
    return [float(total_bandwidth) * sc / denom for sc in sqrt_costs]
