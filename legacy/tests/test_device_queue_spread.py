"""Device virtual queue: "spread" charges each slot only the drafting time it
contains, so one long draft does not look like an overloaded device."""

from __future__ import annotations

import pytest

from edge_specsim.controller import OnlineController


def _ctl(mode: str) -> OnlineController:
    c = OnlineController(gamma_choices=[0, 4, 8], V=50.0, min_tps=4.5, server_price_scale=0.0,
                         policy="capacity_dpp", slot_ms=100.0)
    c.device_service_accounting = mode
    return c


def _round(c, finish_ms, gamma=8, tau=30.0):
    return c.observe_device_service(client_id=0, edge_finish_ms=finish_ms, gamma=gamma,
                                    tau_d_ms=tau, kappa_over_r_ms=0.0)


def test_lumped_spikes_after_one_long_draft():
    # 8 x 30 ms = 240 ms of drafting ending at t=350 ms, all charged to slot 3
    assert _round(_ctl("lumped"), 350.0) == pytest.approx(140.0)


def test_spread_device_never_over_capacity():
    c = _ctl("spread")
    assert _round(c, 350.0) == pytest.approx(0.0)
    # back-to-back drafting (device 100% busy) still keeps Q_i at 0
    assert _round(c, 590.0) == pytest.approx(0.0)


def test_spread_still_counts_overload():
    # 300 ms of service claimed inside [250, 350] cannot fit; spread charges
    # only the overlap, so a real overload shows up via repeated rounds
    c = _ctl("spread")
    q = 0.0
    for k in range(10):
        q = _round(c, 100.0 + 50.0 * k, gamma=4, tau=25.0)  # 100 ms of work every 50 ms
    assert q > 0.0
