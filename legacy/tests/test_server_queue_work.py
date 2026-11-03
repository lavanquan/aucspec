"""Q_s under continuous dispatch: charging the whole batch wall time keeps
the verifier at ~100% load so Q_s drifts up; charging theta_f x tokens
(server_queue_work=theta_f) drains when the marginal work fits the slot."""
from __future__ import annotations

import random

from tests.test_capacity import _ctrl


def _run(theta_per_token, tokens, period_ms, n=2000, jitter=0.15, seed=0):
    rng = random.Random(seed)
    ctrl = _ctrl()
    t = 0.0
    qs = []
    for b in range(n):
        dur = period_ms * (1.0 + rng.uniform(-jitter, jitter))
        t += dur
        theta = theta_per_token if theta_per_token is not None else dur / tokens
        ctrl.observe_server_batch(b, t, theta, tokens)
        qs.append(ctrl.server_queue)
    return qs


def test_wall_work_never_drains_a_cold_start_spike():
    # back-to-back batches charged their wall time -> load exactly 1.0, so a
    # cold-start 900 ms batch leaves Q_s ~800 for the whole run
    ctrl = _ctrl()
    ctrl.observe_server_batch(0, 900.0, 900.0 / 14, 14)
    t = 900.0
    for b in range(1, 2000):
        t += 77.0
        ctrl.observe_server_batch(b, t, 77.0 / 14, 14)
    assert ctrl.server_queue > 700.0


def test_theta_f_work_drains_a_cold_start_spike():
    ctrl = _ctrl()
    ctrl.observe_server_batch(0, 900.0, 900.0 / 14, 14)
    t = 900.0
    for b in range(1, 2000):
        t += 77.0
        ctrl.observe_server_batch(b, t, 3.83, 14)
    assert ctrl.server_queue < 60.0


def test_theta_f_work_drains_when_marginal_work_fits():
    # 14 tokens x 3.83 = 54 ms of work per 77 ms batch -> load 0.7
    qs = _run(3.83, tokens=14, period_ms=77.0)
    assert max(qs[-200:]) < 100.0


def test_theta_f_work_grows_when_overloaded():
    # 68 tokens x 3.83 = 260 ms of work per 222 ms batch -> load 1.18
    qs = _run(3.83, tokens=68, period_ms=222.0)
    assert qs[-1] > qs[len(qs) // 2] > 1000.0
