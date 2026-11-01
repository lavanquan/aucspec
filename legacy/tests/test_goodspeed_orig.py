"""goodspeed_orig follows GoodSpeed (arXiv 2512.09963) Alg. 1: a budget-C
integer allocation of draft lengths maximizing sum_i x_hat_i(S_i)/X_i."""

from __future__ import annotations

import itertools
import types

import pytest

from edge_specsim.controller import OnlineController


def _ctl(budget=6, gamma_max=4):
    c = OnlineController(gamma_choices=list(range(gamma_max + 1)), V=50.0, min_tps=4.5,
                         server_price_scale=0.0, policy="goodspeed_orig")
    c.goodspeed_budget_tokens = budget
    return c


def _objective(alloc, alpha, goodput):
    return sum(((1 - alpha[i] ** (s + 1)) / (1 - alpha[i])) / goodput[i] for i, s in alloc.items())


def test_greedy_allocation_is_optimal():
    c = _ctl(budget=6)
    c._gs_alpha = {0: 0.9, 1: 0.6, 2: 0.8}
    c._gs_goodput = {0: 3.0, 1: 1.0, 2: 2.0}
    alloc = c._goodspeed_orig_allocation([0, 1, 2])
    assert sum(alloc.values()) <= 6
    best = max(
        (dict(zip([0, 1, 2], s)) for s in itertools.product(range(5), repeat=3) if sum(s) <= 6),
        key=lambda a: _objective(a, c._gs_alpha, c._gs_goodput),
    )
    assert _objective(alloc, c._gs_alpha, c._gs_goodput) == pytest.approx(
        _objective(best, c._gs_alpha, c._gs_goodput))


def test_low_goodput_client_gets_longer_drafts():
    c = _ctl(budget=4)
    c._gs_alpha = {0: 0.8, 1: 0.8}
    c._gs_goodput = {0: 1.0, 1: 5.0}
    alloc = c._goodspeed_orig_allocation([0, 1])
    assert alloc[0] > alloc[1]


def test_ema_updates():
    c = _ctl()
    client = types.SimpleNamespace(client_id=0)
    c.observe_goodspeed_orig(client, gamma=4, useful_tokens=3)  # 2 accepted + correction
    assert c._gs_alpha[0] == pytest.approx(0.9 * 0.5 + 0.1 * 0.5)
    assert c._gs_goodput[0] == pytest.approx(3.0)
    c.observe_goodspeed_orig(client, gamma=0, useful_tokens=1)
    assert c._gs_goodput[0] == pytest.approx(2.0)
    assert c._gs_alpha[0] == pytest.approx(0.5)
