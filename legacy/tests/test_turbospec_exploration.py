"""controller.turbospec_exploration: warm-up/epsilon exploration for TurboSpec."""
from __future__ import annotations

from tests.test_capacity import _client, _ctrl


def _ts(explore: bool, min_rounds: int = 2, eps: float = 0.0):
    ctrl = _ctrl(V=50.0)
    ctrl.policy = "turbospec"
    ctrl.min_exploration_rounds = min_rounds
    ctrl.exploration_epsilon = eps
    ctrl.exploration_gamma = 1
    ctrl.turbospec_exploration = explore
    return ctrl


def test_default_off_keeps_gamma_zero():
    ctrl = _ts(explore=False)
    cl = _client(alpha=0.9)
    cl.speculative_rounds = 0
    assert ctrl.turbospec_exploration is False
    assert ctrl._apply_exploration(cl, 0) == 0


def test_bootstrap_forces_exploration_gamma():
    ctrl = _ts(explore=True, min_rounds=2)
    cl = _client(alpha=0.9)
    cl.speculative_rounds = 0
    assert ctrl._apply_exploration(cl, 0) == ctrl.exploration_gamma
    assert cl.last_forced_exploration is True


def test_positive_gamma_untouched():
    ctrl = _ts(explore=True, min_rounds=2, eps=1.0)
    cl = _client(alpha=0.9)
    cl.speculative_rounds = 0
    assert ctrl._apply_exploration(cl, 3) == 3


def test_epsilon_after_bootstrap():
    ctrl = _ts(explore=True, min_rounds=0, eps=1.0)
    cl = _client(alpha=0.9)
    cl.speculative_rounds = 10
    assert ctrl._apply_exploration(cl, 0) == ctrl.exploration_gamma
