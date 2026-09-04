import pytest

from sim.controllers.auc_controller import StreamLinkState
from sim.controllers.goodspeed import GoodSpeedController


def make_controller(n=2) -> GoodSpeedController:
    return GoodSpeedController(
        n_streams=n,
        gamma_max=6,
        kappa_bits_per_token=500.0,
        theta_f_seconds_per_token=0.001,
        total_bandwidth_hz=1.0e6,
        gamma_budget=1000.0,
        alpha_hat=[0.8] * n,
    )


def test_decide_returns_one_gamma_per_stream():
    ctrl = make_controller()
    links = [StreamLinkState(0.01, 1.0e6, 3.0), StreamLinkState(0.01, 1.0e6, 3.0)]
    decision = ctrl.decide(t=0, links=links)
    assert len(decision.gammas) == 2


def test_stream_with_lower_running_x_gets_priced_more_favourably():
    # After update_queues, give stream 0 a much higher observed rate than
    # stream 1; stream 1's weight (1/x_hat) becomes larger, so it should be
    # speculated at least as aggressively as stream 0 going forward.
    ctrl = make_controller()
    links = [StreamLinkState(0.01, 1.0e6, 3.0), StreamLinkState(0.01, 1.0e6, 3.0)]
    decision = ctrl.decide(t=0, links=links)
    ctrl.update_queues(decision, tokens_delivered=[50.0, 1.0], links=links, verification_batch_tokens=10.0, slot_seconds=1.0)
    next_decision = ctrl.decide(t=1, links=links)
    assert next_decision.gammas[1] >= next_decision.gammas[0]


def test_bandwidth_sums_to_budget():
    ctrl = make_controller()
    links = [StreamLinkState(0.01, 1.0e6, 3.0), StreamLinkState(0.01, 1.0e6, 3.0)]
    decision = ctrl.decide(t=0, links=links)
    assert sum(decision.bandwidth_allocation_hz) == pytest.approx(1.0e6)
