import pytest

from sim.controllers.auc_controller import StreamLinkState
from sim.controllers.max_throughput import MaxThroughputController


def test_every_stream_gets_the_same_fixed_gamma():
    ctrl = MaxThroughputController(n_streams=4, fixed_gamma=1, gamma_budget=100.0, total_bandwidth_hz=1.0e6)
    links = [StreamLinkState(0.01, 1.0e6, 3.0)] * 4
    decision = ctrl.decide(t=0, links=links)
    assert decision.gammas == [1, 1, 1, 1]


def test_admits_largest_batch_the_budget_allows():
    # gamma=1 -> cost 2 per stream; budget 5 fits 2 streams, not 3.
    ctrl = MaxThroughputController(n_streams=4, fixed_gamma=1, gamma_budget=5.0, total_bandwidth_hz=1.0e6)
    links = [StreamLinkState(0.01, 1.0e6, 3.0)] * 4
    decision = ctrl.decide(t=0, links=links)
    assert decision.selected_batch == [0, 1]


def test_bandwidth_split_evenly_among_admitted():
    ctrl = MaxThroughputController(n_streams=2, fixed_gamma=0, gamma_budget=100.0, total_bandwidth_hz=100.0)
    links = [StreamLinkState(0.01, 1.0e6, 3.0)] * 2
    decision = ctrl.decide(t=0, links=links)
    assert decision.bandwidth_allocation_hz == [50.0, 50.0]


def test_rejects_negative_fixed_gamma():
    with pytest.raises(ValueError):
        MaxThroughputController(n_streams=2, fixed_gamma=-1, gamma_budget=10.0, total_bandwidth_hz=1.0)
