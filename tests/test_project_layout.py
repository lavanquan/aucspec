import qwen_edge_specsim
import qwen_edge_specsim.clients as clients
import qwen_edge_specsim.controller as controller
import qwen_edge_specsim.datasets as datasets
import qwen_edge_specsim.metrics as metrics
import qwen_edge_specsim.network as network
import qwen_edge_specsim.simulation as simulation
import qwen_edge_specsim.target as target


def test_layout_imports_exist() -> None:
    assert "EdgeSpecSimulator" in qwen_edge_specsim.__all__
    assert clients.ClientProfile is not None
    assert clients.DraftWorker is not None
    assert clients.VerificationResult is not None
    assert controller.OnlineController is not None
    assert datasets.load_samples is not None
    assert metrics.summarize_round_csv is not None
    assert metrics.trapezoidal_auc is not None
    assert metrics.upper_concave_hull is not None
    assert network.estimate_uplink_bytes_from_gamma(3) > 0
    assert target.confirmed_prefix_hash([1, 2, 3]) == hash((1, 2, 3))
    assert simulation.SimulationEventLoop is not None
    event = network.NetworkEventTimestamps(1, 2, 3, 4, 5, 6, 7, 8)
    assert event.client_receive_ms == 8
