import importlib.util
from pathlib import Path

from edge_specsim.capacity import CandidateMeasurement

_spec = importlib.util.spec_from_file_location(
    "certify_x3_report", Path(__file__).resolve().parents[2] / "scripts" / "certify_x3_report.py"
)
rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rep)

X = 3.0


def _m(seed, rate, n=1, exhausted=None):
    return CandidateMeasurement(
        scale=n, n_active=n, x_requirement=X, seed=seed, min_rate_tps=rate,
        per_client_rates={i: rate for i in range(n)},
        extra={"exhausted_clients": exhausted or []},
    )


def _by_seed(rates, **kw):
    return {i + 1: _m(i + 1, r, **kw) for i, r in enumerate(rates)}


def test_feasible_uses_shortest_decisive_prefix():
    d = rep.decide(_by_seed([3.6] * 12), X, 1, 5, 12, 0.05)
    assert d["status"] == "feasible" and d["seeds_used"] == [1, 2, 3, 4, 5]


def test_infeasible():
    d = rep.decide(_by_seed([2.0] * 6), X, 1, 5, 12, 0.05)
    assert d["status"] == "infeasible" and len(d["seeds_used"]) == 5


def test_uncertain_until_more_seeds_then_decisive():
    rates = [2.9, 3.1, 2.95, 3.05, 3.0] + [2.2] * 7
    d = rep.decide(_by_seed(rates), X, 1, 5, 12, 0.05)
    assert d["status"] in ("infeasible", "feasible")
    assert len(d["seeds_used"]) > 5


def test_uncertain_at_max_seeds_is_not_forced():
    rates = [2.9, 3.1, 2.95, 3.05, 3.0, 2.97, 3.03, 3.0, 2.98, 3.02, 3.0, 3.01]
    d = rep.decide(_by_seed(rates), X, 1, 5, 12, 0.05)
    assert d["status"] == "uncertain_at_max_seeds"


def test_fewer_than_initial_seeds_is_incomplete():
    assert rep.decide(_by_seed([3.6] * 4), X, 1, 5, 12, 0.05)["status"] == "incomplete"


def test_gap_in_seed_ids_stops_contiguous_prefix():
    by = _by_seed([3.6] * 6)
    del by[3]
    assert rep.decide(by, X, 1, 5, 12, 0.05)["status"] == "incomplete"


def test_exhausted_client_invalidates_candidate():
    by = _by_seed([3.6] * 5)
    by[2] = _m(2, 3.6, exhausted=[{"client_id": 0, "active_span_s": 100.0}])
    d = rep.decide(by, X, 1, 5, 12, 0.05)
    assert d["status"] == "invalid_exhausted" and d["invalid_seeds"] == [2]


def test_conclude_branches():
    f, i, u = {"status": "feasible"}, {"status": "infeasible"}, {"status": "uncertain_at_max_seeds"}
    assert rep.conclude({10: i, 9: f})[1] == "N*(3.0) = 9, certified"
    assert rep.conclude({10: f, 11: i})[1] == "N*(3.0) = 10, certified"
    assert rep.conclude({10: f, 11: f, 12: i})[1] == "N*(3.0) = 11, certified"
    assert rep.conclude({10: f, 11: f, 12: f})[0] == "anomaly"
    assert rep.conclude({10: u})[0] == "unresolved"
    assert rep.conclude({10: i, 9: u})[0] == "unresolved"
    assert rep.conclude({10: i})[0] == "unresolved"
