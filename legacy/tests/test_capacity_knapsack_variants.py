from types import SimpleNamespace

from edge_specsim.verification_queue import VerificationQueue, VerificationRequest


def _req(cid, t, gamma, weight, qs, theta):
    return VerificationRequest(
        client_id=cid, arrival_time=t, gamma=gamma, candidate_token_ids=[0] * gamma,
        weight=weight, deadline=None, expected_acceptance_rate=0.7,
        expected_acceptance_profile=[0.7] * max(1, gamma), context_token_ids=[1],
        draft=SimpleNamespace(), remaining_tokens=100, future=None,
        server_queue_price=qs, theta_f_ms_per_token=theta,
    )


def _queue(reqs, **kw):
    q = VerificationQueue()
    for k, v in kw.items():
        setattr(q, k, v)
    q.extend(reqs)
    return q


def _pop(q):
    return q.pop_batch("capacity_knapsack", 8, 1e9, 64, 0.0)


def test_default_starves_when_all_values_negative():
    reqs = [_req(i, float(i), 4, 50.0, 300.0, 3.83) for i in range(4)]
    assert all(r.capacity_dpp_value < 0 for r in reqs)
    q = _queue(reqs)
    batch = _pop(q)
    assert len(batch) == 1 and q.last_batch_forced_service


def test_work_conserving_serves_full_batch():
    reqs = [_req(i, float(i), 4, 50.0, 300.0, 3.83) for i in range(4)]
    q = _queue(reqs, capacity_work_conserving=True)
    batch = _pop(q)
    assert len(batch) == 4 and not q.last_batch_forced_service


def test_work_conserving_respects_budget_and_ranks_by_value():
    reqs = [_req(0, 0.0, 4, 10.0, 0.0, 0.0), _req(1, 1.0, 4, 90.0, 0.0, 0.0),
            _req(2, 2.0, 4, 50.0, 0.0, 0.0)]
    q = VerificationQueue(); q.capacity_work_conserving = True; q.extend(reqs)
    batch = q.pop_batch("capacity_knapsack", 8, 1e9, 10, 0.0)
    assert [r.client_id for r in batch] == [1, 2]


def test_scheduler_without_theta_f_serves_all():
    reqs = [_req(i, float(i), 4, 50.0, 300.0, 3.83) for i in range(4)]
    q = _queue(reqs, capacity_scheduler_theta_f=False)
    assert len(_pop(q)) == 4
