import pytest

from sim.metrics.virtual_queues import VirtualQueues, total_backlog, update_virtual_queues


def test_zeros_factory():
    q = VirtualQueues.zeros(3)
    assert q.Z == [0.0, 0.0, 0.0]
    assert q.Q == 0.0
    assert q.Qi == [0.0, 0.0, 0.0]


def test_update_matches_eq12_formula():
    q = VirtualQueues(Z=[1.0, 5.0], Q=2.0, Qi=[0.5, 3.0])
    updated = update_virtual_queues(
        q,
        x_target=2.0,
        a_i=[3.0, 1.0],
        theta_f_seconds_per_token=0.1,
        verification_batch_tokens=20.0,
        gammas=[1, 4],
        device_time_per_gamma=[0.2, 0.1],
        slot_seconds=1.0,
    )
    # Z_i(t+1) = [Z_i(t) + x*Delta_s - a_i(t)]^+
    assert updated.Z[0] == pytest.approx(max(0.0, 1.0 + 2.0 * 1.0 - 3.0))
    assert updated.Z[1] == pytest.approx(max(0.0, 5.0 + 2.0 * 1.0 - 1.0))
    # Q(t+1) = [Q(t) + theta_f*Gamma(B(t)) - Delta_s]^+
    assert updated.Q == pytest.approx(max(0.0, 2.0 + 0.1 * 20.0 - 1.0))
    # Q_i(t+1) = [Q_i(t) + gamma_i(t)*(tau_i^d+kappa/r_i(t)) - Delta_s]^+
    assert updated.Qi[0] == pytest.approx(max(0.0, 0.5 + 1 * 0.2 - 1.0))
    assert updated.Qi[1] == pytest.approx(max(0.0, 3.0 + 4 * 0.1 - 1.0))


def test_update_clips_at_zero():
    q = VirtualQueues(Z=[0.0], Q=0.0, Qi=[0.0])
    updated = update_virtual_queues(
        q,
        x_target=0.0,
        a_i=[100.0],
        theta_f_seconds_per_token=0.0,
        verification_batch_tokens=0.0,
        gammas=[0],
        device_time_per_gamma=[0.0],
        slot_seconds=1.0,
    )
    assert updated.Z == [0.0]
    assert updated.Q == 0.0
    assert updated.Qi == [0.0]


def test_total_backlog_sums_everything():
    q = VirtualQueues(Z=[1.0, 2.0], Q=3.0, Qi=[4.0, 5.0])
    assert total_backlog(q) == pytest.approx(15.0)


def test_update_rejects_mismatched_lengths():
    q = VirtualQueues.zeros(2)
    with pytest.raises(ValueError):
        update_virtual_queues(
            q,
            x_target=1.0,
            a_i=[1.0],  # wrong length
            theta_f_seconds_per_token=0.1,
            verification_batch_tokens=1.0,
            gammas=[0, 0],
            device_time_per_gamma=[0.1, 0.1],
            slot_seconds=1.0,
        )
