"""C5-A: counterexample and validity-condition tests for the finite-cap capacity claims.  They use an INDEPENDENT mini DES (scripts/c5a/mini_des.py) and never touch
the project simulator.  A test either pins a claim that holds under stated assumptions or pins a counterexample to a claim whose assumptions were incomplete."""
import collections
import itertools
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "c5a"))
from mini_des import greedy_partition, maxplus_cycle, saturated_cycle, simulate  # noqa: E402


def gaps(res, t0, t1):
    last, g = {}, collections.defaultdict(list)
    for (s, f, ids) in res.batches:
        if t0 <= f <= t1:
            for i in ids:
                if i in last:
                    g[i].append(f - last[i])
                last[i] = f
    per = {i: sum(v) / len(v) for i, v in g.items() if v}
    allg = [x for v in g.values() for x in v]
    return sum(allg) / len(allg), min(per.values()), max(per.values())


def R(N, B, a, d, D, W=0.0, **kw):
    res = simulate(N, B, a, d, D, W=W, t_max=kw.pop("t_max", 150000.0), **kw)
    return gaps(res, 60000.0, 140000.0), res


# --------------------------------------------------------------------------- claims that hold (with assumptions)
@pytest.mark.parametrize("N,B", [(8, 8), (8, 16), (12, 8), (16, 8), (16, 16), (20, 16), (24, 16), (28, 16), (32, 16)])
def test_max_plus_cycle_time_holds_for_N_up_to_2B_deterministic_identical_D_hold_head(N, B):
    """Theorem-A domain: K = ceil(N/B) <= 2, deterministic affine batches, identical return delay D, hold-head work-conserving FCFS, simultaneous start."""
    for a, d, D, W in itertools.product((10.0, 52.0), (5.0, 30.0), (1.0, 50.0, 300.0, 1000.0), (0.0, 150.0)):
        (m, lo, hi), _ = R(N, B, a, d, D, W)
        assert hi <= 1.0001 * lo                                                                    # every client is served at the same frequency
        assert m == pytest.approx(maxplus_cycle(N, B, a, [d] * N, D, W), rel=1e-6)


def test_heterogeneous_context_cost_is_covered_through_the_sum_and_the_largest_batch():
    rng = random.Random(7)
    N, B = 24, 16
    dl = [rng.uniform(10, 60) for _ in range(N)]
    (m, lo, hi), _ = R(N, B, 52.0, dl, 50.0, 0.0)
    assert hi <= 1.0001 * lo and m == pytest.approx(maxplus_cycle(N, B, 52.0, dl, 50.0, 0.0), rel=1e-6)


def test_partitions_with_the_same_number_of_batches_have_the_same_saturated_cycle():
    """16+8 (cap 16) and 12+12 (cap 12) are different partitions of the same total affine cost: only the NUMBER of batches enters."""
    (m16, _, _), _ = R(24, 16, 52.0, 30.0, 50.0)
    (m12, _, _), _ = R(24, 12, 52.0, 30.0, 50.0)
    assert m16 == pytest.approx(m12, rel=1e-9)
    assert m16 == pytest.approx(saturated_cycle(24, 16, 52.0, [30.0] * 24), rel=1e-9)


def test_saturated_law_is_robust_to_arrival_jitter_when_the_server_never_idles_and_N_le_2B():
    for N in (24, 28, 32):
        (m0, _, _), _ = R(N, 16, 52.0, 30.0, 50.0)
        (mj, _, _), _ = R(N, 16, 52.0, 30.0, 50.0, jitter=100.0, seed=3)
        assert mj == pytest.approx(m0, rel=1e-6)


def test_existential_restriction_lemma_removing_a_client_never_lengthens_any_batch():
    """Lemma (existential monotonicity): with admissible idling, restricting a schedule to a subset leaves every remaining client's completion times unchanged;
    the affine service time is monotone in membership, so each batch is not longer after the removal (idling pads it back to the original length)."""
    rng = random.Random(1)
    N, B = 20, 8
    dl = [rng.uniform(5, 40) for _ in range(N)]
    res = simulate(N, B, 52.0, dl, 80.0, W=0.0, t_max=30000.0)
    for (s, f, ids) in res.batches:
        for c in ids:
            reduced = 52.0 + sum(dl[i] for i in ids if i != c)
            assert reduced <= (f - s) + 1e-9


# --------------------------------------------------------------------------- counterexamples (claims whose assumptions were incomplete)
def test_cap_can_bind_with_N_le_2B_in_a_transient_with_a_partial_previous_batch():
    """C3's claim 'N <= 2B => selection vacuous' needs a FULL previous batch; with a partial first batch more than B requests are eligible."""
    N, B = 20, 16
    start = [0.0] * 3 + [10.0] * 17
    res = simulate(N, B, 52.0, 30.0, 50.0, W=0.0, start=start, t_max=5000.0)
    sizes = [len(ids) for (_, _, ids) in res.batches[:6]]
    assert sizes[:2] == [3, 16]                                    # batch 2 starts at t=142 with 17 arrived requests (> B): one is left over
    assert sizes[2] == 4                                           # the leftover plus the three returning clients
    assert sizes[3:] == [16, 4, 16]                                # and the system settles into the (4, 16) rotation: K = 2, steady-state eligible <= B


def test_ceiling_partition_law_is_wrong_beyond_2B_because_batches_stay_full():
    N, B, a, d, D = 40, 16, 52.0, 30.0, 50.0
    (m, lo, hi), res = R(N, B, a, d, D)
    assert saturated_cycle(N, B, a, [d] * N) == 1356.0
    assert m < 1356.0 * 0.99                                       # measured cycle is shorter: batches are (almost) all full
    assert m == pytest.approx((N / B) * a + N * d, rel=0.005)      # fractional N/B overheads
    sizes = [len(ids) for (_, _, ids) in res.batches[-200:]]
    assert sum(sizes) / len(sizes) > 15.9


def test_service_frequency_is_not_uniform_beyond_2B_with_heterogeneous_return_delays():
    """Beyond N = 2B (where selection is non-trivial) a single cycle time does not exist: clients are served at different frequencies, so the floor is NOT phi/mean-cycle."""
    rng = random.Random(5)
    N, B = 40, 16
    D = [rng.uniform(0, 300) for _ in range(N)]
    (m, lo, hi), _ = R(N, B, 52.0, 30.0, D)
    assert hi > 1.4 * lo


def test_a_fixed_policy_can_have_a_locally_increasing_achieved_rate_but_hold_head_cannot():
    """x_pi(N) is policy-dependent: launch-when-full-or-timeout makes N=B faster than N=B-1 (no hold needed); hold-head is monotone."""
    B, a, d, D, W = 16, 52.0, 30.0, 50.0, 150.0
    r_fot = {N: R(N, B, a, d, D, W, rule="full_or_timeout")[0][0] for N in range(8, 21)}
    assert r_fot[16] < r_fot[15] - 100.0                           # cycle shortens by > 100 ms when the batch can launch full
    r_hold = [R(N, B, a, d, D, W)[0][0] for N in range(1, 41)]
    assert all(r_hold[i + 1] >= r_hold[i] - 1e-6 for i in range(len(r_hold) - 1))


def test_threshold_policy_has_zero_rate_below_its_threshold_then_positive():
    res = simulate(5, 16, 52.0, 30.0, 50.0, rule="wait_k", k=8, t_max=5000.0)
    assert res.deadlocked and sum(res.completions.values()) == 0
    res = simulate(8, 16, 52.0, 30.0, 50.0, rule="wait_k", k=8, t_max=5000.0)
    assert not res.deadlocked and sum(res.completions.values()) > 0


def test_c4_cohort_example_for_N28_was_wrong_it_is_16_plus_12_not_14_plus_14():
    assert greedy_partition(28, 16) == [16, 12]
    res = simulate(28, 16, 52.0, 30.0, 50.0, W=0.0, t_max=20000.0)
    sizes = collections.Counter(len(ids) for (_, _, ids) in res.batches[2:])
    assert set(sizes) == {16, 12}
