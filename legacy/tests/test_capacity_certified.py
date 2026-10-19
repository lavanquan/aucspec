"""Tests for the certified capacity layer (C1: M3 classifier, M5-M7 boundaries/ICA, M8 dual consistency)."""
import math
import random

import pytest

from edge_specsim import capacity as legacy
from edge_specsim.capacity import CandidateMeasurement, NStarResult, SearchDecision
from edge_specsim.capacity_certified import (
    FEASIBLE,
    INFEASIBLE,
    INVALID,
    UNCERTAIN,
    area_from_boundaries,
    area_from_x_boundaries,
    bootstrap_sensitivity,
    certified_n_search,
    certified_x_search,
    classify_certified,
    dual_consistency,
    student_t_cdf,
    student_t_ppf,
    summarize_probes,
)


def meas(rates, invalid=False, seed=1):
    return CandidateMeasurement(scale=len(rates), n_active=len(rates), x_requirement=1.0, seed=seed,
                                min_rate_tps=min(rates.values()), per_client_rates=dict(rates), invalid=invalid)


def gen(mu, sd, k, n_clients=10, weak=0, seed=0, dist="normal"):
    rng = random.Random(seed)
    out = []
    for s in range(k):
        rates = {}
        for c in range(n_clients):
            m = mu if c == weak else mu + 5.0
            if dist == "normal":
                e = rng.gauss(0, sd)
            elif dist == "lognormal":
                e = (rng.lognormvariate(0, 1.0) - math.exp(0.5)) * sd
            elif dist == "t3":
                z = rng.gauss(0, 1)
                chi = sum(rng.gauss(0, 1) ** 2 for _ in range(3))
                e = z / math.sqrt(chi / 3) * sd / math.sqrt(3)       # unit variance scaled to sd
            else:
                raise ValueError(dist)
            rates[c] = m + e
        out.append(meas(rates, seed=s))
    return out


# --------------------------------------------------------------------- Student t
@pytest.mark.parametrize("p,df,expected", [(0.975, 4, 2.776445), (0.995, 4, 4.604095), (0.95, 9, 1.833113),
                                           (0.9995, 6, 5.958816), (0.975, 1000, 1.962339), (0.75, 2, 0.816497)])
def test_student_t_ppf_known_values(p, df, expected):
    assert student_t_ppf(p, df) == pytest.approx(expected, abs=2e-5)
    assert student_t_cdf(student_t_ppf(p, df), df) == pytest.approx(p, abs=1e-9)
    assert student_t_ppf(1 - p, df) == pytest.approx(-expected, abs=2e-5)


# --------------------------------------------------------------------- classifier
def test_invalid_is_never_classified():
    ms = gen(0.1, 0.01, 5)
    ms[2] = meas({c: 0.0 for c in range(10)}, invalid=True)
    assert classify_certified(ms, 1.0, 10).status == INVALID


def test_minimum_certification_size_and_uncertain_outcome():
    ms = gen(8.0, 0.1, 4)                                  # clearly feasible but only K=4
    d = classify_certified(ms, 1.0, 10)
    assert d.status == UNCERTAIN and not d.certified and d.screening_status == FEASIBLE
    assert "insufficient_seeds" in d.reason
    d5 = classify_certified(gen(8.0, 0.1, 5), 1.0, 10)
    assert d5.status == FEASIBLE and d5.certified
    assert classify_certified(gen(8.0, 0.1, 1), 1.0, 10).status == UNCERTAIN


def test_feasible_infeasible_uncertain_regions():
    assert classify_certified(gen(3.0, 0.05, 6), 1.0, 10).status == FEASIBLE
    assert classify_certified(gen(0.5, 0.05, 6), 1.0, 10).status == INFEASIBLE
    assert classify_certified(gen(1.0, 0.3, 6), 1.0, 10).status == UNCERTAIN


def test_quantile_family_adjustment_sidedness_and_spending():
    ms = gen(1.0, 0.2, 5)
    one = classify_certified(ms, 1.0, 10, sided="one")
    two = classify_certified(ms, 1.0, 10, sided="two")
    assert two.quantile > one.quantile
    assert classify_certified(ms, 1.0, 40).quantile > classify_certified(ms, 1.0, 10).quantile   # Bonferroni over N
    assert classify_certified(ms, 1.0, 10, n_looks=2, n_probes=3).quantile > one.quantile        # alpha spending
    assert one.delta_effective == pytest.approx(0.05) and classify_certified(ms, 1.0, 10, n_looks=2).delta_effective == pytest.approx(0.025)
    # bounds are symmetric around the mean for the weak client and use K-1 degrees of freedom
    k = 5
    vals = [m.per_client_rates[0] for m in ms]
    mean = sum(vals) / k
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (k - 1))
    assert one.per_client_lcb[0] == pytest.approx(mean - student_t_ppf(1 - 0.05 / 10 / 1, 4) * sd / math.sqrt(k))


def _false_feasible_rate(k, dist, classifier, trials=400, sd=0.3, n_clients=10, seed=0):
    hits = 0
    for t in range(trials):
        ms = gen(1.0, sd, k, n_clients=n_clients, seed=seed * 100003 + t, dist=dist)
        hits += classifier(ms) == FEASIBLE
    return hits / trials


def test_small_k_bootstrap_is_anticonservative_while_student_t_is_not():
    """The audit finding behind M3, on the weakest client sitting exactly at x (mu = x)."""
    boot = lambda ms: bootstrap_sensitivity(ms, 1.0, 10, n_boot=300)["status"]
    stud = lambda ms: classify_certified(ms, 1.0, 10, min_seeds=2).status
    r_boot = _false_feasible_rate(3, "normal", boot, trials=200)
    r_t = _false_feasible_rate(3, "normal", stud, trials=200)
    assert r_boot > 0.05 and r_t <= 0.02
    assert r_boot > 4 * r_t + 0.02


@pytest.mark.parametrize("dist", ["normal", "lognormal", "t3"])
def test_student_t_false_feasible_at_the_boundary_is_controlled_for_k5_and_k7(dist):
    for k in (5, 7):
        rate = _false_feasible_rate(k, dist, lambda ms: classify_certified(ms, 1.0, 10).status)
        assert rate <= 0.05 + 0.025, (dist, k, rate)        # nominal delta=0.05 plus Monte-Carlo slack


def test_bootstrap_is_descriptive_only():
    out = bootstrap_sensitivity(gen(3.0, 0.1, 6), 1.0, 10, n_boot=200)
    assert out["certificate"] is False and out["status"] == FEASIBLE


# --------------------------------------------------------------------- boundaries (M5/M7)
def table_decide(table):
    calls = []

    def decide(n):
        calls.append(n)
        return table(n) if callable(table) else table.get(n, INFEASIBLE)

    decide.calls = calls
    return decide


def test_uncertain_point_is_never_an_endpoint_and_old_search_claimed_it_was():
    tab = {1: FEASIBLE, 2: FEASIBLE, 3: FEASIBLE, 4: UNCERTAIN, 5: UNCERTAIN, 6: UNCERTAIN}
    dec = lambda n: tab.get(n, INFEASIBLE)
    # legacy: an uncertain probe becomes the upper bound
    old = legacy.monotone_scale_search_final(
        lambda n: SearchDecision(n=n, status=dec(n), min_lcb=0.0, min_ucb=0.0, seeds_used=5), hi_cap=16, x_requirement=3.0)
    assert old.upper_bound_n == 4, "legacy claimed the uncertain N=4 as the upper bound although the truth may be 6"
    new = certified_n_search(dec, hi_cap=16, x_requirement=3.0)
    assert new.lower_bound_n == 3
    assert new.upper_bound_monotone == 6 and new.upper_bound_n == 6        # first CERTIFIED infeasible N=7, minus 1
    assert set(new.gray_ns) >= {4}
    assert new.status == "bracketed" and not new.exact_conditional


def test_exact_conditional_requires_adjacent_certified_points():
    tab = lambda n: FEASIBLE if n <= 5 else INFEASIBLE
    r = certified_n_search(table_decide(tab), hi_cap=32)
    assert (r.lower_bound_n, r.upper_bound_monotone, r.status, r.exact_conditional) == (5, 5, "exact_conditional", True)
    assert r.upper_bound_unconditional is None, "a monotone search alone gives no unconditional upper bound"


def test_gray_zone_wider_than_search_is_reported_unresolved():
    tab = lambda n: FEASIBLE if n <= 3 else UNCERTAIN
    r = certified_n_search(table_decide(tab), hi_cap=16)
    assert r.lower_bound_n == 3 and r.upper_bound_n is None and r.status == "unresolved_upper"
    assert r.gray_ns


def test_cap_limited_is_a_lower_bound():
    r = certified_n_search(table_decide(lambda n: FEASIBLE), hi_cap=12)
    assert r.hit_search_cap and r.lower_bound_n == 12 and r.upper_bound_n is None and r.status == "cap_limited"


def test_nonmonotone_policy_counterexample_withdraws_the_conditional_bound():
    probes = {1: FEASIBLE, 2: FEASIBLE, 3: FEASIBLE, 4: INFEASIBLE, 6: FEASIBLE, 9: INFEASIBLE}
    r = summarize_probes(3.0, probes, hi_cap=16)
    assert r.monotone_violations == [(6, 4)]
    assert r.lower_bound_n == 6                              # achieved capacity: unconditional
    assert r.upper_bound_monotone is None and r.upper_bound_n is None
    assert r.status == "nonmonotone_unresolved" and not r.exact_conditional


def test_monotone_search_cannot_see_a_nonmonotone_region_but_extra_probes_falsify_it():
    truth = lambda n: FEASIBLE if (n <= 3 or 6 <= n <= 7) else INFEASIBLE   # feasible again at 6,7
    dec = table_decide(truth)
    r = certified_n_search(dec, hi_cap=16)
    assert r.lower_bound_n == 3 and r.upper_bound_monotone == 3 and r.status == "exact_conditional"
    # the dedicated falsification probes (N*+2, N*+3) expose it and withdraw the conditional bound
    probes = dict(r.probes)
    for n in (r.lower_bound_n + 2, r.lower_bound_n + 3):
        probes[n] = truth(n)
    r2 = summarize_probes(3.0, probes, 16)
    assert r2.monotone_violations and r2.status == "nonmonotone_unresolved" and r2.lower_bound_n >= 6


def test_unconditional_upper_bound_when_the_whole_tail_is_certified_infeasible():
    probes = {n: (FEASIBLE if n <= 4 else INFEASIBLE) for n in range(1, 9)}
    r = summarize_probes(3.0, probes, hi_cap=8)
    assert r.upper_bound_unconditional == 4 and r.upper_bound_monotone == 4


def test_invalid_aborts_search_and_poisons_area():
    tab = lambda n: INVALID if n == 4 else (FEASIBLE if n < 4 else INFEASIBLE)
    r = certified_n_search(table_decide(tab), hi_cap=16, x_requirement=3.0)
    assert r.status == "invalid" and r.invalid_ns == [4]
    ok = summarize_probes(4.0, {1: FEASIBLE, 2: INFEASIBLE}, 16)
    area = area_from_boundaries([r, ok], x_L=3.0, x_U=4.0, n_ref=16)
    assert area["status"] == "INVALID" and area["area_lower"] is None and area["invalid_x"] == [3.0]


# --------------------------------------------------------------------- ICA / staircase (M6)
def exact_result(x, n_true, hi_cap=64):
    probes = {n: (FEASIBLE if n <= n_true else INFEASIBLE) for n in range(1, min(hi_cap, n_true + 2) + 1)}
    return summarize_probes(x, probes, hi_cap)


def true_area(fn, x_L, x_U, n_ref, steps=200000):
    h = (x_U - x_L) / steps
    return sum(min(fn(x_L + (i + 0.5) * h), n_ref) * h for i in range(steps))


def test_staircase_bounds_bracket_the_true_area_and_legacy_lower_bound_does_not():
    n_of_x = lambda x: math.floor(24.0 / x + 1e-12)
    xs = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    res = [exact_result(x, n_of_x(x)) for x in xs]
    x_L, x_U, n_ref = 1.0, 12.0, 24
    truth = true_area(n_of_x, x_L, x_U, n_ref)
    a = area_from_boundaries(res, x_L=x_L, x_U=x_U, n_ref=n_ref)
    assert a["status"] == "OK"
    assert a["area_lower"] <= truth + 1e-3 <= a["area_upper"] + 2e-3
    assert a["discretisation_gap"] > 0 and a["boundary_uncertainty_gap"] == pytest.approx(0.0, abs=1e-9)
    # legacy area_from_nstar uses the LEFT (higher) value for BOTH bounds -> "lower" bound above the truth
    old_res = [NStarResult(x_requirement=x, last_confirmed_feasible_n=n_of_x(x), first_confirmed_infeasible_n=n_of_x(x) + 1,
                           lower_bound_n=n_of_x(x), upper_bound_n=n_of_x(x) + 1, exact=True, hit_search_cap=False) for x in xs]
    old = legacy.area_from_nstar(old_res, x_L=x_L, x_U=x_U, n_ref=n_ref)
    assert old["area_lower"] > truth + 1.0


def test_unresolved_upper_points_widen_the_upper_integral_to_n_ref():
    xs = [1.0, 2.0, 3.0]
    res = [exact_result(1.0, 12), summarize_probes(2.0, {1: FEASIBLE, 5: FEASIBLE, 8: UNCERTAIN}, 16), exact_result(3.0, 4)]
    a = area_from_boundaries(res, x_L=1.0, x_U=3.0, n_ref=16)
    assert a["unresolved_upper_points"] == [2.0]
    assert a["area_upper"] >= 16 * 1.0                         # cell [1,2] uses max(up(1), up(2)=n_ref)
    assert a["area_lower"] <= a["area_upper"]


def test_grid_must_cover_window_and_partial_cells_are_handled():
    res = [exact_result(1.0, 10), exact_result(2.0, 5), exact_result(4.0, 2)]
    with pytest.raises(ValueError):
        area_from_boundaries(res, x_L=0.5, x_U=3.0, n_ref=16)
    a = area_from_boundaries(res, x_L=1.5, x_U=3.0, n_ref=16)
    assert a["area_lower"] == pytest.approx(5 * 0.5 + 2 * 1.0)     # right ends: cell [1.5,2]->N(2)=5, [2,3]->N(4)=2
    assert a["area_upper"] == pytest.approx(10 * 0.5 + 5 * 1.0)    # left ends


def test_nonmonotone_in_x_is_flagged_and_uses_the_conservative_endpoint_for_the_lower_bound():
    res = [exact_result(1.0, 4), exact_result(2.0, 9)]             # N*(x) increases with x: monotonicity violated
    a = area_from_boundaries(res, x_L=1.0, x_U=2.0, n_ref=16)
    assert a["x_monotonicity_violations"] == [(1.0, 2.0)]
    assert a["area_lower"] == pytest.approx(4.0)                   # min(4, 9) * 1


# --------------------------------------------------------------------- x-axis and duality (M7/M8)
def build_dual(frontier, xs, ns, x_hi=30.0, tol=0.05):
    """frontier(N, x) -> bool for the controller-achieved region F = {(N,x): feasible}."""
    n_res = [certified_n_search(lambda n, x=x: FEASIBLE if frontier(n, x) else INFEASIBLE, hi_cap=max(ns), x_requirement=x)
             for x in xs]
    x_res = [certified_x_search(lambda xx, n=n: FEASIBLE if frontier(n, xx) else INFEASIBLE, n_active=n,
                                x_lo=0.1, x_hi=x_hi, x_tolerance_tps=tol) for n in ns]
    return n_res, x_res


def test_x_search_uncertain_midpoint_is_not_an_endpoint_and_invalid_aborts():
    f = lambda x: FEASIBLE if x <= 3.0 else INFEASIBLE
    g = lambda x: UNCERTAIN if 2.9 < x < 3.2 else f(x)
    r = certified_x_search(g, n_active=4, x_lo=0.1, x_hi=10.0, x_tolerance_tps=0.05)
    assert r.lower_feasible_x <= 2.9 + 1e-9 or r.lower_feasible_x == pytest.approx(3.0, abs=0.05)
    assert all(r.probes[x] != UNCERTAIN or x in r.gray_xs for x in r.probes)
    assert r.upper_infeasible_x is None or r.probes[r.upper_infeasible_x] == INFEASIBLE
    bad = certified_x_search(lambda x: INVALID if x > 5 else FEASIBLE, n_active=2, x_lo=0.1, x_hi=10.0)
    assert bad.status == "invalid"
    assert area_from_x_boundaries([bad], x_L=1.0, x_U=4.0, n_ref=1)["status"] == "INVALID"


def test_dual_consistency_on_a_synthetic_monotone_region_holds_and_asserts_exact_identities():
    frontier = lambda n, x: n * x <= 24.0 + 1e-12
    xs = [3.0, 4.0, 6.0, 8.0]
    n_res, x_res = build_dual(frontier, xs, list(range(1, 13)))
    out = dual_consistency(n_res, x_res, x_L=3.0, x_U=8.0, n_ref=12)
    assert out["consistent"], out["violations"]
    assert out["interval_overlap"] is True
    assert out["exact_identities_asserted"] is True
    truth = true_area(lambda x: math.floor(24.0 / x + 1e-12), 3.0, 8.0, 12)
    assert out["area_N_axis"]["area_lower"] <= truth + 1e-3
    assert out["area_x_axis"]["area_lower"] <= truth + 1e-3 <= out["area_x_axis"]["area_upper"] + 0.5


def test_dual_consistency_detects_contradiction_between_axes():
    frontier = lambda n, x: n * x <= 24.0
    n_res, x_res = build_dual(frontier, [2.0, 4.0], list(range(1, 9)))
    # corrupt x*(N=3): claim it is infeasible already at x=1 although N-axis says N=3 is feasible at x=2
    bad = certified_x_search(lambda x: FEASIBLE if x <= 0.5 else INFEASIBLE, n_active=3, x_lo=0.1, x_hi=30.0)
    x_res = [r for r in x_res if r.n_active != 3] + [bad]
    out = dual_consistency(n_res, x_res, x_L=2.0, x_U=4.0, n_ref=8)
    assert not out["consistent"] and any("N=3" in v for v in out["violations"])


def test_exact_identities_are_not_asserted_when_prerequisites_fail():
    frontier = lambda n, x: n * x <= 24.0
    n_res, x_res = build_dual(frontier, [2.0, 4.0], list(range(1, 9)))
    n_res[0].gray_ns.append(5)                                     # a gray point appears on the N axis
    out = dual_consistency(n_res, x_res, x_L=2.0, x_U=4.0, n_ref=8)
    assert out["exact_identities_asserted"] is False and out["exact_check"] is None


def test_near_deterministic_runs_do_not_yield_zero_width_certificates():
    rates = {c: 1.0 for c in range(4)}                       # five identical measurements exactly AT x
    ms = [meas(dict(rates), seed=s) for s in range(5)]
    d = classify_certified(ms, 1.0, 4)
    assert d.status == UNCERTAIN and d.degenerate_zero_variance_clients
    far = [meas({c: 3.0 for c in range(4)}, seed=s) for s in range(5)]
    assert classify_certified(far, 1.0, 4).status == FEASIBLE
