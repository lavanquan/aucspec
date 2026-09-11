"""CPU-only tests for the N* capacity-search plumbing
(CAPACITY_AUC_NSTAR_IMPLEMENTATION.md Section 22, items that do not need a GPU).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from edge_specsim.population import PopulationCatalog, PopulationTemplate
from edge_specsim.capacity import (
    CandidateCache,
    CandidateMeasurement,
    NStarResult,
    SearchDecision,
    XStarResult,
    area_from_nstar,
    area_from_xstar,
    classify_feasibility,
    classify_feasibility_final,
    ica,
    monotone_scale_search,
    monotone_scale_search_final,
    step_area_bounds,
    sustained_rates,
    tail_slope,
    xstar_search,
)
from edge_specsim.controller import OnlineController
from edge_specsim.models import ClientProfile
from edge_specsim.metrics import (
    compute_sustained_service_rates,
    summarize_sustained_service_rates,
)


# ------------------------------------------------------------------ 1, 2, 8
def test_nested_population_is_prefix_stable():
    cat = PopulationCatalog(seed=7, n_max=30)
    small = cat.prefix(12)
    big = cat.prefix(24)
    assert [s.stable_id for s in small] == list(range(12))
    assert big[:12] == small  # increasing N only ADDS clients
    # a fresh catalog with the same seed reproduces the same prefix
    cat2 = PopulationCatalog(seed=7, n_max=60)
    assert cat.is_nested_with(cat2, 12)
    assert cat.fingerprint(12) == cat2.fingerprint(12)


def test_template_scaling_keeps_class_proportions_exact():
    tmpl = PopulationTemplate.from_config([1, 1, 1])
    cat = PopulationCatalog(seed=1, n_max=30, template=tmpl)
    for m in (1, 3, 5, 9):
        specs = cat.scale(m)
        assert len(specs) == 3 * m
        counts: dict[str, int] = {}
        for s in specs:
            counts[s.class_name] = counts.get(s.class_name, 0) + 1
        assert set(counts.values()) == {m}  # every class appears exactly m times


def test_template_scaling_is_nested():
    tmpl = PopulationTemplate.from_config({"easy": 2, "hard": 1})
    cat = PopulationCatalog(seed=3, n_max=60, template=tmpl)
    assert cat.scale(4) == cat.scale(7)[: 4 * cat.block_size]


# ------------------------------------------------------------------ 3
def test_monotone_scale_search_brackets_and_binary_searches():
    # true boundary: feasible for m <= 5, infeasible for m >= 6
    calls: list[int] = []

    def feas(m: int) -> str:
        calls.append(m)
        return "feasible" if m <= 5 else "infeasible"

    m_star, trace = monotone_scale_search(feas, lo_feasible=1, hi_cap=100)
    assert m_star == 5
    assert trace["boundary_first_infeasible"] is not None
    assert trace["boundary_first_infeasible"] >= 6


def test_monotone_scale_search_respects_cap():
    m_star, _ = monotone_scale_search(lambda m: "feasible", lo_feasible=1, hi_cap=8)
    assert m_star == 8


def test_monotone_scale_search_lo_infeasible_returns_zero():
    m_star, trace = monotone_scale_search(lambda m: "infeasible", lo_feasible=1, hi_cap=10)
    assert m_star == 0
    assert "note" in trace


def test_monotone_scale_search_narrows_through_uncertain():
    # feasible <=4, uncertain at 5-9, infeasible >=10: N* must be 4, not 1
    def feas(m: int) -> str:
        if m <= 4:
            return "feasible"
        if m <= 9:
            return "uncertain"
        return "infeasible"

    m_star, trace = monotone_scale_search(feas, lo_feasible=1, hi_cap=100, hi_hint=12)
    assert m_star == 4
    assert 5 in trace["uncertain_scales"] or any(s >= 5 for s in trace["uncertain_scales"])


def test_monotone_scale_search_verifies_hi_hint_equal_to_cap():
    # regression: hi_hint == hi_cap must still be tested, not skipped
    m_star, _ = monotone_scale_search(lambda m: "feasible", lo_feasible=1, hi_cap=8, hi_hint=8)
    assert m_star == 8


# ------------------------------------------------------------------ 4
def test_candidate_cache_roundtrip(tmp_path):
    path = tmp_path / "cache.jsonl"
    c = CandidateCache(path)
    m = CandidateMeasurement(scale=3, n_active=9, x_requirement=1.0, seed=1, min_rate_tps=1.2)
    assert c.get(1.0, 3, "capacity_dpp", 1, "hashA") is None
    c.put(1.0, 3, "capacity_dpp", 1, "hashA", m)
    assert c.get(1.0, 3, "capacity_dpp", 1, "hashA").min_rate_tps == pytest.approx(1.2)
    # a new cache object reads the same file back (resume)
    c2 = CandidateCache(path)
    got = c2.get(1.0, 3, "capacity_dpp", 1, "hashA")
    assert got is not None and got.n_active == 9
    # a different config hash misses
    assert c2.get(1.0, 3, "capacity_dpp", 1, "hashB") is None


# ------------------------------------------------------------------ 5
def _meas(seed, rate, z_slope=0.0):
    return CandidateMeasurement(
        scale=3, n_active=9, x_requirement=1.0, seed=seed,
        min_rate_tps=rate, max_z_slope=z_slope,
    )


def test_feasibility_classifier_feasible():
    r = classify_feasibility([_meas(1, 1.20), _meas(2, 1.19), _meas(3, 1.21)], x=1.0)
    assert r.status == "feasible"


def test_feasibility_classifier_infeasible_by_rate():
    r = classify_feasibility([_meas(1, 0.80), _meas(2, 0.79), _meas(3, 0.81)], x=1.0)
    assert r.status == "infeasible"


def test_feasibility_classifier_infeasible_by_queue_growth():
    # rate is only marginally over the floor AND Z_i clearly diverging
    r = classify_feasibility(
        [_meas(1, 0.98, z_slope=0.15), _meas(2, 0.99, z_slope=0.16)], x=1.0
    )
    assert r.status == "infeasible"


def test_feasibility_classifier_ignores_transient_z_slope_when_rate_is_high():
    # min-rate 6.2 vs x=4.5: a small positive Z_i tail slope is a transient,
    # not instability -- must be "feasible" (this was the pilot bug)
    r = classify_feasibility(
        [_meas(1, 6.15, z_slope=0.05), _meas(2, 6.38, z_slope=0.06), _meas(3, 6.11, z_slope=0.04)],
        x=4.5,
    )
    assert r.status == "feasible"


def test_feasibility_classifier_uncertain_on_boundary():
    # mean ~0.973, floor = 0.97: lower estimate dips just below, upper just above
    r = classify_feasibility([_meas(1, 0.99), _meas(2, 0.96), _meas(3, 0.97)], x=1.0)
    assert r.status == "uncertain"


# ------------------------------------------------------------------ 6
def test_sustained_rate_common_denominator_and_zero_service():
    useful = {0: 30.0, 1: 60.0}  # client 2 admitted but got nothing
    rates = sustained_rates(useful, admitted_client_ids=[0, 1, 2], t_measurement_s=30.0)
    assert rates == pytest.approx({0: 1.0, 1: 2.0, 2: 0.0})
    assert min(rates.values()) == 0.0  # zero-service client is NOT dropped


def test_metrics_sustained_rate_frame_includes_admitted_zero_service_client():
    df = pd.DataFrame(
        {
            "client_id": [0, 0, 1],
            "useful_tokens": [10.0, 20.0, 45.0],
            "in_measurement_window": [True, True, True],
        }
    )
    frame = compute_sustained_service_rates(df, t_measurement_s=15.0, admitted_client_ids=[0, 1, 2])
    by_id = dict(zip(frame["client_id"], frame["sustained_service_rate_tps"]))
    assert by_id[0] == pytest.approx(2.0)
    assert by_id[1] == pytest.approx(3.0)
    assert by_id[2] == pytest.approx(0.0)
    summ = summarize_sustained_service_rates(frame)
    assert summ["min_sustained_service_rate_tps"] == pytest.approx(0.0)
    assert summ["n_admitted"] == 3


# ------------------------------------------------------------------ 7
def test_tail_slope_detects_positive_drift():
    t = [float(i) for i in range(20)]
    growing = [0.1 * i for i in range(20)]
    flat = [5.0 + (0.001 if i % 2 else -0.001) for i in range(20)]
    assert tail_slope(t, growing) == pytest.approx(0.1, abs=1e-6)
    assert abs(tail_slope(t, flat)) < 1e-2


# ------------------------------------------------------------------ 9
def _client(z_queue=0.0, device_queue=0.0, alpha=0.9):
    return ClientProfile(
        client_id=0,
        prompt_queue=None,
        draft_model_id="d",
        draft_device_class="default",
        draft_tokens_per_second=40.0,
        draft_fixed_latency_ms=5.0,
        rtt_ms=40.0,
        uplink_mbps=20.0,
        downlink_mbps=40.0,
        packet_loss=0.0,
        uplink_base_snr_db=15.0,
        downlink_base_snr_db=18.0,
        uplink_snr_jitter_db=2.0,
        downlink_snr_jitter_db=2.0,
        snr_period_ms=1000.0,
        snr_phase_rad=0.0,
        wireless_weight=1.0,
        z_queue=z_queue,
        device_queue=device_queue,
        alpha_hat=alpha,
        alpha_ucb=alpha,
        alpha_profile_mode="prefix",
    )


def _ctrl(**kw):
    return OnlineController(
        gamma_choices=[0, 1, 2, 3, 4],
        V=kw.pop("V", 50.0),
        min_tps=1.0,
        server_price_scale=0.0,
        policy="capacity_dpp",
        use_virtual_queues=kw.pop("use_virtual_queues", True),
        **kw,
    )


def test_capacity_dpp_gamma_is_exact_argmax():
    ctrl = _ctrl()
    ctrl.server_queue = 3.0
    ctrl.current_verifier_theta_f_ms_per_token = 0.5
    cl = _client(z_queue=2.0, device_queue=1.0, alpha=0.85)

    def brute() -> int:
        best_g, best_s = 0, ctrl.capacity_dpp_score(cl, 0, [])
        for g in ctrl.gamma_choices:
            if g == 0:
                continue
            prof = cl.positional_acceptance_profile(g, use_ucb=True)
            s = ctrl.capacity_dpp_score(cl, g, prof)
            if s > best_s + 1e-12:
                best_g, best_s = g, s
        return best_g

    assert ctrl._capacity_dpp_gamma(cl, use_ucb=True) == brute()


def test_capacity_dpp_no_queues_picks_largest_gamma():
    # with V>0 and all queue prices zero, S_i(gamma) = V*phi(gamma) is
    # strictly increasing in gamma -> pick the max action
    ctrl = _ctrl(use_virtual_queues=False)
    cl = _client(alpha=0.8)
    assert ctrl._capacity_dpp_gamma(cl, use_ucb=False) == 4


def test_capacity_dpp_scalar_closed_form_matches_discrete_under_scalar_model():
    ctrl = _ctrl()
    ctrl.server_queue = 2.0
    ctrl.current_verifier_theta_f_ms_per_token = 0.4
    for alpha in (0.5, 0.7, 0.9, 0.95):
        for zq in (0.0, 1.0, 5.0):
            cl = _client(z_queue=zq, device_queue=0.5, alpha=alpha)
            # discrete argmax under a FLAT scalar profile of length gamma
            best_g, best_s = 0, ctrl.capacity_dpp_score(cl, 0, [])
            for g in ctrl.gamma_choices:
                if g == 0:
                    continue
                s = ctrl.capacity_dpp_score(cl, g, [alpha] * g)
                if s > best_s + 1e-12:
                    best_g, best_s = g, s
            # closed form: grow while marginal benefit >= marginal cost
            g_cf = 0
            while g_cf < max(ctrl.gamma_choices) and ctrl.capacity_dpp_scalar_should_grow(cl, g_cf, alpha):
                g_cf += 1
            assert g_cf == best_g


# ------------------------------------------------------------------ 12, 13
def test_step_area_bounds_ordered_and_correct():
    x = [1.0, 2.0, 4.0]
    n = [10.0, 6.0, 3.0]
    b = step_area_bounds(x, n)
    # lower uses right value: 6*1 + 3*2 = 12 ; upper uses left: 10*1 + 6*2 = 22
    assert b["area_lower"] == pytest.approx(12.0)
    assert b["area_upper"] == pytest.approx(22.0)
    assert b["area_lower"] <= b["area_step_estimate"] <= b["area_upper"]


def test_ica_in_unit_interval_and_matches_hand_calc():
    x = [1.0, 2.0, 4.0]
    n = [10.0, 6.0, 3.0]
    val = ica(x, n, x_L=1.0, x_U=4.0, n_ref=10.0)
    # step (left-carried): 10 on [1,2), 6 on [2,4) -> area = 10*1 + 6*2 = 22
    # normalized: 22 / (10 * 3) = 0.7333...
    assert val == pytest.approx(22.0 / 30.0)
    assert 0.0 <= val <= 1.0


def test_ica_caps_at_n_ref():
    x = [0.5, 1.0]
    n = [1000.0, 1000.0]
    assert ica(x, n, x_L=0.5, x_U=1.0, n_ref=50.0) == pytest.approx(1.0)


def test_ica_rejects_nonpositive_x_L():
    with pytest.raises(ValueError):
        ica([0.0, 1.0], [5.0, 3.0], x_L=0.0, x_U=1.0, n_ref=10.0)


# ------------------------------------------------------------------ 14
def test_symmetric_nstar_is_monotone_nonincreasing_and_scales_like_inverse_x():
    from edge_specsim.nstar_symmetric import (
        SymmetricParams,
        nstar,
        nstar_curve,
        scaling_exponent,
    )

    p = SymmetricParams(
        alpha=0.8,
        gamma_choices=[0, 1, 2, 3, 4],
        theta_f_s_per_token=1e-3,
        theta0_s=5e-3,
        draft_s_per_token=1e-3,
        rtt_s=20e-3,
    )
    xs = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    curve = nstar_curve(xs, p)
    ns = [row["nstar"] for row in curve]
    assert all(b <= a for a, b in zip(ns, ns[1:]))  # P1 monotone nonincreasing
    s = scaling_exponent([x for x in xs if nstar(x, p)[0] > 0], p)
    assert 0.7 <= s <= 1.3  # Theta(1/x)


# =========================================================================
# CAPACITY_AWARE_FRAMEWORK_CODEX_IMPLEMENTATION.md "final mode" tests
# (Section 15).
# =========================================================================

def _cm(seed, per_client, x=1.0, n=None):
    n = n if n is not None else len(per_client)
    return CandidateMeasurement(
        scale=n, n_active=n, x_requirement=x, seed=seed,
        min_rate_tps=min(per_client.values()), per_client_rates=dict(per_client),
    )


# ---- classify_feasibility_final -----------------------------------------

def test_final_classifier_k1_is_heuristic_only():
    r = classify_feasibility_final([_cm(1, {0: 5.0, 1: 5.0})], x=1.0, n_clients=2)
    assert r.status == "heuristic_only"


def test_final_classifier_feasible_all_lcb_above_x():
    meas = [_cm(s, {0: 5.0 + 0.01 * s, 1: 5.2 + 0.01 * s}) for s in range(5)]
    r = classify_feasibility_final(meas, x=1.0, n_clients=2, confidence_delta=0.05)
    assert r.status == "feasible"
    assert r.min_lcb >= 1.0


def test_final_classifier_infeasible_all_ucb_below_x():
    meas = [_cm(s, {0: 0.5 + 0.01 * s, 1: 0.4 + 0.01 * s}) for s in range(5)]
    r = classify_feasibility_final(meas, x=1.0, n_clients=2, confidence_delta=0.05)
    assert r.status == "infeasible"
    assert r.min_ucb < 1.0


def test_final_classifier_uncertain_on_overlapping_bounds():
    # rates scattered right around x=1.0 with real seed-to-seed spread
    vals0 = [0.7, 1.3, 0.9, 1.1, 1.0]
    vals1 = [0.8, 1.2, 1.0, 0.9, 1.1]
    meas = [_cm(s, {0: vals0[s], 1: vals1[s]}) for s in range(5)]
    r = classify_feasibility_final(meas, x=1.0, n_clients=2, confidence_delta=0.05)
    assert r.status == "uncertain"


def test_final_classifier_never_relaxes_the_floor():
    # a rate exactly at x with essentially zero seed variance must not be
    # rejected by a (1-eps)*x floor -- it should read as feasible/uncertain,
    # never silently pass because the true test was against a lower number
    meas = [_cm(s, {0: 1.0, 1: 1.0}) for s in range(5)]
    r = classify_feasibility_final(meas, x=1.0, n_clients=2)
    assert r.status in {"feasible", "uncertain"}
    assert r.min_lcb <= 1.0 <= r.min_ucb + 1e-9


# ---- NStarResult / monotone_scale_search_final ---------------------------

def _decision(n, status):
    return SearchDecision(n=n, status=status, min_lcb=0.0, min_ucb=0.0, seeds_used=3)


def test_nstar_exact_monotone_boundary_recovered():
    def feas(n: int) -> SearchDecision:
        return _decision(n, "feasible" if n <= 7 else "infeasible")

    res = monotone_scale_search_final(feas, lo_feasible=1, hi_cap=30, x_requirement=2.0)
    assert res.last_confirmed_feasible_n == 7
    assert res.first_confirmed_infeasible_n == 8
    assert res.exact is True
    assert res.hit_search_cap is False


def test_nstar_cap_feasible_is_lower_bound_not_equality():
    def feas(n: int) -> SearchDecision:
        return _decision(n, "feasible")  # never infeasible up to the cap

    res = monotone_scale_search_final(feas, lo_feasible=1, hi_cap=14, x_requirement=2.0)
    assert res.hit_search_cap is True
    assert res.exact is False
    assert res.lower_bound_n == 14
    assert res.upper_bound_n is None


def test_nstar_uncertain_neighbor_gives_interval():
    def feas(n: int) -> SearchDecision:
        if n <= 5:
            return _decision(n, "feasible")
        if n == 6:
            return _decision(n, "uncertain")
        return _decision(n, "infeasible")

    res = monotone_scale_search_final(feas, lo_feasible=1, hi_cap=30, x_requirement=2.0)
    assert res.last_confirmed_feasible_n == 5
    assert 6 in res.uncertain_n
    assert res.exact is False


# ---- xstar_search ----------------------------------------------------

def test_xstar_bisection_matches_synthetic_oracle():
    true_boundary = 3.37

    def feas(x: float) -> SearchDecision:
        return _decision(0, "feasible" if x <= true_boundary else "infeasible")

    res = xstar_search(feas, n_active=8, x_lo=0.5, x_hi=6.0, x_tolerance_tps=0.05)
    assert res.exact_within_tolerance
    assert abs(res.estimate_x - true_boundary) <= 0.1


# ---- area from N*(x) vs x*(N): dual bounds ---------------------------

def test_area_from_nstar_known_staircase():
    results = [
        NStarResult(x_requirement=x, last_confirmed_feasible_n=n, first_confirmed_infeasible_n=n + 1,
                    lower_bound_n=n, upper_bound_n=n + 1, exact=True, hit_search_cap=False)
        for x, n in [(1.0, 10), (2.0, 10), (3.0, 6), (4.0, 6)]
    ]
    out = area_from_nstar(results, x_L=1.0, x_U=4.0, n_ref=30)
    # step (carry-forward): 10 on [1,3), 6 on [3,4) -> 10*2 + 6*1 = 26
    assert out["area_lower"] == pytest.approx(26.0)
    assert out["area_upper"] == pytest.approx(26.0)  # exact everywhere -> bounds coincide


def test_area_from_nstar_propagates_cap_and_uncertainty():
    results = [
        NStarResult(x_requirement=1.0, last_confirmed_feasible_n=14, first_confirmed_infeasible_n=None,
                    lower_bound_n=14, upper_bound_n=None, exact=False, hit_search_cap=True),
        NStarResult(x_requirement=4.0, last_confirmed_feasible_n=14, first_confirmed_infeasible_n=None,
                    lower_bound_n=14, upper_bound_n=None, exact=False, hit_search_cap=True),
    ]
    out = area_from_nstar(results, x_L=1.0, x_U=4.0, n_ref=30)
    assert out["any_cap_limited"] is True
    assert out["area_lower"] == pytest.approx(14.0 * 3.0)
    assert out["area_upper"] == pytest.approx(30.0 * 3.0)  # unresolved -> capped at n_ref


def test_area_from_nstar_refuses_extrapolation_outside_measured_grid():
    results = [
        NStarResult(x_requirement=2.0, last_confirmed_feasible_n=10, first_confirmed_infeasible_n=11,
                    lower_bound_n=10, upper_bound_n=11, exact=True, hit_search_cap=False),
    ]
    with pytest.raises(ValueError):
        area_from_nstar(results, x_L=0.5, x_U=4.0, n_ref=30)


def test_area_from_xstar_matches_area_from_nstar_on_consistent_data():
    # a simple step frontier: N*(x)=10 for x in [1,3), N*(x)=6 for x in [3,4]
    # <=> x*(N)=4 for N<=6, x*(N)=3 for 6<N<=10 (using x_U=4 as the cap for N<=6)
    nstar_results = [
        NStarResult(x_requirement=x, last_confirmed_feasible_n=n, first_confirmed_infeasible_n=n + 1,
                    lower_bound_n=n, upper_bound_n=n + 1, exact=True, hit_search_cap=False)
        for x, n in [(1.0, 10), (2.0, 10), (3.0, 6), (4.0, 6)]
    ]
    area_n = area_from_nstar(nstar_results, x_L=1.0, x_U=4.0, n_ref=10)

    xstar_results = []
    for n in range(1, 11):
        x_est = 4.0 if n <= 6 else 3.0
        xstar_results.append(
            XStarResult(n_active=n, lower_feasible_x=x_est, upper_infeasible_x=x_est,
                        estimate_x=x_est, exact_within_tolerance=True)
        )
    area_x = area_from_xstar(xstar_results, x_L=1.0, x_U=4.0, n_ref=10)
    assert area_x["area_lower"] == pytest.approx(area_n["area_lower"], rel=0.05)
