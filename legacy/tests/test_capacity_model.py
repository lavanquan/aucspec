"""CPU-only tests for the AUC_SPEC Section 3 symmetric reference model."""

from __future__ import annotations

import math

import pytest

from edge_specsim.capacity_model import RooflineParams, SymmetricSystem, fit_roofline, phi
from edge_specsim.nstar_symmetric import SymmetricParams, nstar


def _system(**kw) -> SymmetricSystem:
    roofline = kw.pop("roofline", RooflineParams(theta_0_ms=5.0, a_mem_ms=40.0, kv_ms_per_ctx=0.002, theta_f_ms=0.8))
    base = dict(alpha=0.7, gamma_choices=[0, 1, 2, 3, 4], roofline=roofline, context_length=300.0,
                draft_ms_per_token=10.0, draft_fixed_ms=5.0, rtt_ms=40.0,
                kappa_bits=32.0, bandwidth_hz=1e6, spectral_efficiency=2.0)
    base.update(kw)
    return SymmetricSystem(**base)


def test_phi_matches_series_and_alpha_one_limit():
    assert phi(3, 0.5) == pytest.approx(1 + 0.5 + 0.25 + 0.125)
    assert phi(4, 1.0) == pytest.approx(5.0)
    assert phi(0, 0.9) == pytest.approx(1.0)


def test_roofline_knee_separates_regimes():
    s = _system()
    for g in (1, 2, 4):
        knee = s.n_knee(g)
        assert knee is not None
        below, above = max(1, math.floor(knee) - 1), math.ceil(knee) + 1
        assert s.roofline.regime(below, g, s.context_length) == "memory"
        assert s.roofline.regime(above, g, s.context_length) == "compute"
    # deeper speculation moves the knee to smaller N (Section 3.2)
    assert s.n_knee(4) < s.n_knee(1)


def test_no_knee_when_compute_slope_below_memory_slope():
    s = _system(roofline=RooflineParams(theta_0_ms=5.0, a_mem_ms=40.0, kv_ms_per_ctx=1.0, theta_f_ms=0.1))
    assert s.n_knee(0) is None


def test_proposition1_closed_form_matches_scan_in_compute_regime():
    s = _system()
    g = 2
    for x in (0.5, 1.0, 2.0, 4.0):
        n_closed = s.n_gamma_closed_form(x, g, "compute")
        if n_closed <= (s.n_knee(g) or 0):
            continue  # maximizing N not in the compute regime
        n_scan = max((n for n in range(1, 2000) if s.x_gamma(n, g) >= x), default=0)
        assert n_closed == n_scan


def test_capacity_scales_like_one_over_x():
    s = _system()
    # Corollary 1: x * N_gamma*(x) -> phi / c_gamma as x -> 0
    g = 2
    _, c = s.affine(g, "compute")
    limit = 1000.0 * phi(g, s.alpha) / c
    small_x = 0.01
    assert small_x * s.n_gamma_closed_form(small_x, g, "compute") == pytest.approx(limit, rel=0.02)


def test_n_sym_is_nonincreasing_in_x_and_matches_inverse_of_x_sym():
    s = _system()
    xs = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    n_max = 5000
    ns = [s.n_sym(x, n_max=n_max) for x in xs]
    assert all(a >= b for a, b in zip(ns, ns[1:]))
    for x, n in zip(xs, ns):
        assert n < n_max
        if n > 0:
            assert s.x_sym(n)[0] >= x
        assert s.x_sym(n + 1)[0] < x


def test_network_term_consumes_concurrency():
    fast = _system(bandwidth_hz=math.inf)
    slow = _system(bandwidth_hz=2e3)
    # at a fixed speculative depth, uplink time is part of the per-client slope
    assert slow.n_gamma_closed_form(1.0, 2, "compute") < fast.n_gamma_closed_form(1.0, 2, "compute")
    # target-only rounds send no speculative payload
    assert slow.c_net_ms(0) == 0.0


def test_reduces_to_compute_only_model():
    # no memory term, no network: same N*(x) as nstar_symmetric
    roof = RooflineParams(theta_0_ms=0.0, a_mem_ms=0.0, kv_ms_per_ctx=0.0, theta_f_ms=2.0)
    s = _system(roofline=roof, draft_ms_per_token=0.0, draft_fixed_ms=0.0, rtt_ms=30.0, bandwidth_hz=math.inf)
    p = SymmetricParams(alpha=0.7, gamma_choices=[0, 1, 2, 3, 4], theta_f_s_per_token=0.002, rtt_s=0.03)
    for x in (1.3, 2.7, 5.1):  # off exact-equality boundaries
        assert s.n_sym(x, n_max=1000) == nstar(x, p)[0]


def test_fit_recovers_roofline_parameters():
    true = RooflineParams(theta_0_ms=20.0, a_mem_ms=60.0, kv_ms_per_ctx=0.02, theta_f_ms=1.5)
    b, g, sctx, t = [], [], [], []
    for batch in (1, 2, 4, 8, 16, 32):
        for gamma in (0, 2, 4, 8):
            for ctx in (128, 512, 1024):
                b.append(batch)
                g.append(batch * (gamma + 1))
                sctx.append(batch * ctx)
                t.append(true.t_v_ms(batch, gamma, ctx))
    fit = fit_roofline(b, g, sctx, t)
    assert fit["r2"] > 0.999
    assert fit["theta_f_ms"] == pytest.approx(true.theta_f_ms, rel=0.02)
    assert fit["kv_ms_per_ctx"] == pytest.approx(true.kv_ms_per_ctx, rel=0.05)
    assert fit["theta_0_ms"] + fit["a_mem_ms"] == pytest.approx(true.theta_0_ms + true.a_mem_ms, rel=0.05)
    assert fit["compute_regime_sampled"] is True
