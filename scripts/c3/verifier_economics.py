"""C3 Stage B: full-context verifier economics - what the (preliminary) cost structure implies for the speculative-depth decision.
CPU/analytic.  usage: verifier_economics.py <out.json>

Cost structure (design eq. (S1)):  T_batch = a + sum_r (b*ctx_r + b_g*gamma_r).
  a, b           preliminary, from the loaded C1 runs (results/c2/verifier_cost_model_offline.json, M2); NOT an independent calibration.
  b_g            per-CANDIDATE-token verifier cost: NOT IDENTIFIED by the C1 traces (gamma ~ constant 2, collinear with batch size).  Every
                 result below is therefore a function of b_g; it is scanned over a range that includes the (wide) bootstrap CI of M3.
No constant here is presented as validation: the C1 fit only supplies the order of magnitude of (a, b) and the identifiability warning."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
A_MS, B_MS = 52.0, 0.19                      # preliminary (M2, loaded runs)
GAMMAS = [0, 1, 2, 3, 4]
BG_GRID = [0.0, 0.19, 0.76, 2.1, 5.5, 19.0]   # 0 (context-only cost) ... 5.5 (upper end of the M3 bootstrap CI) ... 19 (extreme)


def phi(alpha: float, g: int) -> float:
    """E[useful tokens per round] = (1 - alpha^(g+1)) / (1 - alpha)  (paper phi(gamma, alpha); g=0 -> the bonus token only)."""
    return float(g + 1) if abs(1 - alpha) < 1e-12 else (1 - alpha ** (g + 1)) / (1 - alpha)


def call_share_ms(ctx: float, g: int, B: int, bg: float, a: float = A_MS, b: float = B_MS) -> float:
    """Verifier time attributable to ONE request in a symmetric batch of B: fixed overhead amortised + its own full-context work."""
    return a / B + b * ctx + bg * g


def ms_per_useful(alpha, ctx, g, B, bg):
    return call_share_ms(ctx, g, B, bg) / phi(alpha, g)


def breakeven_bg(alpha, ctx, B, g_hi=4, g_lo=2):
    """b_g at which gamma=g_hi stops beating gamma=g_lo in verifier-ms per useful token (None = never: g_hi always better)."""
    A = A_MS / B + B_MS * ctx
    p_hi, p_lo = phi(alpha, g_hi), phi(alpha, g_lo)
    den = g_hi * p_lo - g_lo * p_hi
    if den <= 0:
        return None
    return A * (p_hi - p_lo) / den


def admission_cost_ms(ctx, g, bg):
    """Marginal batch time of admitting one more request (the overhead a is already paid)."""
    return B_MS * ctx + bg * g


def fluid_equilibrium(alphas, ctx, B, bg, gam, tau_d, rtt=50.0, x_req=3.0):
    """Closed-loop saturated fluid model: N clients, one verifier serving batches of B (round-robin), common context.
    Returns per-client rates (tok/s), goodput, verifier ms per useful token, mean round time."""
    N = len(alphas)
    d = np.array([B_MS * ctx + bg * g for g in gam])
    T = A_MS + (B / N) * d.sum() if N >= B else A_MS + d.sum()          # mean batch time (a batch holds B of the N clients)
    cyc = (N / B) * T if N >= B else T
    R = np.maximum(cyc, np.array(gam) * tau_d + rtt + T)                # verifier-bound vs device-bound round time (ms)
    ph = np.array([phi(a, g) for a, g in zip(alphas, gam)])
    x = ph / R * 1000.0
    served = min(N, B)
    cost = T / (served / N * ph.sum() if N >= B else ph.sum())
    return {"x_min": float(x.min()), "x_mean": float(x.mean()), "Y": float(x.sum()), "verifier_ms_per_useful": float(cost),
            "round_ms_mean": float(R.mean()), "deficit_sum": float(np.maximum(0.0, x_req - x).sum())}


def policies(alphas, ctx, B, bg, tau_d):
    out = {}
    N = len(alphas)
    for g in (1, 2, 3, 4):
        out[f"fixed{g}"] = [g] * N
    # per-client argmin of verifier-ms per useful token (price-only policy)
    out["vmin"] = [min(GAMMAS, key=lambda g: ms_per_useful(a, ctx, g, B, bg)) for a in alphas]
    # per-client best response in OWN rate given the fixed-point cycle (device latency included), 6 iterations
    gam = [2] * N
    for _ in range(6):
        new = []
        for i, a in enumerate(alphas):
            best, bx = 0, -1
            for g in GAMMAS:
                trial = list(gam)
                trial[i] = g
                x = fluid_equilibrium(alphas, ctx, B, bg, trial, tau_d)["x_mean"]  # mean is smooth; own-rate below
                d = np.array([B_MS * ctx + bg * q for q in trial])
                T = A_MS + (B / N) * d.sum()
                R = max((N / B) * T, g * tau_d + 50.0 + T)
                own = phi(a, g) / R
                if own > bx + 1e-15:
                    best, bx = g, own
            new.append(best)
        if new == gam:
            break
        gam = new
    out["own_best_response"] = gam
    return out


def main(out_path: str) -> None:
    res: dict = {"label": "Analytic economics under the preliminary (a, b) and a SCANNED, unidentified per-candidate-token cost b_g. "
                          "Not a calibration, not a GPU result."}
    # 1. total batch time, 2. admission cost, 3. gamma-change cost
    res["batch_time_ms"] = {f"B={B},ctx={c}": {f"bg={bg}": round(A_MS + B * (B_MS * c + bg * 2), 1) for bg in BG_GRID} for B in (4, 8, 16) for c in (100, 200, 400)}
    res["admission_cost_ms"] = {f"ctx={c},g=2": {f"bg={bg}": round(admission_cost_ms(c, 2, bg), 2) for bg in BG_GRID} for c in (100, 200, 400)}
    res["avg_vs_marginal_per_request_ms"] = {f"B={B},ctx=160,g=2": {"average_share": round(call_share_ms(160, 2, B, 0.19), 1),
                                                                    "marginal_admission": round(admission_cost_ms(160, 2, 0.19), 1)} for B in (2, 4, 8, 16)}
    res["gamma_increment_cost_ms_per_request"] = {f"bg={bg}": {"gamma2_to_4": round(2 * bg, 2)} for bg in BG_GRID}
    # 4. yield and verifier calls for continued decoding (L = 96 new tokens)
    L = 96
    res["verifier_calls_for_96_tokens"] = {f"alpha={a}": {f"g={g}": round(L / phi(a, g), 1) for g in GAMMAS} for a in (0.5, 0.7, 0.85, 0.92)}
    # 5. verifier-ms per useful token and the optimal depth
    tab, opt = {}, {}
    for B in (8, 16):
        for c in (100, 200, 400):
            for a in (0.4, 0.6, 0.8, 0.92):
                key = f"B={B},ctx={c},alpha={a}"
                tab[key] = {f"bg={bg}": {f"g={g}": round(ms_per_useful(a, c, g, B, bg), 2) for g in GAMMAS} for bg in (0.19, 5.5)}
                opt[key] = {f"bg={bg}": int(min(GAMMAS, key=lambda g: ms_per_useful(a, c, g, B, bg))) for bg in BG_GRID}
    res["verifier_ms_per_useful_token"] = tab
    res["argmin_depth_per_verifier_ms"] = opt
    res["breakeven_bg_gamma4_vs_gamma2_ms_per_token"] = {
        f"B={B},ctx={c},alpha={a}": (None if breakeven_bg(a, c, B) is None else round(breakeven_bg(a, c, B), 2))
        for B in (8, 16) for c in (100, 200, 400) for a in (0.3, 0.4, 0.6, 0.8, 0.92)}
    # context monotonicity: with an affine cost the cost-minimising depth is non-DEcreasing in context
    res["context_effect_on_optimal_depth"] = {f"alpha={a},bg={bg}": [int(min(GAMMAS, key=lambda g: ms_per_useful(a, c, g, 16, bg))) for c in (50, 100, 200, 400, 800)]
                                              for a in (0.4, 0.6, 0.8) for bg in (0.19, 5.5)}
    # candidate policies in the saturated fluid model (heterogeneous alpha, heterogeneous device speed)
    alphas = [0.60 + 0.35 * ((i * 0.6180339887) % 1.0) for i in range(24)]
    pol = {}
    for bg in (0.0, 0.19, 5.5, 19.0):
        for ctx in (160, 400):
            for tau_d in (22.2, 50.0):
                key = f"bg={bg},ctx={ctx},tau_d={tau_d},B=16,N=24"
                pol[key] = {name: dict(fluid_equilibrium(alphas, ctx, 16, bg, gam, tau_d), gamma_mean=round(float(np.mean(gam)), 2))
                            for name, gam in policies(alphas, ctx, 16, bg, tau_d).items()}
    res["policies_fluid_saturated"] = pol
    Path(out_path).write_text(json.dumps(res, indent=1, default=float))
    print("breakeven b_g (gamma4 vs gamma2), B=16, ctx=160:", {a: (None if breakeven_bg(a, 160, 16) is None else round(breakeven_bg(a, 160, 16), 2)) for a in (0.3, 0.4, 0.6, 0.8, 0.92)})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "c3" / "verifier_economics.json"))
