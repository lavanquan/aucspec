"""Fit and evaluate verifier cost models from calibration_raw.jsonl (CPU only).

Models (all on full-batch wall time, never wall/B):
  A  affine:     T = th0 + a*B + c*S_L + b*S_g          (S_L = sum of context lengths, S_g = sum of draft tokens)
  A' two parts:  generate_ms and host_ms fitted separately with the same regressors
  R  broken-stick on processed tokens W = S_L + S_g: A + d*max(0, W - w*)   (regime test; reported, not assumed)
Uncertainty: block bootstrap over timed rounds (repetition index), which keeps within-round drift intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(12345)


def load(path: Path):
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return recs


def design(r: dict):
    return [1.0, r["B"], r["sum_context"], r["sum_gamma"]]


def arrays(recs):
    X = np.array([design(r) for r in recs], float)
    y = np.array([r["wall_ms"] for r in recs], float)
    return X, y


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def metrics(y, yhat):
    res = y - yhat
    return {
        "n": int(len(y)), "rmse_ms": float(np.sqrt(np.mean(res ** 2))), "mae_ms": float(np.mean(np.abs(res))),
        "mean_residual_ms": float(np.mean(res)), "mape_pct": float(100 * np.mean(np.abs(res) / y)),
        "median_rel_err_pct": float(100 * np.median(np.abs(res) / y)),
        "resid_q05_q50_q95_ms": [float(q) for q in np.percentile(res, [5, 50, 95])],
    }


def boot_blocks(recs, fit_fn, n_boot=1000):
    """Resample timed rounds with replacement; fit_fn(list_of_recs) -> vector."""
    rounds = sorted({r["rep"] for r in recs})
    by_round = {k: [r for r in recs if r["rep"] == k] for k in rounds}
    draws = []
    for _ in range(n_boot):
        pick = RNG.choice(rounds, size=len(rounds), replace=True)
        sample = [r for k in pick for r in by_round[k]]
        draws.append(fit_fn(sample))
    return np.array(draws)


def ci(draws):
    return np.percentile(draws, [2.5, 97.5], axis=0)


def fit_affine(recs):
    X, y = arrays(recs)
    return ols(X, y)


def broken_stick(recs, w_grid):
    X, y = arrays(recs)
    W = np.array([r["processed_tokens"] for r in recs], float)
    best = None
    for w in w_grid:
        Xk = np.column_stack([X, np.maximum(0.0, W - w)])
        beta = ols(Xk, y)
        sse = float(np.sum((y - Xk @ beta) ** 2))
        if best is None or sse < best[0]:
            best = (sse, w, beta)
    return best


def predict_broken(beta, w, recs):
    X, _ = arrays(recs)
    W = np.array([r["processed_tokens"] for r in recs], float)
    return np.column_stack([X, np.maximum(0.0, W - w)]) @ beta


def paired_depth(recs, n_boot=1000):
    """Within-round paired comparison: (T(gamma) - T(0)) / (B*gamma) per (B, Lq, round)."""
    idx = {}
    for r in recs:
        idx[(r["B"], r["cell"].split("_")[1], r["cell"].split("_")[2], r["rep"])] = r["wall_ms"]
    rows = []
    for (B, L, g, rep), t in idx.items():
        if g == "g0":
            continue
        base = idx.get((B, L, "g0", rep))
        if base is None:
            continue
        gamma = int(g[1:])
        rows.append((B, L, gamma, rep, (t - base) / (B * gamma), t - base, B * gamma))
    arr = np.array([(r[0], r[2], r[3], r[4], r[5], r[6]) for r in rows], float)
    rounds = sorted(set(arr[:, 2]))

    def pooled(sel):
        # slope through the origin: sum(dT * Bg) / sum(Bg^2)
        return float(np.sum(sel[:, 4] * sel[:, 5]) / np.sum(sel[:, 5] ** 2))

    point = pooled(arr)
    draws = []
    for _ in range(n_boot):
        pick = RNG.choice(rounds, size=len(rounds), replace=True)
        sel = np.vstack([arr[arr[:, 2] == k] for k in pick])
        draws.append(pooled(sel))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    by_g = {}
    for g in (1, 2, 3, 4):
        s = arr[arr[:, 1] == g]
        by_g[str(g)] = {"mean_ms_per_draft_token_per_request": float(np.mean(s[:, 3])), "n": int(len(s))}
    return {"pooled_ms_per_draft_token_per_request": point, "ci95": [float(lo), float(hi)],
            "by_gamma": by_g, "n_pairs": int(len(arr))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-csv", nargs="*", default=[])
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    raw_path = Path(a.raw)
    recs = load(raw_path)
    sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    timed_all = [r for r in recs if r.get("phase") == "timed"]
    timed = [r for r in timed_all if r["well_formed"]]
    val = [r for r in recs if r.get("phase") == "val" and r["well_formed"]]
    warm = [r for r in recs if r.get("phase") == "warmup"]
    out: dict = {
        "raw_file": raw_path.name, "raw_sha256": sha,
        "counts": {"records": len(recs), "warmup": len(warm), "timed_primary": len(timed_all),
                   "timed_primary_well_formed": len(timed), "val_timed": len([r for r in recs if r.get("phase") == "val"]),
                   "val_well_formed": len(val),
                   "malformed_primary_cells": sorted({r["cell"] for r in timed_all if not r["well_formed"]})},
    }
    # --- model A on all primary data
    X, y = arrays(timed)
    beta = ols(X, y)
    names = ["theta0_ms", "per_request_const_ms", "per_context_token_ms", "per_draft_token_ms"]
    draws = boot_blocks(timed, fit_affine, a.n_boot)
    lo, hi = ci(draws)
    out["affine"] = {
        "coef": {n: float(b) for n, b in zip(names, beta)},
        "ci95": {n: [float(l), float(h)] for n, l, h in zip(names, lo, hi)},
        "train": metrics(y, X @ beta),
    }
    # derived components with uncertainty
    def comp(bv):
        return np.array([bv[0], bv[1] + bv[2] * 113, bv[1] + bv[2] * 158, bv[1] + bv[2] * 205, bv[3]])
    cd = np.array([comp(d) for d in draws])
    clo, chi = ci(cd)
    cp = comp(beta)
    labels = ["fixed_batch_overhead_ms", "per_request_cost_at_ctx113_ms", "per_request_cost_at_ctx158_ms",
              "per_request_cost_at_ctx205_ms", "marginal_cost_per_draft_token_ms"]
    out["components"] = {l: {"estimate": float(v), "ci95": [float(lo_), float(hi_)]} for l, v, lo_, hi_ in zip(labels, cp, clo, chi)}
    d_ci = out["affine"]["ci95"]["per_draft_token_ms"]
    out["draft_token_coefficient_indistinguishable_from_zero"] = bool(d_ci[0] <= 0.0 <= d_ci[1])

    # --- generate vs host split
    for key in ("generate_ms", "host_ms"):
        yy = np.array([r[key] for r in timed], float)
        bb = ols(X, yy)
        out[f"affine_{key}"] = {"coef": {n: float(b) for n, b in zip(names, bb)}, "train": metrics(yy, X @ bb)}

    # --- paired within-round depth comparison
    out["paired_depth"] = paired_depth(timed, a.n_boot)

    # --- held-out evaluations
    held = {}
    rounds = sorted({r["rep"] for r in timed})
    cut = int(len(rounds) * 0.7)
    tr = [r for r in timed if r["rep"] in rounds[:cut]]
    te = [r for r in timed if r["rep"] in rounds[cut:]]
    Xt, yt = arrays(tr)
    bt = ols(Xt, yt)
    Xe, ye = arrays(te)
    held["temporal_rounds_70_30"] = {"train": metrics(yt, Xt @ bt), "test": metrics(ye, Xe @ bt)}
    tr = [r for r in timed if r["B"] != 16]
    te = [r for r in timed if r["B"] == 16]
    if te:
        Xt, yt = arrays(tr)
        bt = ols(Xt, yt)
        Xe, ye = arrays(te)
        held["leave_B16_out_extrapolation"] = {"train": metrics(yt, Xt @ bt), "test": metrics(ye, Xe @ bt)}
    if val:
        Xv, yv = arrays(val)
        pv = Xv @ beta
        held["heterogeneous_validation"] = {"test": metrics(yv, pv), "by_B": {
            str(b): metrics(yv[[r["B"] == b for r in val]], pv[[r["B"] == b for r in val]])
            for b in sorted({r["B"] for r in val})}}
    out["held_out"] = held

    # --- regime test (broken stick on processed tokens)
    W = np.array([r["processed_tokens"] for r in timed], float)
    grid = np.unique(np.percentile(W, np.arange(10, 91, 5)))
    sse, w_star, bb = broken_stick(timed, grid)
    n = len(y)
    sse_a = float(np.sum((y - X @ beta) ** 2))
    bic_a = n * np.log(sse_a / n) + X.shape[1] * np.log(n)
    bic_r = n * np.log(sse / n) + (X.shape[1] + 2) * np.log(n)
    def fit_r(sample):
        s, w, b_ = broken_stick(sample, grid)
        return np.array([w, b_[-1]])
    rd = boot_blocks(timed, fit_r, max(200, a.n_boot // 4))
    rlo, rhi = ci(rd)
    # held-out comparison on the heterogeneous set
    reg_held = None
    if val:
        yv = np.array([r["wall_ms"] for r in val], float)
        rg = metrics(yv, predict_broken(bb, w_star, val))
        reg_held = rg
    out["regime_test"] = {
        "breakpoint_processed_tokens": float(w_star), "slope_change_ms_per_token": float(bb[-1]),
        "breakpoint_ci95": [float(rlo[0]), float(rhi[0])], "slope_change_ci95": [float(rlo[1]), float(rhi[1])],
        "delta_bic_broken_minus_affine": float(bic_r - bic_a), "train": metrics(y, predict_broken(bb, w_star, timed)),
        "heterogeneous_validation": reg_held,
        "grid_range": [float(grid[0]), float(grid[-1])],
        "supports_distinct_regimes": bool(
            (bic_r - bic_a) < -10 and not (rlo[1] <= 0 <= rhi[1]) and grid[0] < w_star < grid[-1]
            and (reg_held is None or reg_held["rmse_ms"] < out["held_out"]["heterogeneous_validation"]["test"]["rmse_ms"])),
        "note": "regime label is derived from data only; False means the affine model is adequate on this grid",
    }

    # --- residual structure by cell
    res_by_B = {}
    for b in sorted({r["B"] for r in timed}):
        sel = [r for r in timed if r["B"] == b]
        Xs, ys = arrays(sel)
        res_by_B[str(b)] = metrics(ys, Xs @ beta)
    out["residual_by_B"] = res_by_B
    # measurement noise: within-cell dispersion
    cv = {}
    for cell in sorted({r["cell"] for r in timed}):
        ws = np.array([r["wall_ms"] for r in timed if r["cell"] == cell], float)
        cv[cell] = float(np.std(ws, ddof=1) / np.mean(ws))
    out["within_cell_cv"] = {"median": float(np.median(list(cv.values()))), "max": float(max(cv.values()))}
    out["wall_vs_stamped_max_abs_diff_ms"] = float(max(abs(r["wall_ms"] - r["stamped_batch_wall_ms"]) for r in timed))

    # --- independent smoke-test comparison
    if a.smoke_csv:
        import pandas as pd
        sm = {}
        for p in a.smoke_csv:
            d = pd.read_csv(p, usecols=["verification_batch_id", "verification_batch_size", "verification_batch_wall_ms",
                                        "verification_batch_context_tokens_sum", "verification_batch_token_cost",
                                        "in_measurement_window"], low_memory=False).drop_duplicates("verification_batch_id")
            B = d["verification_batch_size"].to_numpy(float)
            SL = d["verification_batch_context_tokens_sum"].to_numpy(float)
            Sg = d["verification_batch_token_cost"].to_numpy(float) - B
            pred = beta[0] + beta[1] * B + beta[2] * SL + beta[3] * Sg
            wall = d["verification_batch_wall_ms"].to_numpy(float)
            res = wall - pred
            sm[Path(p).parent.parent.name + "/" + Path(p).name] = {
                "n_batches": int(len(d)), **metrics(wall, pred),
                "median_ratio_wall_over_pred": float(np.median(wall / pred)),
                "by_B": {str(int(b)): {"n": int((B == b).sum()), "mean_wall_ms": float(wall[B == b].mean()),
                                       "mean_pred_ms": float(pred[B == b].mean()),
                                       "mean_residual_ms": float(res[B == b].mean())} for b in np.unique(B)},
                "p99_residual_ms": float(np.percentile(res, 99)), "max_residual_ms": float(res.max())}
        out["smoke_comparison"] = sm

    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("counts", "components", "draft_token_coefficient_indistinguishable_from_zero")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
