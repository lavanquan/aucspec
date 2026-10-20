"""OFFLINE verifier service-time predictors from the corrected-runtime C1 traces (C2 Phase C).  CPU only; nothing here is wired into serving.

IMPORTANT LABEL: these are LOADED-RUN traces (draft threads, other clients and the emulator share the node), not an independent hardware
calibration of the verifier.  Every feature used is known when the batch is FORMED (membership, context lengths, speculative depths); the
measured wall time is only the target - no look-ahead.  Held-out evaluation is by whole run (and by out-of-distribution slices)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(0)


def load() -> pd.DataFrame:
    b = pd.read_csv(ROOT / "results" / "c2" / "batches.csv.gz")
    b = b.rename(columns={"verification_batch_size": "B", "verification_batch_wall_ms": "wall", "verification_batch_processed_tokens": "proc",
                          "verification_batch_context_tokens_sum": "ctx_sum", "verification_batch_context_tokens_max": "ctx_max"})
    b["group"] = b["run"] + ":N" + b["N"].astype(str)
    b["ctx_max_x_B"] = b["ctx_max"] * b["B"]
    return b


MODELS = {
    "M0 wall~1+B": ["B"],
    "M1 theta_f*(gamma+1) positions (+intercept)": ["positions_sum"],
    "M1o theta_f*(gamma+1) positions (no intercept)": ["positions_sum"],
    "M2 full-context tokens (processed)": ["proc"],
    "M3 ctx_sum + B + gamma_sum": ["ctx_sum", "B", "gamma_sum"],
    "M4 M3 + padded ctx_max*B": ["ctx_sum", "B", "gamma_sum", "ctx_max_x_B"],
}


def design(df, cols, intercept=True):
    X = df[cols].to_numpy(float)
    return np.column_stack([np.ones(len(df)), X]) if intercept else X


def fit(df, name):
    cols = MODELS[name]
    inter = not name.startswith("M1o")
    X, y = design(df, cols, inter), df["wall"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, cols, inter


def predict(beta, cols, inter, df):
    return design(df, cols, inter) @ beta


def metrics(y, p):
    e = p - y
    ss = float(((y - y.mean()) ** 2).sum())
    return {"n": int(len(y)), "mae_ms": float(np.abs(e).mean()), "mape": float((np.abs(e) / y).mean()), "p90_abs_ms": float(np.quantile(np.abs(e), 0.9)),
            "bias_ms": float(e.mean()), "r2": float(1 - (e ** 2).sum() / ss) if ss > 0 else float("nan")}


def interval_coverage(train_y, train_p, y, p, level=0.9):
    """Prediction interval from the TRAINING residual quantiles; coverage measured on held-out data."""
    r = train_y - train_p
    lo, hi = np.quantile(r, (1 - level) / 2), np.quantile(r, 1 - (1 - level) / 2)
    return float(((y >= p + lo) & (y <= p + hi)).mean()), float(hi - lo)


def evaluate_split(df_tr, df_te):
    out = {}
    for name in MODELS:
        beta, cols, inter = fit(df_tr, name)
        p_te = predict(beta, cols, inter, df_te)
        p_tr = predict(beta, cols, inter, df_tr)
        m = metrics(df_te["wall"].to_numpy(float), p_te)
        cov, width = interval_coverage(df_tr["wall"].to_numpy(float), p_tr, df_te["wall"].to_numpy(float), p_te)
        m.update({"pi90_coverage": cov, "pi90_width_ms": width})
        out[name] = m
    return out


def main():
    b = load()
    res = {"label": "OFFLINE diagnostics on loaded-run traces; not an independent hardware calibration; not activated in production",
           "n_batches": int(len(b)), "runs": sorted(b["group"].unique().tolist())}
    # in-sample
    res["in_sample"] = {name: metrics(b["wall"].to_numpy(float), predict(*fit(b, name), b)) for name in MODELS}
    # leave-one-group-out (whole run held out)
    logo = {name: [] for name in MODELS}
    for g in b["group"].unique():
        tr, te = b[b["group"] != g], b[b["group"] == g]
        ev = evaluate_split(tr, te)
        for name in MODELS:
            logo[name].append((len(te), ev[name]))
    res["leave_one_run_out"] = {name: {"mae_ms": float(np.average([m["mae_ms"] for _, m in v], weights=[n for n, _ in v])),
                                        "mape": float(np.average([m["mape"] for _, m in v], weights=[n for n, _ in v])),
                                        "worst_run_mae_ms": float(max(m["mae_ms"] for _, m in v)),
                                        "pi90_coverage": float(np.average([m["pi90_coverage"] for _, m in v], weights=[n for n, _ in v]))}
                                 for name, v in logo.items()}
    # out-of-distribution slices
    res["ood"] = {
        "train B<=8 -> test B>=12 (batch-size extrapolation)": evaluate_split(b[b["B"] <= 8], b[b["B"] >= 12]),
        "train N<=16 -> test N>=24 (concurrency extrapolation)": evaluate_split(b[b["N"] <= 16], b[b["N"] >= 24]),
        "train GSM8K-test pilot -> test GSM8K-train runs (catalog shift)": evaluate_split(b[b["catalog"] == "GSM8K-test"], b[b["catalog"] == "GSM8K-train"]),
        "train cap-32 jobs -> test cap-16 jobs (job/config shift)": evaluate_split(b[b["serving"].str.contains("cap32")], b[b["serving"].str.startswith("cap16")]),
    }
    # coefficients + run-block bootstrap
    groups = b["group"].unique()
    for name in ("M3 ctx_sum + B + gamma_sum", "M1 theta_f*(gamma+1) positions (+intercept)", "M2 full-context tokens (processed)"):
        beta, cols, inter = fit(b, name)
        boots = []
        for _ in range(300):
            pick = rng.choice(groups, size=len(groups), replace=True)
            bb = pd.concat([b[b["group"] == g] for g in pick])
            boots.append(fit(bb, name)[0])
        boots = np.array(boots)
        names = (["intercept"] if inter else []) + cols
        res.setdefault("coefficients", {})[name] = {n_: {"estimate": float(beta[i]), "ci90_runblock_bootstrap": [float(np.quantile(boots[:, i], 0.05)), float(np.quantile(boots[:, i], 0.95))]}
                                                    for i, n_ in enumerate(names)}
    # marginal request cost: M3 (context-aware) vs the theta_f*(gamma+1) abstraction
    beta3, cols3, _ = fit(b, "M3 ctx_sum + B + gamma_sum")
    beta1, _, _ = fit(b, "M1 theta_f*(gamma+1) positions (+intercept)")
    marg = {}
    for c in (100, 200, 300, 450):
        for g in (0, 2, 4):
            marg[f"ctx={c},gamma={g}"] = {"M3_marginal_ms": float(beta3[1] * c + beta3[2] + beta3[3] * g), "theta_f_abstraction_ms": float(beta1[1] * (g + 1))}
    res["marginal_request_cost_ms"] = marg
    res["observed_ranges"] = {k: [float(b[k].min()), float(b[k].max())] for k in ("B", "ctx_sum", "ctx_max", "proc", "wall", "gamma_sum")}
    res["gamma_values_in_data"] = sorted(b["gamma_sum"].div(b["B"]).round(1).unique().tolist())[:10]
    out = ROOT / "results" / "c2" / "verifier_cost_model_offline.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"in_sample": res["in_sample"], "leave_one_run_out": res["leave_one_run_out"]}, indent=1))


if __name__ == "__main__":
    main()
