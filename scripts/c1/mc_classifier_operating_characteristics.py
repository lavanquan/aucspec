"""Monte-Carlo operating characteristics of the certified Student-t classifier vs the legacy percentile bootstrap (C1/M3).

Setup: N clients; the weak client has true mean mu_w = x - d, the others are far above x.  Between-run-seed noise
sd = sigma*mu scale; K run-seeds.  We report P(declare FEASIBLE) (false-feasible when d >= 0), P(declare
INFEASIBLE) when d < 0 is false... precisely:
  false_feasible = P(status == feasible) when the weakest true mean is <= x
  power_infeasible = P(status == infeasible) when the weakest true mean is < x - d
under normal, lognormal (skew), and Student-t(3) (heavy-tailed) noise, plus sequential escalation 5 -> 7 and the
effect of alpha-spending, plus the implied family-wise error over M independent boundary probes.
Outputs one JSON (default stdout).
"""
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "src"))

from edge_specsim.capacity import CandidateMeasurement  # noqa: E402
from edge_specsim.capacity_certified import (  # noqa: E402
    FEASIBLE,
    INFEASIBLE,
    bootstrap_sensitivity,
    classify_certified,
)

N, X, SIGMA_FRAC = 10, 3.0, 0.10
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 600


def noise(rng, dist):
    if dist == "normal":
        return rng.gauss(0, 1)
    if dist == "lognormal":                       # skewed, unit variance
        s = 0.6
        return (rng.lognormvariate(0, s) - math.exp(s * s / 2)) / math.sqrt((math.exp(s * s) - 1) * math.exp(s * s))
    if dist == "t3":                              # heavy tailed, unit variance
        z = rng.gauss(0, 1)
        chi = sum(rng.gauss(0, 1) ** 2 for _ in range(3))
        return z / math.sqrt(chi / 3) / math.sqrt(3)
    raise ValueError(dist)


def make(rng, k, mu_weak, dist):
    ms = []
    for s in range(k):
        rates = {}
        for c in range(N):
            mu = mu_weak if c == 0 else X * 2.0
            rates[c] = mu + SIGMA_FRAC * X * noise(rng, dist)
        ms.append(CandidateMeasurement(scale=N, n_active=N, x_requirement=X, seed=s, min_rate_tps=min(rates.values()),
                                       per_client_rates=rates))
    return ms


def rate(rng, k, d, dist, fn, trials=TRIALS):
    hits_f = hits_i = 0
    for _ in range(trials):
        st = fn(make(rng, k, X - d, dist))
        hits_f += st == FEASIBLE
        hits_i += st == INFEASIBLE
    return hits_f / trials, hits_i / trials


def main():
    rng = random.Random(20261020)
    out = {"setup": {"N": N, "x": X, "sigma_frac_of_x": SIGMA_FRAC, "delta": 0.05, "trials": TRIALS},
           "rows": [], "sequential": [], "multi_probe": []}
    student = lambda ms: classify_certified(ms, X, N, min_seeds=2).status                # K_min lifted: operating char.
    boot = lambda ms: bootstrap_sensitivity(ms, X, N, n_boot=300)["status"]
    for dist in ("normal", "lognormal", "t3"):
        for k in (3, 5, 7, 12):
            for d in (0.0, 0.1, -0.5):          # d>0 weak client below x (false-feasible possible); d<0 above x
                ft, it = rate(rng, k, d, dist, student)
                fb, ib = rate(rng, k, d, dist, boot, trials=max(100, TRIALS // 3))
                out["rows"].append({"dist": dist, "K": k, "d_below_x": d,
                                    "P_feasible_student_t": ft, "P_infeasible_student_t": it,
                                    "P_feasible_bootstrap": fb, "P_infeasible_bootstrap": ib})
    # sequential escalation: look at K=5,6,7 ; stop at first certified decision
    for spend in (False, True):
        for dist in ("normal", "t3"):
            false_f = 0
            for _ in range(TRIALS):
                ms = make(rng, 7, X, dist)           # weak client exactly at x
                dec = None
                for k in (5, 6, 7):
                    r = classify_certified(ms[:k], X, N, n_looks=3 if spend else 1)
                    if r.status in (FEASIBLE, INFEASIBLE):
                        dec = r.status
                        break
                false_f += dec == FEASIBLE
            out["sequential"].append({"dist": dist, "alpha_spending": spend, "P_false_feasible_any_look": false_f / TRIALS})
    for row in out["rows"]:
        if row["dist"] == "normal" and row["K"] == 5 and row["d_below_x"] == 0.0:
            p = row["P_feasible_student_t"]
            for m in (1, 3, 5, 10):
                out["multi_probe"].append({"M_probes": m, "P_any_false_feasible_if_independent": 1 - (1 - p) ** m})
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
