"""Operating characteristics of the Bonferroni percentile-bootstrap classifier vs a Bonferroni Student-t interval.

Weak client with true mean mu = x - d and between-seed SD sigma; K seeds. We report the probability that the
classifier says FEASIBLE (false-feasible when d > 0) using the same convention as capacity.classify_feasibility_final
(alpha = delta/N per client, two-sided percentile bootstrap of the K-seed mean, 2000 resamples).
"""
import json
import math
import sys

import numpy as np

rng = np.random.default_rng(1)
delta, N, x, sigma = 0.05, 10, 3.0, 0.3
n_boot, trials = 2000, 600


def tcrit(df, p):
    # inverse Student-t via bisection on the cdf using the incomplete beta through numpy-free series (small df)
    from math import lgamma, exp, sqrt

    def pdf(t):
        return exp(lgamma((df + 1) / 2) - lgamma(df / 2)) / sqrt(df * math.pi) * (1 + t * t / df) ** (-(df + 1) / 2)

    def cdf(t):
        # numerical integration of the pdf (Simpson) from 0 to |t|
        n = 2000
        h = abs(t) / n
        s = pdf(0) + pdf(abs(t)) + sum((4 if i % 2 else 2) * pdf(i * h) for i in range(1, n))
        c = 0.5 + s * h / 3 * (1 if t >= 0 else -1)
        return c

    lo, hi = 0.0, 200.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


res = {"setup": {"N": N, "x": x, "sigma": sigma, "delta": delta, "trials": trials, "n_boot": n_boot}, "rows": []}
for K in (3, 5, 7, 12):
    tc = tcrit(K - 1, 1 - delta / (2 * N))
    for d in (0.0, 0.1, 0.3, 0.6):
        mu = x - d
        feas_boot = feas_t = 0
        cov_boot = cov_t = 0
        for _ in range(trials):
            vals = rng.normal(mu, sigma, size=K)
            idx = rng.integers(0, K, size=(n_boot, K))
            boots = np.sort(vals[idx].mean(axis=1))
            a = delta / N
            lo = boots[int((a / 2) * (n_boot - 1))]
            hi = boots[int((1 - a / 2) * (n_boot - 1))]
            feas_boot += lo >= x
            cov_boot += lo <= mu <= hi
            m, s = vals.mean(), vals.std(ddof=1)
            lo_t, hi_t = m - tc * s / math.sqrt(K), m + tc * s / math.sqrt(K)
            feas_t += lo_t >= x
            cov_t += lo_t <= mu <= hi_t
        res["rows"].append({"K": K, "d_below_x": d, "P_feasible_bootstrap": feas_boot / trials, "P_feasible_t": feas_t / trials,
                            "per_client_coverage_bootstrap": cov_boot / trials, "per_client_coverage_t": cov_t / trials,
                            "nominal_per_client_coverage": 1 - delta / N})
print(json.dumps(res, indent=1))
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(json.dumps(res, indent=1))
