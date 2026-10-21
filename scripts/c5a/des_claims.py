"""C5-A: test the candidate finite-cap capacity formulas on the independent mini DES (no project simulator, no tuning).  usage: des_claims.py <out.json>

Candidate formulas for the long-run per-client cycle R (= time between a client's consecutive completions), homogeneous clients:
  F_sat    : R = ceil(N/B) a + N d                                   (C4 "saturated law")
  F_maxplus: R = max{ ceil(N/B) a + N d ,  D + W + a + B' d }        (max-plus cycle time of the fixed greedy partition; B' = size of the largest batch)
  F_frac   : R = max{ Keff a + N d , D + W + a + B d } , Keff = N/B for N >= 2B, 2 for B < N < 2B ;  N <= B: R = D + W + a + N d
Rates are x_i = phi_i / R.  A formula 'holds' if the DES mean gap is within 0.5 %.  Uniformity = max/min per-client mean gap (1.0 = every client served at the same frequency)."""
import collections
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mini_des import greedy_partition, simulate  # noqa: E402


def gaps(res, t0, t1):
    last, g = {}, collections.defaultdict(list)
    for (s, f, ids) in res.batches:
        if t0 <= f <= t1:
            for i in ids:
                if i in last:
                    g[i].append(f - last[i])
                last[i] = f
    per = [sum(v) / len(v) for v in g.values() if v]
    allg = [x for v in g.values() for x in v]
    bs = [len(ids) for (s, f, ids) in res.batches if t0 <= f <= t1]
    return sum(allg) / len(allg), min(per), max(per), sum(bs) / len(bs)


def f_sat(N, B, a, d, D, W):
    return -(-N // B) * a + N * d


def f_maxplus(N, B, a, d, D, W):
    part = greedy_partition(N, B)
    return max(len(part) * a + N * d, D + W + a + max(part) * d)


def f_frac(N, B, a, d, D, W):
    if N <= B:
        return D + W + a + N * d
    K = max(2.0, N / B)
    return max(K * a + N * d, D + W + a + B * d)


def regime(N, B):
    return "N<=B" if N <= B else ("B<N<=2B" if N <= 2 * B else "N>2B")


def main(out):
    rows = []
    for N, B, a, d, D, W in itertools.product((8, 12, 16, 17, 20, 24, 28, 32, 33, 36, 40, 48, 49), (8, 16), (10.0, 52.0), (5.0, 30.0), (1.0, 50.0, 300.0, 1000.0), (0.0, 150.0)):
        r = simulate(N, B, a, d, D, W=W, t_max=300000.0)
        m, lo, hi, ms = gaps(r, 150000.0, 290000.0)
        rows.append({"N": N, "B": B, "a": a, "d": d, "D": D, "W": W, "R": m, "gap_min": lo, "gap_max": hi, "mean_batch": ms, "regime": regime(N, B),
                     "F_sat": f_sat(N, B, a, d, D, W), "F_maxplus": f_maxplus(N, B, a, d, D, W), "F_frac": f_frac(N, B, a, d, D, W)})
    tol = 5e-3
    summary = {"label": "Independent deterministic closed DES (homogeneous clients, hold_head work-conserving FCFS); 832 parameter combinations; formula holds if within 0.5 %.", "n": len(rows), "by_regime": {}}
    for reg in ("N<=B", "B<N<=2B", "N>2B"):
        sub = [x for x in rows if x["regime"] == reg]
        e = {"n": len(sub)}
        for f in ("F_sat", "F_maxplus", "F_frac"):
            e[f + "_holds"] = sum(abs(x["R"] - x[f]) <= tol * x[f] for x in sub) / len(sub)
        e["uniform_service_fraction"] = sum(x["gap_max"] <= 1.005 * x["gap_min"] for x in sub) / len(sub)
        for w in (0.0, 150.0):
            s2 = [x for x in sub if x["W"] == w]
            for f in ("F_sat", "F_maxplus", "F_frac"):
                e[f"{f}_holds_W{int(w)}"] = sum(abs(x["R"] - x[f]) <= tol * x[f] for x in s2) / len(s2)
        summary["by_regime"][reg] = e
    summary["failures_F_frac"] = [x for x in rows if abs(x["R"] - x["F_frac"]) > tol * x["F_frac"]][:25]
    # saturated, uniform subregion where the C4 law is exact
    sat_uniform = [x for x in rows if x["regime"] == "B<N<=2B" and x["D"] <= 300 and x["W"] == 0.0]
    summary["sat_ceil_exact_B<N<=2B_W0_D<=300"] = {"n": len(sat_uniform), "holds": sum(abs(x["R"] - x["F_sat"]) <= tol * x["F_sat"] for x in sat_uniform) / max(1, len(sat_uniform))}
    Path(out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    for k, v in summary["by_regime"].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if "W" not in a})
    print(summary["sat_ceil_exact_B<N<=2B_W0_D<=300"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "des_claims.json")
