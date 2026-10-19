"""Analytical workload-exhaustion budget for a candidate (N, window) design (CPU only, no simulator needed).

A client can consume at most x_max(N) = min(phi(gamma, alpha_hi) / T_cycle_min, Y_cap / N) tokens/s, where T_cycle_min is
the cycle time of a lone request (draft + RTT + batch wait + one-request verifier wall) and Y_cap an upper bound on the
verifier-limited aggregate goodput.  To be sure no client runs out of prompts before the measurement window ends, the
per-client token supply must satisfy  questions_per_client * tokens_per_question >= safety * x_max(N) * (warmup + window).
"""
import json
import math
import sys


def phi(gamma, alpha):
    return float(gamma + 1) if alpha >= 1 else (1 - alpha ** (gamma + 1)) / (1 - alpha)


def budget(ns, gamma, alpha_hi, draft_ms, rtt_ms, batch_wait_ms, verify_wall_b1_ms, y_cap_tps, warmup_s, window_s, tokens_per_question, safety=1.5):
    t_min_s = (draft_ms + rtt_ms + batch_wait_ms + verify_wall_b1_ms) / 1000.0
    rows = []
    for n in ns:
        x_max = min(phi(gamma, alpha_hi) / t_min_s, y_cap_tps / n)
        need_tokens = safety * x_max * (warmup_s + window_s)
        rows.append({"N": n, "x_max_tps": round(x_max, 2), "tokens_needed": round(need_tokens, 1),
                     "questions_per_client_needed": math.ceil(need_tokens / tokens_per_question)})
    return {"t_cycle_min_s": round(t_min_s, 3), "rows": rows, "questions_per_client_required": max(r["questions_per_client_needed"] for r in rows)}


if __name__ == "__main__":
    out = budget(ns=[1, 2, 4, 8, 12, 16], gamma=2, alpha_hi=0.9, draft_ms=150.0, rtt_ms=50.0, batch_wait_ms=150.0,
                 verify_wall_b1_ms=145.0, y_cap_tps=30.0, warmup_s=30.0, window_s=120.0, tokens_per_question=96)
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(json.dumps(out, indent=1))
