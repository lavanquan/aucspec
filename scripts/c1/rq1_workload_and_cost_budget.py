"""RQ1 pilot: analytical workload-supply budget and GPU-cost model (CPU only; every input is a stated constant).

Workload.  A client can never consume tokens faster than its hard bound  x_max = (gamma+1) / T_cycle_min, where
T_cycle_min = fixed draft latency + gamma * tau_iso + RTT + batch_wait + B=1 verifier wall (the batch-wait floor applies
to every request).  Supply per client = questions_per_client * L, where L is the ACTUAL generated tokens per question,
L <= max_new_tokens and unknown in advance (EOS can shorten it).  We therefore tabulate the supply for several L and
require   qpc * L_scenario >= safety * x_max * T_stop   with T_stop = warmup + window + early-stop margin.

Cost.  Per (N, seed) run: startup + (T_stop * r(N)) + teardown, where r(N) = real seconds per virtual second.
Measured anchors: r = 0.54 (N=1, job sm_bw_N1: 230 s wall for 428 s virtual), r = 1.00 (N=8, tfa33 smoke: 128.8 s wall for
128.2 s virtual); startup = 130 s (job elapsed 360 s minus 230 s run).  r(N>8) and the causal-coordination overhead are
ASSUMPTIONS with stated ranges, to be measured by the first job.
"""
import json
import math
import sys

GAMMA = 2
TAU_ISO_MS = 22.2           # isolated draft ms/token (N=1 run: gamma=1..4 -> 22.0, 43.8, 66.0, ~88-90 ms)
FIXED_DRAFT_MS = 10.0
RTT_MS = 50.0
BATCH_WAIT_MS = 150.0
VERIFY_B1_MS = 78.5         # mean B=1 batch wall in the N=1 run
WARMUP_S, WINDOW_S, MARGIN_S = 30.0, 120.0, 10.0
SAFETY = 1.1                # x_max is a hard upper bound, so only a small margin is needed
LENGTHS = {"cap_96 (no EOS; observed 88/88 samples)": 96, "L=64": 64, "L=48": 48, "L=32": 32, "L=24": 24}
N_LIST = [1, 2, 4, 8, 12, 16]
DATASET_PROMPTS = 1319      # GSM8K test split (dataset.split: test in the canonical config)
R_MEASURED = {1: 0.54, 8: 1.00}
R_ASSUMED = {2: 0.8, 4: 0.9, 12: 1.3, 16: 1.6}
STARTUP_S, TEARDOWN_S = 130.0, 20.0


def workload():
    t_cycle = (FIXED_DRAFT_MS + GAMMA * TAU_ISO_MS + RTT_MS + BATCH_WAIT_MS + VERIFY_B1_MS) / 1000.0
    x_max = (GAMMA + 1) / t_cycle
    t_stop = WARMUP_S + WINDOW_S + MARGIN_S
    tokens_needed = SAFETY * x_max * t_stop
    rows = {name: math.ceil(tokens_needed / L) for name, L in LENGTHS.items()}
    return {"T_cycle_min_s": round(t_cycle, 4), "x_max_hard_bound_tps": round(x_max, 2), "T_stop_s": t_stop,
            "tokens_needed_per_client": round(tokens_needed, 1), "questions_needed_by_length": rows,
            "chosen_questions_per_client": 64,
            "min_mean_tokens_per_question_tolerated": round(tokens_needed / 64, 1),
            "prompts_required_at_N16": 16 * 64, "prompts_available": DATASET_PROMPTS,
            "prompts_ok": 16 * 64 <= DATASET_PROMPTS}


def cost(n_seeds, ns, coord_factor):
    per_seed = 0.0
    for n in ns:
        r = {**R_MEASURED, **R_ASSUMED}[n] * coord_factor
        per_seed += STARTUP_S + (WARMUP_S + WINDOW_S + MARGIN_S) * r + TEARDOWN_S
    return {"seconds_per_seed_sweep": round(per_seed), "node_hours": round(n_seeds * per_seed / 3600.0, 2)}


def main():
    out = {"inputs": {"gamma": GAMMA, "tau_iso_ms": TAU_ISO_MS, "rtt_ms": RTT_MS, "batch_wait_ms": BATCH_WAIT_MS,
                      "verify_b1_ms": VERIFY_B1_MS, "warmup_s": WARMUP_S, "window_s": WINDOW_S, "margin_s": MARGIN_S,
                      "r_measured": R_MEASURED, "r_assumed": R_ASSUMED, "startup_s": STARTUP_S},
           "workload": workload(), "cost": {}}
    for label, (seeds, ns) in {"full_6N_3seeds": (3, N_LIST), "reduced_4N_2seeds": (2, [1, 4, 8, 16])}.items():
        out["cost"][label] = {f"coord_factor_{c}": cost(seeds, ns, c) for c in (1.0, 2.0, 3.0)}
    # without early stop the run lasts until ALL questions are consumed
    no_stop_virtual = 64 * 96 / 2.4        # ~2560 s at the N=8 per-client rate 2.4 tok/s
    out["without_early_stop_virtual_seconds_per_run_at_N8"] = round(no_stop_virtual)
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
