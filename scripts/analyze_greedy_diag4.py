"""Remaining checks: k=1 vs k=2 acceptance at B=1, sampler next-token correctness by B, decode-vs-fp32 (CPU only)."""
import json
import sys
from pathlib import Path

import numpy as np

raw = json.load(open(sys.argv[1]))
ctxs, oracle = raw["contexts"], {int(k): v for k, v in raw["oracle_fp32"].items()}


def argmax_first(entry):
    return int(max(entry.items(), key=lambda kv: kv[1]["lp"])[0])


def accepted(entries, proposal):
    a = 0
    for e, p in zip(entries, proposal):
        if argmax_first(e) != p:
            break
        a += 1
    return a


out = {}
for B in (1, 4, 16):
    for g in (1, 2, 4):
        v1 = raw["v1_runs"][f"V1k1_B{B}_g{g}"]
        k2 = raw["prefill"][f"k2_B{B}_g{g}"]
        eq = [accepted(k2[c]["entries"], ctxs[int(c)]["proposal"][:g]) == r["accepted_length"] for c, r in v1.items()]
        out[f"V1(k=1)_vs_k2_dict_accepted_equal_B{B}_g{g}"] = {"n": len(eq), "equal_rate": float(np.mean(eq))}
# fp32 oracle acceptance (what a correct target-only verifier would give): accepted length of each proposal
fp32_acc = {}
for c in range(len(ctxs)):
    a = 0
    for j in range(4):
        if oracle[c][j]["top_ids"][0] != ctxs[c]["proposal"][j]:
            break
        a += 1
    fp32_acc[c] = a
out["fp32_true_greedy_accept_rate_of_the_B60_decode_proposals"] = {
    f"g{g}_full": float(np.mean([fp32_acc[c] >= g for c in fp32_acc])) for g in (1, 2, 4)}
# V1 exact acceptance versus the fp32 truth, by B
for B in (1, 4, 16):
    v1 = raw["v1_runs"][f"V1k1_B{B}_g4"]
    out[f"V1_g4_accepted_equals_fp32_truth_B{B}"] = float(np.mean([r["accepted_length"] == fp32_acc[int(c)] for c, r in v1.items()]))
# sampler next token (bonus) vs fp32 argmax at the next position (gamma = 1, 2 only; oracle holds positions 0..3)
for B in (1, 4, 16):
    for g in (1, 2):
        P = raw["prefill"][f"k2_B{B}_g{g}"]
        ok = [P[str(c)]["sampler_next_token"] == oracle[c][g]["top_ids"][0] for c in range(len(ctxs))]
        out[f"sampler_next_token_equals_fp32_argmax_B{B}_g{g}"] = float(np.mean(ok))
# B=1 prefill vs fp32: all positions
P1 = raw["prefill"]["k2_B1_g4"]
out["prefill_B1_g4_argmax_equals_fp32"] = float(np.mean([argmax_first(P1[str(c)]["entries"][j]) == oracle[c][j]["top_ids"][0] for c in range(len(ctxs)) for j in range(4)]))
# the four B=1 decode re-generations
out["decode_B1_vs_B60"] = raw["decode_b1"] and [{k: d[k] for k in ("slot", "same_as_B60", "history_same")} for d in raw["decode_b1"]]
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
