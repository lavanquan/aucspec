"""Is the batched-prefill error a misattribution of rows between sequences, or numerical corruption? (CPU only)"""
import json
import sys
from pathlib import Path

import numpy as np

raw = json.load(open(sys.argv[1]))
ctxs = raw["contexts"]
B1, B16, B4 = raw["prefill"]["k2_B1_g4"], raw["prefill"]["k2_B16_g4"], raw["prefill"]["k2_B4_g4"]


def keyset(e):
    return {int(k): v["lp"] for k, v in e.items()}


def same(a, b, tol=0.35):
    A, B = keyset(a), keyset(b)
    common = set(A) & set(B)
    return len(common) >= 1 and set(A) == set(B) and all(abs(A[t] - B[t]) < tol for t in common)


def analyse(Bx, name):
    n = len(ctxs)
    bad, hits, explained = [], 0, []
    for c in range(n):
        for j in range(4):
            ref = B1[str(c)]["entries"][j]
            got = Bx[str(c)]["entries"][j]
            if same(got, ref):
                continue
            bad.append((c, j))
            found = None
            for c2 in range(n):
                for j2 in range(4):
                    if (c2, j2) != (c, j) and same(got, B1[str(c2)]["entries"][j2]):
                        found = (c2, j2)
                        break
                if found:
                    break
            if found:
                hits += 1
                explained.append({"ctx": c, "j": j, "matches_B1_of_ctx": found[0], "pos": found[1]})
    total = n * 4
    return {"positions": total, "wrong_vs_B1": len(bad), "wrong_rate": len(bad) / total,
            "wrong_that_equal_another_sequences_correct_dict": hits, "examples": explained[:8],
            "wrong_by_ctx_slot_in_group": {str(c): sum(1 for (cc, _) in bad if cc == c) for c in range(n)}}


out = {"B16_vs_B1": analyse(B16, "B16"), "B4_vs_B1": analyse(B4, "B4")}
# cumulative flattened-token offset of each sequence in a 16-batch (for threshold hypotheses)
L = {c: len(ctxs[c]["context"]) + 4 for c in range(len(ctxs))}
off = {}
for g in range(0, 32, 16):
    tot = 0
    for c in range(g, g + 16):
        off[c] = tot
        tot += L[c]
bad = out["B16_vs_B1"]["wrong_by_ctx_slot_in_group"]
out["flattened_token_offset_vs_wrong"] = [{"ctx": c, "start_offset_tokens": off[c], "wrong_positions": bad[str(c)]} for c in range(32)]
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(json.dumps({k: (v if k != "flattened_token_offset_vs_wrong" else v[:16]) for k, v in out.items()}, indent=1))
