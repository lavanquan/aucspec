"""Summarise diag_raw.json from diagnose_greedy_verifier.py (CPU only)."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

raw = json.load(open(sys.argv[1]))
ctxs = raw["contexts"]
oracle = {int(k): v for k, v in raw.get("oracle_fp32", {}).items()}
S = {}


def argmax_first(entry):
    """V1 rule: first maximal entry in dict order."""
    best = max(entry.items(), key=lambda kv: kv[1]["lp"])
    return int(best[0])


def max_ties(entry):
    m = max(v["lp"] for v in entry.values())
    return sorted(int(k) for k, v in entry.items() if v["lp"] == m)


def top2(entry):
    s = sorted(((v["lp"], int(k)) for k, v in entry.items()), reverse=True)
    return s


# ---------- 1. exact-V1 (k=1 requests, unmodified verify_batch_sync) reproduction by B and gamma
rep = {}
for name, run in raw["v1_runs"].items():
    _, B, g = name.split("_")
    g = int(g[1:])
    accs = [(int(c), r["accepted_length"]) for c, r in run.items()]
    n = len(accs)
    rep[name] = {"n_requests": n, "full_accept_rate": sum(a == g for _, a in accs) / n,
                 "first_token_accept_rate": sum(a >= 1 for _, a in accs) / n,
                 "mean_accepted_fraction": float(np.mean([a / g for _, a in accs]))}
S["v1_exact_reproduction_by_B_gamma"] = rep
# B-invariance of the accepted length for identical (context, gamma)
inv = {}
for g in (1, 2, 4):
    a16 = {int(c): r["accepted_length"] for c, r in raw["v1_runs"][f"V1k1_B16_g{g}"].items()}
    a4 = {int(c): r["accepted_length"] for c, r in raw["v1_runs"][f"V1k1_B4_g{g}"].items()}
    a1 = {int(c): r["accepted_length"] for c, r in raw["v1_runs"][f"V1k1_B1_g{g}"].items()}
    inv[f"g{g}"] = {"B16_vs_B4_equal": sum(a16[c] == a4[c] for c in a16) / len(a16),
                    "B16_vs_B1_equal": sum(a16[c] == a1[c] for c in a1) / len(a1),
                    "B4_vs_B1_equal": sum(a4[c] == a1[c] for c in a1) / len(a1)}
S["v1_accepted_length_invariance_across_B"] = inv
# pool reproduction against the calibration job
flags = [(i, e["calibration_first_token_accepted"]) for i, e in enumerate(ctxs) if e["calibration_first_token_accepted"] is not None]
a16g4 = {int(c): r["accepted_length"] for c, r in raw["v1_runs"]["V1k1_B16_g4"].items()}
S["pool_reproduces_calibration_first_token_outcome"] = {
    "compared": len(flags), "equal": sum((a16g4[i] >= 1) == f for i, f in flags)}

# ---------- 2. per-position analysis (k=2, B=16, gamma=4: one prefill, full dictionaries)
P = raw["prefill"]["k2_B16_g4"]
rows = []
for c, e in enumerate(ctxs):
    rec = P[str(c)]
    for j in range(4):
        ent = rec["entries"][j]
        prop = e["proposal"][j]
        am = argmax_first(ent)
        ties = max_ties(ent)
        t2 = top2(ent)
        dec = e["decode_logprobs"][j]
        dec_t = top2(dec)
        o = oracle.get(c, [None] * 4)[j]
        rows.append({
            "ctx": c, "L": e["L"], "j": j, "proposal": prop, "verifier_argmax": am, "match": am == prop,
            "prop_lp": ent[str(prop)]["lp"] if str(prop) in ent else None,
            "prop_in_dict": str(prop) in ent, "n_entries": len(ent), "tie_ids": ties,
            "exact_tie_with_proposal": prop in ties and len(ties) > 1,
            "gap_top1_top2": (t2[0][0] - t2[1][0]) if len(t2) > 1 else None,
            "decode_top2_gap": (dec_t[0][0] - dec_t[1][0]) if len(dec_t) > 1 else None,
            "decode_top1_id": dec_t[0][1], "decode_argmax_is_proposal": dec_t[0][1] == prop,
            "oracle_argmax": o["top_ids"][0] if o else None,
            "oracle_gap": (o["top_logprobs"][0] - o["top_logprobs"][1]) if o else None,
            "oracle_proposal_rank": o["proposal_rank"] if o else None,
            "oracle_proposal_is_argmax": (o["top_ids"][0] == prop) if o else None,
            "oracle_verifier_argmax_is_oracle_argmax": (o["top_ids"][0] == am) if o else None,
            "verifier_argmax_in_oracle_top2": (am in o["top_ids"]) if o else None,
            "shift_m1_match": argmax_first(rec["entries_shift_minus1"][j]) == prop,
            "shift_p1_match": (argmax_first(rec["entries_shift_plus1"][j]) == prop) if rec["entries_shift_plus1"][j] else None,
        })
n = len(rows)
mism = [r for r in rows if not r["match"]]
S["positions_k2_B16_g4"] = {
    "n_positions": n, "mismatches": len(mism), "mismatch_rate": len(mism) / n,
    "by_offset": {j: sum(not r["match"] for r in rows if r["j"] == j) / sum(1 for r in rows if r["j"] == j) for j in range(4)},
    "proposal_always_in_dict": all(r["prop_in_dict"] for r in rows),
    "dict_sizes": dict(Counter(r["n_entries"] for r in rows)),
    "alignment": {"correct_index_match_rate": float(np.mean([r["match"] for r in rows])),
                  "shift_minus1_match_rate": float(np.mean([r["shift_m1_match"] for r in rows])),
                  "shift_plus1_match_rate": float(np.mean([r["shift_p1_match"] for r in rows if r["shift_p1_match"] is not None]))},
}
# classify mismatches
cls = Counter()
for r in mism:
    if r["exact_tie_with_proposal"]:
        cls["exact_logprob_tie_between_proposal_and_dict_first"] += 1
    elif r["gap_top1_top2"] is not None and r["gap_top1_top2"] < 0.2:
        cls["near_tie_gap_lt_0.2"] += 1
    else:
        cls["clear_disagreement"] += 1
S["mismatch_classes"] = dict(cls)
S["mismatch_details"] = [{k: r[k] for k in ("ctx", "L", "j", "proposal", "verifier_argmax", "tie_ids", "prop_lp",
                                            "gap_top1_top2", "decode_top2_gap", "oracle_argmax", "oracle_gap",
                                            "oracle_proposal_rank", "oracle_proposal_is_argmax",
                                            "oracle_verifier_argmax_is_oracle_argmax")} for r in mism]
S["mismatch_vs_decode"] = {
    "decode_argmax_is_proposal_all": all(r["decode_argmax_is_proposal"] for r in rows),
    "decode_top2_gap_at_mismatch": [r["decode_top2_gap"] for r in mism],
    "decode_top2_gap_at_matches_median": float(np.median([r["decode_top2_gap"] for r in rows if r["match"] and r["decode_top2_gap"] is not None])),
}
if oracle:
    S["oracle_fp32"] = {
        "proposal_is_fp32_argmax_rate": float(np.mean([r["oracle_proposal_is_argmax"] for r in rows])),
        "verifier_argmax_is_fp32_argmax_rate": float(np.mean([r["oracle_verifier_argmax_is_oracle_argmax"] for r in rows])),
        "at_mismatches_proposal_is_fp32_argmax": sum(bool(r["oracle_proposal_is_argmax"]) for r in mism),
        "at_mismatches_verifier_argmax_is_fp32_argmax": sum(bool(r["oracle_verifier_argmax_is_oracle_argmax"]) for r in mism),
        "fp32_top2_gap_at_mismatches": [round(r["oracle_gap"], 4) for r in mism],
        "fp32_top2_gap_quantiles_all": [float(x) for x in np.percentile([r["oracle_gap"] for r in rows], [5, 25, 50, 75])],
    }

# ---------- 3. does B change the k=2 prefill result? (same contexts, same gamma)
bi = {}
for g in (1, 2, 4):
    base = raw["prefill"][f"k2_B16_g{g}"]
    for B in (4, 1):
        other = raw["prefill"][f"k2_B{B}_g{g}"]
        am_eq, dl = [], []
        for c in base:
            for j in range(g):
                a, b = base[c]["entries"][j], other[c]["entries"][j]
                am_eq.append(argmax_first(a) == argmax_first(b))
                for t in a:
                    if t in b:
                        dl.append(abs(a[t]["lp"] - b[t]["lp"]))
        bi[f"g{g}_B16_vs_B{B}"] = {"argmax_equal_rate": float(np.mean(am_eq)), "max_abs_logprob_diff": float(max(dl)),
                                   "median_abs_logprob_diff": float(np.median(dl))}
S["prefill_batch_invariance_k2"] = bi
# gamma prefix invariance: does a longer candidate list change the first positions' logprobs?
gi = {}
for g in (1, 2):
    a, b = raw["prefill"][f"k2_B16_g{g}"], raw["prefill"]["k2_B16_g4"]
    d = [abs(a[c]["entries"][j][t]["lp"] - b[c]["entries"][j][t]["lp"]) for c in a for j in range(g)
         for t in a[c]["entries"][j] if t in b[c]["entries"][j]]
    gi[f"g{g}_vs_g4"] = {"max_abs_logprob_diff": float(max(d)), "argmax_equal_rate": float(np.mean(
        [argmax_first(a[c]["entries"][j]) == argmax_first(b[c]["entries"][j]) for c in a for j in range(g)]))}
S["prefill_gamma_prefix_consistency_k2"] = gi
# k=1 versus k=2 argmax
k1 = raw["prefill"].get("k1_B16_g4")
if k1:
    eq = [argmax_first(k1[str(c)]["entries"][j]) == argmax_first(P[str(c)]["entries"][j]) for c in range(len(ctxs)) for j in range(4)]
    S["k1_vs_k2_argmax_equal_rate"] = float(np.mean(eq))
    S["k1_dict_sizes"] = dict(Counter(len(k1[str(c)]["entries"][j]) for c in range(len(ctxs)) for j in range(4)))
# decode path batch invariance
S["decode_batch_invariance_B60_vs_B1"] = [
    {"slot": d["slot"], "proposal_same": d["same_as_B60"], "history_same": d["history_same"]} for d in raw["decode_b1"]]
# sampler (bonus) token vs dict argmax at the last candidate position and fp32 oracle (next-token after proposal)
S["calls_total"] = raw["meta"]["calls_total_before_oracle"] + (2 if oracle else 0)
Path(sys.argv[2]).write_text(json.dumps(S, indent=1))
print(json.dumps({k: v for k, v in S.items() if k not in ("mismatch_details",)}, indent=1))
