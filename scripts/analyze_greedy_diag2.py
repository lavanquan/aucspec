"""Second-pass analysis: accuracy of each vLLM path against the independent fp32 oracle, by B and by batch composition."""
import json
import sys
from pathlib import Path

import numpy as np

raw = json.load(open(sys.argv[1]))
ctxs, oracle = raw["contexts"], {int(k): v for k, v in raw["oracle_fp32"].items()}


def argmax_first(entry):
    return int(max(entry.items(), key=lambda kv: kv[1]["lp"])[0])


out = {}
# prefill (verifier path) vs fp32, by B and gamma
for key in sorted(raw["prefill"]):
    if raw["prefill"][key] is None:
        continue
    P = raw["prefill"][key]
    g = int(key.split("_g")[1])
    agree, err, cnt = [], [], 0
    for c, rec in P.items():
        for j in range(g):
            o = oracle[int(c)][j]
            e = rec["entries"][j]
            agree.append(argmax_first(e) == o["top_ids"][0])
            prop = ctxs[int(c)]["proposal"][j]
            err.append(abs(e[str(prop)]["lp"] - o["proposal_logprob"]))
    out[key] = {"verifier_argmax_equals_fp32_argmax": float(np.mean(agree)),
                "abs_err_proposal_logprob_vs_fp32_median": float(np.median(err)),
                "abs_err_p90": float(np.percentile(err, 90)), "abs_err_max": float(np.max(err)), "n": len(agree)}
# decode path (B=60 generate) vs fp32: proposal is argmax?  and reported logprob error
dec_err, dec_arg, dec_top = [], [], []
for c, e in enumerate(ctxs):
    for j in range(4):
        o = oracle[c][j]
        d = e["decode_logprobs"][j]
        prop = e["proposal"][j]
        dec_arg.append(o["top_ids"][0] == prop)
        dec_err.append(abs(d[str(prop)]["lp"] - o["proposal_logprob"]))
        top_id = max(d.items(), key=lambda kv: kv[1]["lp"])[0]
        dec_top.append(int(top_id) == prop)
out["decode_B60"] = {"proposal_equals_fp32_argmax": float(np.mean(dec_arg)),
                     "reported_top1_equals_sampled_token": float(np.mean(dec_top)),
                     "abs_err_sampled_logprob_vs_fp32_median": float(np.median(dec_err)),
                     "abs_err_p90": float(np.percentile(dec_err, 90)), "abs_err_max": float(np.max(dec_err))}
# fp32 self-consistency of the proposals: is the fp32 rank of each proposal token small?
ranks = [oracle[c][j]["proposal_rank"] for c in range(len(ctxs)) for j in range(4)]
out["proposal_rank_under_fp32"] = {"rank1": float(np.mean([r == 1 for r in ranks])), "rank<=2": float(np.mean([r <= 2 for r in ranks])),
                                   "rank<=5": float(np.mean([r <= 5 for r in ranks])), "max": int(max(ranks))}
# per-context: does each context's verifier error depend on position within the 16-batch?
P16 = raw["prefill"]["k2_B16_g4"]
by_slot = {}
for c, rec in P16.items():
    c = int(c)
    by_slot[c] = float(np.mean([argmax_first(rec["entries"][j]) == oracle[c][j]["top_ids"][0] for j in range(4)]))
out["B16_g4_fp32_agreement_by_batch_position"] = {
    "L158_group_pos0-15": [round(by_slot[i], 2) for i in range(16)],
    "L205_group_pos0-15": [round(by_slot[16 + i], 2) for i in range(16)]}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
