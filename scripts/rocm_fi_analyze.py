"""Tables for the ROCm fault isolation from result_*.json + frozen_contexts.json (CPU only)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rocm_fi_lib as lib  # noqa: E402

D = Path(sys.argv[1])
frozen = json.load(open(D / "frozen_contexts.json"))
orch = json.load(open(D / "orchestration.json"))
summary = {"orchestration": {k: orch[k] for k in ("total_calls_used", "elapsed_s", "not_tested") if k in orch}, "configs": {}}


def prefill_stats(S, run, ref=None):
    """agreement of a prefill run with fp32 (and optionally with a B=1 run)."""
    rows_all, rows_gap = [], []
    for i, r in run.items():
        i = int(i)
        for j in range(4):
            f = S[i]["fp32"][j]
            rows_all.append({"match": r["argmax"][j] == f["top_ids"][0], "fp32_gap": f["gap"]})
    out = {"fp32_all": lib.agreement(rows_all, 0.0), "fp32_gap>=0.5": lib.agreement(rows_all, lib.GAP_OK)}
    err = [abs(r["proposal_lp"][j] - S[int(i)]["fp32"][j]["proposal_logprob"]) for i, r in run.items() for j in range(4)]
    out["abs_proposal_logprob_err_vs_fp32"] = {"median": float(np.median(err)), "p90": float(np.percentile(err, 90)), "max": float(max(err))}
    # sampler next token vs fp32 argmax at the position after the 4 candidates is not in fp32 table: skip
    if ref is not None:
        common = [i for i in run if i in ref]
        eq = [run[i]["argmax"][j] == ref[i]["argmax"][j] for i in common for j in range(4)]
        d = [abs(run[i]["proposal_lp"][j] - ref[i]["proposal_lp"][j]) for i in common for j in range(4)]
        out["vs_B1"] = {"n_positions": len(eq), "argmax_equal": float(np.mean(eq)), "max_abs_proposal_logprob_diff": float(max(d)),
                        "median_abs_proposal_logprob_diff": float(np.median(d))}
    return out


def decode_stats(S, run, pre_same_batch=None):
    first = [r["tokens"][0] == S[int(i)]["proposal"][0] for i, r in run.items()]
    full = [r["tokens"][:4] == S[int(i)]["proposal"] for i, r in run.items()]
    out = {"n": len(run), "first_token_equals_fp32": float(np.mean(first)), "four_tokens_equal_fp32": float(np.mean(full))}
    if pre_same_batch:
        cons = [run[i]["tokens"][0] == pre_same_batch[i]["argmax"][0] for i in run if i in pre_same_batch]
        out["decode_first_token_equals_prefill_argmax_same_context"] = float(np.mean(cons)) if cons else None
    return out


def per_context_correct(S, run):
    return {int(i): all(r["argmax"][j] == S[int(i)]["fp32"][j]["top_ids"][0] for j in range(4) if S[int(i)]["fp32"][j]["gap"] >= lib.GAP_OK)
            for i, r in run.items()}


for name, res in sorted(orch["configs"].items()):
    r = json.load(open(D / f"result_{name}.json"))
    entry = {"status": r["status"], "error": r.get("error"), "spec": r.get("spec"), "versions": r.get("versions"),
             "engine": r.get("engine"), "env": r.get("env"), "init_s": r.get("init_s"), "calls": r.get("calls"),
             "wall_s": r.get("wall_s"), "latency_ms": r.get("latency_ms")}
    if r["status"] == "ok":
        for setname, key in (("dev", "dev"), ("heldout", "heldout")):
            if key not in r:
                continue
            S = frozen["sets"][setname]["contexts"]
            M = r[key]["measurements"]
            P = {k: {int(i): v for i, v in run.items()} for k, run in M["prefill"].items()}
            b1 = P.get("B1")
            pre = {k: prefill_stats(S, run, b1 if k != "B1" else None) for k, run in P.items()}
            dec = {k: decode_stats(S, {int(i): v for i, v in run.items()},
                                   P.get(k.replace("B16_ident", "B16_ident").replace("B4_ident", "B4_ident")) if k in ("B16_ident", "B16_perm", "B4_ident") else None)
                   for k, run in M["decode"].items()}
            slot = None
            if "B16_ident" in P and "B16_perm" in P:
                ca, cb = per_context_correct(S, P["B16_ident"]), per_context_correct(S, P["B16_perm"])
                slot = lib.slot_consistency(ca, cb, {i: P["B16_ident"][i]["slot"] for i in ca}, {i: P["B16_perm"][i]["slot"] for i in cb})
                by_slot_a = {}
                for i, ok in ca.items():
                    by_slot_a[P["B16_ident"][i]["group_pos"]] = by_slot_a.get(P["B16_ident"][i]["group_pos"], []) + [ok]
                by_slot_b = {}
                for i, ok in cb.items():
                    by_slot_b[P["B16_perm"][i]["group_pos"]] = by_slot_b.get(P["B16_perm"][i]["group_pos"], []) + [ok]
                slot["wrong_rate_by_group_position_ident"] = {str(k): float(1 - np.mean(v)) for k, v in sorted(by_slot_a.items())}
                slot["wrong_rate_by_group_position_perm"] = {str(k): float(1 - np.mean(v)) for k, v in sorted(by_slot_b.items())}
            entry[setname] = {"prefill": pre, "decode": dec, "slot_vs_input_B16": slot, "gate_inputs": r[key]["gate_inputs"]}
        entry["dev_gate_pass"] = r.get("dev_gate_pass")
        entry["heldout_gate_pass"] = r.get("heldout_gate_pass")
    summary["configs"][name] = entry

# predicted contamination for the failing path (code-derived), for reference
summary["naive_sdpa_prediction_equal_lengths"] = {
    f"B{b}": [i for i, s in enumerate(lib.naive_sdpa_status([162] * b)) if s != "correct"] for b in (3, 4, 16)}
Path(sys.argv[2]).write_text(json.dumps(summary, indent=1))
brief = {}
for n, e in summary["configs"].items():
    if e["status"] != "ok":
        brief[n] = {"status": e["status"], "error": (e.get("error") or "")[:200]}
        continue
    d = e["dev"]
    brief[n] = {"engine": e["engine"].get("engine_class"), "impl": e["engine"].get("attention_impls_first"),
                "prefill_vs_fp32_gap>=0.5": {k: v["fp32_gap>=0.5"]["rate"] for k, v in d["prefill"].items()},
                "vs_B1": {k: v.get("vs_B1", {}).get("argmax_equal") for k, v in d["prefill"].items()},
                "decode_4tok_vs_fp32": {k: v["four_tokens_equal_fp32"] for k, v in d["decode"].items()},
                "gate": e.get("dev_gate_pass"), "heldout_gate": e.get("heldout_gate_pass"), "calls": e["calls"]}
print(json.dumps(brief, indent=1))
