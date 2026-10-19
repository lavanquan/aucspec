"""Qualification tables (CPU only). Every rate is ok/total; raw and margin-filtered (fp32 top-2 gap >= 0.5)."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rocm_qual_lib as q  # noqa: E402

D = Path(sys.argv[1])
res = json.load(open(D / "result_qualification.json"))
frozen = json.load(open(D / "frozen_qualification_contexts.json"))
out = {"status": res["status"], "calls": res["calls"], "init_s": res.get("init_s"), "latency_ms": res.get("latency_ms"),
       "runtime_evidence": res.get("runtime_evidence"), "runtime_evidence_end": res.get("runtime_evidence_end"),
       "provenance": {k: res["provenance"][k] for k in ("git_sha", "git_dirty", "resolved_config_sha256")} if res.get("provenance") else None}


def frac(ok, tot):
    return f"{ok}/{tot}"


def analyse_set(setname):
    S = frozen["sets"][setname]["contexts"]
    block = res.get(setname)
    if not block:
        return None
    runs = block["runs"]
    b1 = {int(i): r for i, r in runs.get("true_B1_g4_ident", {}).get("records", {}).items()}
    table, mism, slotinfo = {}, [], {}
    for k, run in sorted(runs.items()):
        it = run["item"]
        kind, B, g = it["kind"], it["B"], it["gamma"]
        recs = {int(i): r for i, r in run["records"].items()}
        if kind == "true":
            raw = [0, 0]; mar = [0, 0]; bon = [0, 0]; acc = [0, 0]; com = [0, 0]; vb1 = [0, 0]
            groups = defaultdict(lambda: [0, 0, 0, 0])
            rows = []
            for i, r in recs.items():
                f = S[i]["fp32"]
                tg = r["target_greedy"]
                ctx_ok_margin = True
                for j in range(g):
                    match = j < len(tg) and tg[j] == f[j]["top_ids"][0]
                    raw[0] += match; raw[1] += 1
                    groups[S[i]["group"]][0] += match; groups[S[i]["group"]][1] += 1
                    if f[j]["gap"] >= q.GAP_OK:
                        mar[0] += match; mar[1] += 1
                        groups[S[i]["group"]][2] += match; groups[S[i]["group"]][3] += 1
                        ctx_ok_margin &= match
                    if not match:
                        lp = r["lp_max_and_proposed"][j] if j < len(r["lp_max_and_proposed"]) else None
                        mism.append({"run": k, "ctx": i, "group": S[i]["group"], "pos": j, "slot": r["group_pos"], "vllm": tg[j] if j < len(tg) else None,
                                     "fp32": f[j]["top_ids"][0], "fp32_gap": round(f[j]["gap"], 4), "class": q.margin_class(f[j]["gap"]),
                                     "exact_bf16_tie": bool(lp and lp[1] is not None and lp[1] == lp[0])})
                # bonus token (position g) vs fp32 argmax at g
                bonus_match = len(tg) > g and tg[g] == f[g]["top_ids"][0]
                bon[0] += bonus_match; bon[1] += 1
                if not bonus_match:
                    mism.append({"run": k, "ctx": i, "group": S[i]["group"], "pos": g, "slot": r["group_pos"], "kind": "bonus", "vllm": tg[g] if len(tg) > g else None,
                                 "fp32": f[g]["top_ids"][0], "fp32_gap": round(f[g]["gap"], 4), "class": q.margin_class(f[g]["gap"])})
                acc[0] += r["accepted"] == g; acc[1] += 1
                exp = q.expected_true(f, S[i]["proposal"], g)
                com[0] += r["committed"] == exp["committed"]; com[1] += 1
                if i in b1:
                    vb1[1] += 1
                    vb1[0] += tg[:g + 1] == b1[i]["target_greedy"][:g + 1]
                rows.append((r["group_pos"], ctx_ok_margin, 1.0))
            ws = q.slot_wrongness(rows)
            table[k] = {"B": B, "gamma": g, "order": it["order"], "n_requests": len(recs), "targets_vs_fp32_raw": frac(*raw),
                        "targets_vs_fp32_margin>=0.5": frac(*mar), "bonus_vs_fp32_raw": frac(*bon), "accepted==gamma": frac(*acc),
                        "committed==fp32_expected": frac(*com), "vs_B1_identical_input": frac(*vb1) if vb1[1] else None,
                        "by_length_group(raw ok/tot ; margin ok/tot)": {str(gi): [frac(v[0], v[1]), frac(v[2], v[3])] for gi, v in sorted(groups.items())},
                        "gate_margin_pass": q.passes(*mar) if g >= 1 and B >= 2 else None, "systematic_wrong_slots": q.systematic_slots(ws)}
        elif kind == "corrupt":
            a = [0, 0]; c = [0, 0]; am = [0, 0]; cm = [0, 0]
            for i, r in recs.items():
                f = S[i]["fp32"]; cp = r["corrupt_pos"]
                exp = q.expected_corrupt(f, S[i]["proposal"], cp)
                ok_a, ok_c = r["accepted"] == exp["accepted"], r["committed"] == exp["committed"]
                a[0] += ok_a; a[1] += 1; c[0] += ok_c; c[1] += 1
                if all(x >= q.GAP_OK for x in exp["gap_checked"]):
                    am[0] += ok_a; am[1] += 1; cm[0] += ok_c; cm[1] += 1
                if not (ok_a and ok_c):
                    mism.append({"run": k, "ctx": i, "kind": "corrupt", "corrupt_pos": cp, "accepted": r["accepted"], "expected_accepted": cp,
                                 "gaps": [round(x, 3) for x in exp["gap_checked"]], "class": q.margin_class(min(exp["gap_checked"]))})
            table[k] = {"B": B, "gamma": g, "order": it["order"], "n_requests": len(recs), "accepted_raw": frac(*a), "committed_raw": frac(*c),
                        "accepted_margin": frac(*am), "committed_margin": frac(*cm)}
        else:
            full = [0, 0]; first = [0, 0]; fm = [0, 0]
            for i, r in recs.items():
                f = S[i]["fp32"]
                chain = r["tokens"][:4] == S[i]["proposal"]
                full[0] += chain; full[1] += 1
                first[0] += r["tokens"][:1] == S[i]["proposal"][:1]; first[1] += 1
                if not chain:
                    j = next((t for t in range(4) if t >= len(r["tokens"]) or r["tokens"][t] != S[i]["proposal"][t]), 0)
                    mism.append({"run": k, "ctx": i, "kind": "decode", "pos": j, "fp32_gap": round(f[j]["gap"], 4), "class": q.margin_class(f[j]["gap"])})
                else:
                    pass
                if all(f[j]["gap"] >= q.GAP_OK for j in range(4)):
                    fm[1] += 1; fm[0] += chain
            tr = {int(i): r for i, r in runs.get("true_B16_g4_ident", {}).get("records", {}).items()}
            cons = None
            if k == "decode_B16_g4_ident" and tr:
                cons = frac(sum(recs[i]["tokens"][:4] == tr[i]["target_greedy"][:4] for i in recs if i in tr), len([i for i in recs if i in tr]))
            table[k] = {"B": B, "order": it["order"], "n_requests": len(recs), "four_token_chain_vs_fp32": frac(*full), "first_token_vs_fp32": frac(*first),
                        "four_token_chain_vs_fp32 (all 4 positions margin>=0.5)": frac(*fm), "decode_chain==prefill_target_greedy(B16 g4 ident)": cons}
    # permutation dependence: compare true runs of the same B, gamma across orders
    perm = {}
    for B in (2, 4, 8, 16, 32):
        ks = [k for k in runs if k.startswith(f"true_B{B}_g4_")]
        if len(ks) >= 2:
            base = {int(i): r["target_greedy"][:5] for i, r in runs[ks[0]]["records"].items()}
            diffs = {}
            for k2 in ks[1:]:
                n_diff = sum(base[int(i)] != r["target_greedy"][:5] for i, r in runs[k2]["records"].items() if int(i) in base)
                diffs[k2] = n_diff
            perm[f"B{B}"] = {"reference_run": ks[0], "contexts_whose_result_differs_from_reference": diffs}
    cls = Counter(m["class"] for m in mism)
    sched = {}
    for k, run in runs.items():
        if run["item"]["B"] in (16, 32) and run["item"]["kind"] == "true":
            sched[k] = [[c["n_seq"], [list(s) for s in c["steps"] if s[0] > 0]] for c in run["calls"]][:3]
    return {"table": table, "mismatch_classes": dict(cls), "mismatches": mism, "permutation_dependence": perm,
            "scheduling_examples(n_seq, [(prefill_groups, batched_tokens)...])": sched, "gate_pass_inprocess": block["pass"],
            "gate_inputs": block["gate"]}


for s in ("dev", "heldout"):
    out[s] = analyse_set(s)
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
for s in ("dev", "heldout"):
    if out[s]:
        print("==", s, "in-process gate", out[s]["gate_pass_inprocess"], "mismatch classes", out[s]["mismatch_classes"])
        for k, v in out[s]["table"].items():
            if k.startswith("true"):
                print("  ", k, v["targets_vs_fp32_margin>=0.5"], v["targets_vs_fp32_raw"], v["bonus_vs_fp32_raw"], v["accepted==gamma"], v["committed==fp32_expected"], v["vs_B1_identical_input"], v["systematic_wrong_slots"], v["gate_margin_pass"])
            elif k.startswith("corrupt"):
                print("  ", k, v["accepted_margin"], v["committed_margin"], v["accepted_raw"], v["committed_raw"])
            else:
                print("  ", k, v["four_token_chain_vs_fp32"], v["first_token_vs_fp32"], v["decode_chain==prefill_target_greedy(B16 g4 ident)"])
print(json.dumps(out["runtime_evidence_end"], default=str)[:900])
