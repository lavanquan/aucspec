"""Tables for the candidate G/H validation (CPU only). Every percentage is reported as ok/total."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rocm_fi_lib as lib  # noqa: E402

D = Path(sys.argv[1])
frozen = json.load(open(sys.argv[2]))
out = {"candidates": {}}


def prefill_table(S, M):
    b1 = M.get("prefill_B1_g4_ident", {})
    rows, mismatches = {}, []
    for k, run in sorted(M.items()):
        if not k.startswith("prefill"):
            continue
        g = int(k.split("_g")[1].split("_")[0])
        ok_f = tot_f = ok_all = tot_all = 0
        wrong_slots = {}
        for i, r in run.items():
            i = int(i)
            for j in range(g):
                f = S[i]["fp32"][j]
                match = r["argmax"][j] == f["top_ids"][0]
                ok_all += match
                tot_all += 1
                if f["gap"] >= lib.GAP_OK:
                    tot_f += 1
                    ok_f += match
                    if not match:
                        wrong_slots[r["group_pos"]] = wrong_slots.get(r["group_pos"], 0) + 1
                if not match:
                    top = r["top"][j]
                    mismatches.append({"run": k, "ctx": i, "pos": j, "slot": r["group_pos"], "vllm_argmax": r["argmax"][j],
                                       "fp32_argmax": f["top_ids"][0], "fp32_gap": round(f["gap"], 4),
                                       "vllm_top2_gap": round(top[0][1] - top[1][1], 4) if len(top) > 1 else None,
                                       "B1_argmax": (b1.get(i) or b1.get(str(i)) or {}).get("argmax", [None] * 4)[j] if (i in b1 or str(i) in b1) else None})
        ok_b1 = tot_b1 = 0
        if not k.startswith("prefill_B1_"):
            for i, r in run.items():
                ref = b1.get(i) or b1.get(str(i))
                if ref:
                    for j in range(g):
                        tot_b1 += 1
                        ok_b1 += r["argmax"][j] == ref["argmax"][j]
        rows[k] = {"vs_fp32_gap>=0.5": f"{ok_f}/{tot_f}", "rate_fp32": (ok_f / tot_f) if tot_f else None,
                   "vs_fp32_all_positions": f"{ok_all}/{tot_all}", "vs_B1": f"{ok_b1}/{tot_b1}" if tot_b1 else None,
                   "rate_B1": (ok_b1 / tot_b1) if tot_b1 else None, "wrong_slots(gap>=0.5)": wrong_slots,
                   "passes_99": bool(tot_f and ok_f / tot_f >= lib.GATE and (not tot_b1 or ok_b1 / tot_b1 >= lib.GATE))}
    return rows, mismatches


def decode_table(S, M, prefill_ref):
    rows = {}
    for k, run in sorted(M.items()):
        if not k.startswith("decode"):
            continue
        first = full = 0
        fails = []
        for i, r in run.items():
            i = int(i)
            prop = S[i]["proposal"]
            first += r["tokens"][0] == prop[0]
            same = r["tokens"][:4] == prop
            full += same
            if not same:
                j = next((t for t in range(4) if t >= len(r["tokens"]) or r["tokens"][t] != prop[t]), 0)
                fails.append({"ctx": i, "first_diff_pos": j, "fp32_gap_at_pos": round(S[i]["fp32"][j]["gap"], 4),
                              "vllm_token": r["tokens"][j] if j < len(r["tokens"]) else None, "fp32_token": prop[j]})
        n = len(run)
        rows[k] = {"first_token_eq_fp32": f"{first}/{n}", "four_tokens_eq_fp32": f"{full}/{n}",
                   "failures_with_fp32_gap>=0.5": sum(f["fp32_gap_at_pos"] >= lib.GAP_OK for f in fails),
                   "failures_near_tie(<0.5)": sum(f["fp32_gap_at_pos"] < lib.GAP_OK for f in fails), "failure_details": fails[:12]}
        pref = prefill_ref
        if pref and k == "decode_B16_g4_ident":
            agree = tot = 0
            for i, r in run.items():
                p = pref.get(i) or pref.get(str(i))
                if p:
                    tot += 1
                    # decode continuation vs prefill argmax chain on the same context (4 positions)
                    agree += r["tokens"][:4] == p["argmax"][:4]
            rows[k]["decode_chain_eq_prefill_argmax_chain(B16 g4 ident)"] = f"{agree}/{tot}"
    return rows


for name in ("G_runtime_patch", "H_triton331"):
    f = D / f"result_{name}.json"
    if not f.exists():
        out["candidates"][name] = {"status": "no_result"}
        continue
    r = json.load(open(f))
    c = {k: r.get(k) for k in ("status", "error", "mode", "pid", "calls", "init_s", "versions", "env", "engine", "patch_stats",
                               "triton_fa_stats", "latency_ms", "heldout_error")}
    c["traceback_tail"] = (r.get("traceback") or "")[-700:]
    for setname in ("dev", "heldout"):
        if setname not in r:
            continue
        S = frozen["sets"][setname]["contexts"]
        M = r[setname]["measurements"]
        pt, mm = prefill_table(S, M)
        ref = M.get("prefill_B16_g4_ident")
        c[setname] = {"prefill": pt, "decode": decode_table(S, M, ref), "mismatches": mm, "gate_pass": r[setname]["pass"]}
        # slot profile B16 g4
        slots = {}
        for ordname in ("ident", "perm"):
            key = f"prefill_B16_g4_{ordname}"
            if key in M:
                slots[ordname] = pt[key]["wrong_slots(gap>=0.5)"]
        c[setname]["B16_g4_wrong_slots_by_order"] = slots
    out["candidates"][name] = c
Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
for name, c in out["candidates"].items():
    print("==", name, c.get("status"), (c.get("error") or "")[:300])
    for setname in ("dev", "heldout"):
        if setname in c:
            print(" ", setname, "gate", c[setname]["gate_pass"])
            for k, v in c[setname]["prefill"].items():
                print("   ", k, v["vs_fp32_gap>=0.5"], v["vs_B1"], v["wrong_slots(gap>=0.5)"], v["passes_99"])
            for k, v in c[setname]["decode"].items():
                print("   ", k, v["four_tokens_eq_fp32"], v["first_token_eq_fp32"], v["failures_with_fp32_gap>=0.5"], v["failures_near_tie(<0.5)"])
    if c.get("patch_stats"):
        print("  patch", json.dumps(c["patch_stats"])[:600])
    if c.get("engine"):
        print("  engine", json.dumps(c["engine"])[:500])
