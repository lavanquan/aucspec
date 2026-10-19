"""Pure helpers for the corrected-runtime qualification (CPU-testable): plans, permutations, expectations, metrics."""

from __future__ import annotations

import math
import random
from collections import Counter

GAP_OK = 0.5      # fp32 top-2 margin (nats) above which a disagreement is not a near tie
GATE = 0.99
PERM_SEEDS = {"perm1": 101, "perm2": 202, "perm3": 303}


def order(n: int, name: str) -> list[int]:
    idx = list(range(n))
    if name != "ident":
        random.Random(PERM_SEEDS[name]).shuffle(idx)
    return idx


def subset(n: int, k: int) -> list[int]:
    """k contexts spread evenly over 0..n-1 (covers every length group)."""
    step = n / k
    return [int(i * step) for i in range(k)]


def n_calls(plan: list[dict]) -> int:
    return sum(math.ceil(len(p["ctx"]) / p["B"]) for p in plan)


def build_plan(n_ctx: int, level: str) -> list[dict]:
    """Each item: kind in {true, corrupt, decode}, B, gamma, order, ctx = ordered context indices."""
    allc = list(range(n_ctx))
    sub = lambda k: subset(n_ctx, k)  # noqa: E731
    P = []

    def add(kind, B, g, ordname, ctx=None):
        ids = allc if ctx is None else ctx
        o = order(len(ids), ordname)
        P.append({"kind": kind, "B": B, "gamma": g, "order": ordname, "ctx": [ids[i] for i in o]})

    if level == "dev":
        add("true", 1, 4, "ident", sub(32))
        for o in ("ident", "perm1", "perm2"):
            add("true", 2, 4, o, sub(32))
        for B in (4, 8, 16, 32):
            for o in ("ident", "perm1", "perm2"):
                add("true", B, 4, o)
        for g in (0, 1, 2, 3):
            for o in ("ident", "perm1"):
                add("true", 16, g, o)
            add("true", 32, g, "ident")
        for g in (0, 2):
            add("true", 4, g, "ident")
        add("corrupt", 16, 4, "ident"); add("corrupt", 16, 4, "perm1"); add("corrupt", 16, 2, "ident")
        add("corrupt", 32, 4, "ident"); add("corrupt", 4, 4, "ident")
        add("decode", 1, 4, "ident", sub(16))
        add("decode", 4, 4, "ident"); add("decode", 8, 4, "ident")
        add("decode", 16, 4, "ident"); add("decode", 16, 4, "perm1"); add("decode", 32, 4, "ident")
    elif level == "heldout":
        add("true", 1, 4, "ident", sub(16))
        add("true", 4, 4, "ident"); add("true", 4, 4, "perm1")
        add("true", 8, 4, "ident")
        for o in ("ident", "perm1", "perm2"):
            add("true", 16, 4, o)
        add("true", 32, 4, "ident")
        add("true", 16, 2, "ident")
        add("corrupt", 16, 4, "ident"); add("corrupt", 4, 4, "ident")
        add("decode", 1, 4, "ident", sub(8)); add("decode", 4, 4, "ident")
        add("decode", 16, 4, "ident"); add("decode", 32, 4, "ident")
    else:
        raise ValueError(level)
    return P


def key(p: dict) -> str:
    return f"{p['kind']}_B{p['B']}_g{p['gamma']}_{p['order']}"


def expected_true(fp32: list[dict], proposal: list[int], gamma: int) -> dict:
    """Independent greedy target reference for a fully correct proposal of length gamma (fp32 has positions 0..4)."""
    arg = [fp32[j]["top_ids"][0] for j in range(gamma + 1)]
    return {"accepted": gamma, "committed": list(proposal[:gamma]) + [arg[gamma]], "bonus": arg[gamma],
            "gap_checked": [fp32[j]["gap"] for j in range(gamma + 1)]}


def corrupt_proposal(fp32: list[dict], proposal: list[int], gamma: int, ctx_index: int) -> tuple[list[int], int]:
    """Replace position c = ctx_index % gamma by fp32's second-best token. Returns (payload tokens, c)."""
    c = ctx_index % gamma
    toks = list(proposal[:gamma])
    toks[c] = fp32[c]["top_ids"][1]
    return toks, c


def expected_corrupt(fp32: list[dict], proposal: list[int], c: int) -> dict:
    return {"accepted": c, "committed": list(proposal[:c]) + [fp32[c]["top_ids"][0]], "correction": fp32[c]["top_ids"][0],
            "gap_checked": [fp32[j]["gap"] for j in range(c + 1)]}


def margin_class(gap: float) -> str:
    return "clear(>=0.5)" if gap >= GAP_OK else "near_tie(<0.5)"


def agreement(flags: list[tuple[bool, float]], gap_min: float = 0.0) -> tuple[int, int]:
    sel = [m for m, g in flags if g >= gap_min]
    return sum(sel), len(sel)


def slot_wrongness(rows: list[tuple[int, bool, float]]) -> dict[int, tuple[int, int]]:
    """rows: (group_pos, match, gap) -> {slot: (wrong at gap>=0.5, total at gap>=0.5)}"""
    out: dict[int, list[int]] = {}
    for pos, match, gap in rows:
        if gap >= GAP_OK:
            w, t = out.setdefault(pos, [0, 0])
            out[pos] = [w + (not match), t + 1]
    return {k: (v[0], v[1]) for k, v in sorted(out.items())}


def systematic_slots(wrong_by_slot: dict[int, tuple[int, int]], min_total: int = 4, frac: float = 0.5) -> list[int]:
    return [s for s, (w, t) in wrong_by_slot.items() if t >= min_total and w / t > frac]


def passes(ok: int, tot: int) -> bool:
    return tot > 0 and ok / tot >= GATE


# ----------------------------------------------------------------------------------------------------------------------
# Semantic validation of the actual `verify_batch_sync` outputs against the independent fp32 reference (fail-closed gate)
# ----------------------------------------------------------------------------------------------------------------------
SEMANTIC_FIELDS = ("accepted_length", "committed_tokens", "correction_token", "bonus_token", "reached_eos", "target_greedy_tokens")


def check_true(rec: dict, fp32: list, proposal: list, gamma: int) -> dict:
    """All proposals correct (fp32 greedy chain): every verifier output field must equal the independent expectation."""
    arg = [fp32[j]["top_ids"][0] for j in range(gamma + 1)]
    exp_committed = list(proposal[:gamma]) + [arg[gamma]]
    gaps = [fp32[j]["gap"] for j in range(gamma + 1)]
    return {
        "clear": min(gaps) >= GAP_OK,
        "accepted_length": rec["accepted"] == gamma,
        "committed_tokens": list(rec["committed"]) == exp_committed,
        "correction_token": rec["correction"] is None,
        "bonus_token": rec["bonus"] == arg[gamma],
        "reached_eos": rec["reached_eos"] is False,
        "target_greedy_tokens": list(rec["target_greedy"]) == arg,
    }


def check_corrupt(rec: dict, fp32: list, proposal: list, c: int) -> dict:
    """Proposal wrong at position c: rejection exactly at c, committing the prefix plus fp32 argmax as the correction."""
    arg = [fp32[j]["top_ids"][0] for j in range(c + 1)]
    gaps = [fp32[j]["gap"] for j in range(c + 1)]
    return {
        "clear": min(gaps) >= GAP_OK,
        "accepted_length": rec["accepted"] == c,
        "committed_tokens": list(rec["committed"]) == list(proposal[:c]) + [arg[c]],
        "correction_token": rec["correction"] == arg[c],
        "bonus_token": rec["bonus"] is None,
        "reached_eos": rec["reached_eos"] is False,
        "target_greedy_tokens": list(rec["target_greedy"]) == arg,
    }


def summarize_checks(checks: list) -> dict:
    """{field: {raw: [ok, n], clear: [ok, n]}}; clear keeps requests whose every checked fp32 margin is >= 0.5."""
    out = {}
    for f in SEMANTIC_FIELDS:
        cl = [c[f] for c in checks if c["clear"]]
        out[f] = {"raw": [sum(c[f] for c in checks), len(checks)], "clear": [sum(cl), len(cl)]}
    return out


def _recs(run):
    return {int(i): r for i, r in run["records"].items()}


def _min_gap(S, i, n):
    return min(S[i]["fp32"][j]["gap"] for j in range(n))


def evaluate_set(S: list, runs: dict, level: str) -> dict:
    """Fail-closed evaluation of one context set: {"pass", "failures", "checks"}.

    Every mandatory comparison needs a non-zero margin-filtered denominator and must meet the 99% gate separately
    (per run / per batch size); missing plan runs fail the set.
    """
    failures, checks = [], {}
    expected_keys = {key(p) for p in build_plan(len(S), level)}
    missing = sorted(expected_keys - set(runs))
    if missing:
        failures.append("missing runs: " + ", ".join(missing[:5]))

    def gate(name, ok, tot):
        checks[name] = [ok, tot]
        if tot == 0:
            failures.append(f"{name}: zero margin-clear observations")
        elif ok / tot < GATE:
            failures.append(f"{name}: {ok}/{tot} below {GATE}")

    b1 = _recs(runs["true_B1_g4_ident"]) if "true_B1_g4_ident" in runs else {}
    for k, run in runs.items():
        it, recs = run["item"], _recs(run)
        if it["kind"] in ("true", "corrupt"):
            cs = []
            for i, r in recs.items():
                if it["kind"] == "true":
                    cs.append(check_true(r, S[i]["fp32"], S[i]["proposal"], it["gamma"]))
                else:
                    cs.append(check_corrupt(r, S[i]["fp32"], S[i]["proposal"], r["corrupt_pos"]))
            sm = summarize_checks(cs)
            for f in SEMANTIC_FIELDS:
                gate(f"{k}:{f}", *sm[f]["clear"])
            checks[f"{k}:raw"] = {f: sm[f]["raw"] for f in SEMANTIC_FIELDS}
            if it["kind"] == "true" and it["B"] >= 2 and it["gamma"] == 4 and b1:
                same = [r["target_greedy"][:5] == b1[i]["target_greedy"][:5] for i, r in recs.items()
                        if i in b1 and _min_gap(S, i, 5) >= GAP_OK]
                gate(f"{k}:identical_to_B1_input", sum(same), len(same))
        else:
            clear = [i for i in recs if _min_gap(S, i, 4) >= GAP_OK]
            gate(f"{k}:four_token_chain_vs_fp32", sum(recs[i]["tokens"][:4] == S[i]["proposal"] for i in clear), len(clear))
            checks[f"{k}:raw_chain"] = [sum(r["tokens"][:4] == S[i]["proposal"] for i, r in recs.items()), len(recs)]
    dec, pre = runs.get("decode_B16_g4_ident"), runs.get("true_B16_g4_ident")
    if dec and pre:
        d, p = _recs(dec), _recs(pre)
        clear = [i for i in d if i in p and _min_gap(S, i, 5) >= GAP_OK]
        gate("decode_vs_prefill_B16_g4", sum(d[i]["tokens"][:4] == p[i]["target_greedy"][:4] for i in clear), len(clear))
    else:
        failures.append("decode_vs_prefill_B16_g4: runs missing")
    for B in (2, 4, 8, 16, 32):
        ks = sorted(k for k in runs if k.startswith(f"true_B{B}_g4_"))
        if len(ks) >= 2:
            base = _recs(runs[ks[0]])
            for k2 in ks[1:]:
                other = _recs(runs[k2])
                clear = [i for i in base if i in other and _min_gap(S, i, 5) >= GAP_OK]
                gate(f"permutation_identical_B{B}:{ks[0]}_vs_{k2}", sum(base[i]["target_greedy"] == other[i]["target_greedy"] for i in clear), len(clear))
    return {"pass": not failures, "failures": failures, "checks": checks}


def qualification_verdict(res, frozen: dict, max_calls: int = 400) -> dict:
    """Fail-closed verdict used by the orchestrator (independent of the in-process flags)."""
    reasons = []
    if not res:
        return {"ok": False, "reasons": ["qualification output missing"]}
    if res.get("status") != "ok":
        reasons.append("qualification status %r: %s" % (res.get("status"), str(res.get("error"))[:200]))
    sets = {}
    for s, level in (("dev", "dev"), ("heldout", "heldout")):
        blk = res.get(s)
        if not blk or not blk.get("runs"):
            reasons.append(f"{s} results missing or incomplete")
            continue
        ev = evaluate_set(frozen["sets"][s]["contexts"], blk["runs"], level)
        sets[s] = ev
        if not ev["pass"]:
            reasons.append(f"{s} gate failed: {ev['failures'][:3]}")
        if blk.get("pass") is not True:
            reasons.append(f"{s} in-process gate flag is not true")
    if res.get("calls", 10 ** 9) > max_calls:
        reasons.append(f"call budget exceeded: {res.get('calls')}")
    live = (res.get("runtime_evidence_end") or {}).get("live_engine") or {}
    if not (live.get("n_attention_layers") and live.get("layers_triton_flash_attn") == live.get("n_attention_layers")
            and live.get("layers_naive_sdpa") == 0 and live.get("triton_fa_calls", 0) > 0):
        reasons.append("runtime evidence incomplete: Triton FA in every layer, zero naive layers and a positive call count are required")
    return {"ok": not reasons, "reasons": reasons, "sets": {k: {"pass": v["pass"], "failures": v["failures"]} for k, v in sets.items()}}
