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
