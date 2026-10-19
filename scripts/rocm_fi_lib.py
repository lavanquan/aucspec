"""Pure-Python helpers for the ROCm batched-inference fault isolation (no torch/vLLM; CPU-testable).

* naive_sdpa_status: model of what vLLM 0.9.1's ROCm `_sdpa_attention` does when it is handed the
  cumulative `query_seq_start_loc` (offsets) where it expects per-sequence lengths.
* agreement / permutation / gate helpers shared by the GPU runner and the analysis.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter

GAP_OK = 0.5  # nats: fp32 top-2 gap above which a bf16/fp16 argmax disagreement is not a near tie
GATE = 0.99


def token_hash(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]


def make_perm(n: int, seed: int) -> list[int]:
    p = list(range(n))
    random.Random(seed).shuffle(p)
    return p


def naive_sdpa_status(seq_lens: list[int]) -> list[str]:
    """Replicates the loop in `_sdpa_attention` when `seq_lens` is the cumulative start-loc vector
    [0, l0, l0+l1, ...]:   start = 0; for x in loc: end = start + x; attend(tokens[start:end]); start = end.
    Slicing clips at the total token count. Returns per sequence:
      'correct'      segment starts at the sequence start and covers it fully (it is the FIRST sequence of its
                     segment, so causal attention sees only its own tokens);
      'contaminated' it lies inside a segment behind another sequence (attends to foreign tokens);
      'broken'       a segment boundary cuts it, or some of its tokens are covered by no segment.
    """
    loc = [0]
    for x in seq_lens:
        loc.append(loc[-1] + x)
    total = loc[-1]
    segs, start = [], 0
    for x in loc:
        end = start + x
        s, e = min(start, total), min(end, total)
        if e > s:
            segs.append((s, e))
        start = end
    out = []
    for k in range(len(seq_lens)):
        a, b = loc[k], loc[k + 1]
        covering = [(s, e) for (s, e) in segs if s <= a and e >= b]
        if not covering:
            out.append("broken")
        elif any(s == a for (s, e) in covering):
            out.append("correct")
        else:
            out.append("contaminated")
    return out


def argmax_ids(entry: dict) -> list[int]:
    m = max(v for v in entry.values())
    return sorted(int(k) for k, v in entry.items() if v == m)


def agreement(rows: list[dict], gap_min: float = 0.0) -> dict:
    """rows: dicts with 'match' (bool) and 'fp32_gap' (float). Returns rate over rows with gap >= gap_min."""
    sel = [r for r in rows if r["fp32_gap"] >= gap_min]
    n = len(sel)
    return {"n": n, "rate": (sum(r["match"] for r in sel) / n) if n else None}


def slot_consistency(correct_a: dict[int, bool], correct_b: dict[int, bool], slot_a: dict[int, int], slot_b: dict[int, int]) -> dict:
    """Two batch orders of the same contexts: is wrongness tied to the INPUT or to the SLOT?"""
    c = Counter()
    for i in correct_a:
        c[(correct_a[i], correct_b[i])] += 1
    both_ok, both_bad = c[(True, True)], c[(False, False)]
    only_a, only_b = c[(False, True)], c[(True, False)]
    n = len(correct_a)
    same_slot = sum(1 for i in correct_a if slot_a[i] == slot_b[i])
    return {"n": n, "both_correct": both_ok, "both_wrong": both_bad, "wrong_only_in_order_a": only_a,
            "wrong_only_in_order_b": only_b, "input_specific_fraction": both_bad / max(1, both_bad + only_a + only_b),
            "contexts_whose_slot_did_not_change": same_slot}


def passes_gate(rates: dict[str, float | None]) -> bool:
    return all(v is not None and v >= GATE for v in rates.values())
