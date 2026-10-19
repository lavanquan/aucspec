"""CPU-only tests for the qualification helpers (plans, permutations, independent expectations, margin metrics)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import rocm_qual_lib as q  # noqa: E402


def test_plans_fit_the_budget_and_cover_the_required_matrix():
    dev, ho = q.build_plan(64, "dev"), q.build_plan(32, "heldout")
    nd, nh = q.n_calls(dev), q.n_calls(ho)
    assert nd + 2 + nh + 36 <= 500 and nd + 2 + nh <= 400, (nd, nh)
    true = [p for p in dev if p["kind"] == "true"]
    assert {p["B"] for p in true} == {1, 2, 4, 8, 16, 32}
    assert {p["gamma"] for p in true} == {0, 1, 2, 3, 4}
    for B in (2, 4, 8, 16, 32):                       # at least three distinct batch orders at gamma=4
        orders = {tuple(p["ctx"]) for p in true if p["B"] == B and p["gamma"] == 4}
        assert len(orders) >= 3, B
    assert any(p["kind"] == "corrupt" for p in dev) and any(p["kind"] == "decode" for p in dev)
    assert len({q.key(p) for p in dev}) == len(dev)   # no duplicate run keys


def test_orders_are_permutations_and_deterministic():
    for name in ("ident", "perm1", "perm2", "perm3"):
        o = q.order(32, name)
        assert sorted(o) == list(range(32)) and o == q.order(32, name)
    assert q.order(32, "perm1") != q.order(32, "perm2") != list(range(32))
    s = q.subset(64, 32)
    assert len(set(s)) == 32 and max(s) < 64 and {i // 16 for i in s} == {0, 1, 2, 3}   # covers every length group


FP = [{"top_ids": [10 + j, 99], "top_logprobs": [-0.1, -1.0], "gap": 0.9 if j != 2 else 0.2} for j in range(5)]
PROP = [10, 11, 12, 13]


def test_expected_true_and_corrupt_follow_the_fp32_reference():
    e = q.expected_true(FP, PROP, 3)
    assert e["accepted"] == 3 and e["committed"] == [10, 11, 12, 13] and e["bonus"] == 13
    toks, c = q.corrupt_proposal(FP, PROP, 4, 6)       # ctx index 6 -> position 2
    assert c == 2 and toks == [10, 11, 99, 13]
    x = q.expected_corrupt(FP, PROP, c)
    assert x["accepted"] == 2 and x["committed"] == [10, 11, 12] and x["correction"] == 12


def test_margin_metrics_report_raw_and_filtered_denominators():
    flags = [(True, 2.0), (False, 0.1), (True, 0.7), (False, 1.2)]
    assert q.agreement(flags) == (2, 4)
    assert q.agreement(flags, q.GAP_OK) == (2, 3)
    assert q.margin_class(0.49).startswith("near") and q.margin_class(0.5).startswith("clear")
    assert q.passes(99, 100) and not q.passes(98, 100) and not q.passes(0, 0)


def test_slot_analysis_flags_systematic_slots_only():
    rows = [(s, s not in (2, 5), 1.0) for s in range(8) for _ in range(6)]          # slots 2 and 5 always wrong
    w = q.slot_wrongness(rows)
    assert q.systematic_slots(w) == [2, 5]
    rows_tie = [(0, False, 0.1)] * 10 + [(0, True, 2.0)] * 10                         # near ties are not counted
    assert q.systematic_slots(q.slot_wrongness(rows_tie)) == []
