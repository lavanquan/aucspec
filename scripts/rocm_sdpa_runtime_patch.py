"""DIAGNOSTIC CANDIDATE ONLY (candidate G; not part of the canonical verifier, nothing is edited inside vLLM).

vLLM 0.9.1's ROCm naive-SDPA prefill (`rocm_flash_attn._sdpa_attention`) is handed the CUMULATIVE
`query_seq_start_loc = [0, l0, l0+l1, ...]` where it expects per-sequence lengths. `apply()` replaces the module-level
function by a wrapper that (1) validates the cumulative representation, (2) converts it to lengths, (3) records
invocation statistics. It must run BEFORE the engine is created: the attention layers bind the module global in
`ROCmFlashAttentionImpl.__init__` (`self.sdpa_attn_func = _sdpa_attention`).

Validation (strict once `STATE.strict` is switched on after engine construction; during engine construction, e.g. the
memory-profiling dummy run, violations are only recorded and the call is passed through unchanged):
  first offset == 0, offsets non-decreasing, last offset == number of prefill tokens, every length > 0.
A violation in strict mode raises PatchError (no silent fallback).
"""

from __future__ import annotations

import os
from collections import Counter


class PatchError(RuntimeError):
    pass


class _State:
    def __init__(self):
        self.strict = False
        self.calls = 0
        self.converted = 0
        self.passthrough_unvalidated = 0
        self.batch_sizes = Counter()
        self.tokens = 0
        self.pids = set()
        self.violations = []
        self.first_calls = []
        self.installed = False

    def summary(self):
        return {"installed": self.installed, "calls": self.calls, "converted": self.converted,
                "passthrough_unvalidated": self.passthrough_unvalidated, "batch_size_counts": dict(self.batch_sizes),
                "total_tokens": self.tokens, "pids_that_executed_the_patch": sorted(self.pids), "pid_now": os.getpid(),
                "violations": self.violations[:10], "n_violations": len(self.violations), "first_calls": self.first_calls[:5]}


STATE = _State()


def validate_cumulative(offsets, num_tokens):
    """Return (lengths, None) if `offsets` is a valid cumulative start-loc for `num_tokens` tokens, else (None, reason)."""
    vals = [int(x) for x in offsets]
    if len(vals) < 2:
        return None, "fewer than two offsets"
    if vals[0] != 0:
        return None, f"first offset is {vals[0]}, not 0"
    if any(b < a for a, b in zip(vals, vals[1:])):
        return None, "offsets are not monotone"
    if vals[-1] != int(num_tokens):
        return None, f"last offset {vals[-1]} != number of prefill tokens {int(num_tokens)}"
    lengths = [b - a for a, b in zip(vals, vals[1:])]
    if any(n <= 0 for n in lengths):
        return None, "non-positive sequence length"
    return lengths, None


def cumulative_to_lengths(seq_lens):
    """Permissive helper kept for the unit tests: cumulative -> lengths, anything else unchanged."""
    vals = [int(x) for x in seq_lens]
    if len(vals) >= 2 and vals[0] == 0 and all(b >= a for a, b in zip(vals, vals[1:])):
        return [b - a for a, b in zip(vals, vals[1:])]
    return vals


def apply():
    """Install the wrapper. Returns the shared STATE. Idempotent."""
    import vllm.attention.backends.rocm_flash_attn as m

    original = m._sdpa_attention
    if getattr(original, "_aucspec_patched", False):
        return STATE

    def fixed(query, key, value, output, seq_lens, num_tokens, *args, **kw):
        offs = seq_lens.tolist() if hasattr(seq_lens, "tolist") else list(seq_lens)
        lengths, why = validate_cumulative(offs, num_tokens)
        STATE.calls += 1
        STATE.pids.add(os.getpid())
        if lengths is None:
            STATE.violations.append({"reason": why, "offsets_head": offs[:6], "num_tokens": int(num_tokens)})
            if STATE.strict:
                raise PatchError(f"unexpected seq_lens representation: {why}")
            STATE.passthrough_unvalidated += 1
            return original(query, key, value, output, seq_lens, num_tokens, *args, **kw)
        STATE.converted += 1
        STATE.batch_sizes[len(lengths)] += 1
        STATE.tokens += int(num_tokens)
        if len(STATE.first_calls) < 5:
            STATE.first_calls.append({"n_seq": len(lengths), "lengths_head": lengths[:4], "num_tokens": int(num_tokens)})
        return original(query, key, value, output, lengths, num_tokens, *args, **kw)

    fixed._aucspec_patched = True
    fixed._aucspec_original = original
    m._sdpa_attention = fixed
    STATE.installed = True
    return STATE
