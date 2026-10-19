"""DIAGNOSTIC CANDIDATE ONLY (not applied anywhere, not part of the canonical verifier).

Runtime (in-process, no reinstall) correction for vLLM 0.9.1's ROCm naive-SDPA prefill: the backend hands
`_sdpa_attention` the cumulative query_seq_start_loc where per-sequence lengths are expected. `apply()` wraps the
module-level function so that a cumulative vector is converted to lengths. It must be validated on GPU against the
correctness gates (B=4/16 >= 99% vs B=1 and fp32, permutations, held-out) before any use. apply() must run BEFORE the engine is created (the attention layers bind the function at init).
"""

from __future__ import annotations


def cumulative_to_lengths(seq_lens):
    """[0, l0, l0+l1, ...] -> [l0, l1, ...]; anything else is returned unchanged (already lengths)."""
    vals = [int(x) for x in seq_lens]
    if len(vals) >= 2 and vals[0] == 0 and all(b >= a for a, b in zip(vals, vals[1:])):
        return [b - a for a, b in zip(vals, vals[1:])]
    return vals


def apply():
    import vllm.attention.backends.rocm_flash_attn as m

    original = m._sdpa_attention
    if getattr(original, "_aucspec_patched", False):
        return False

    def fixed(query, key, value, output, seq_lens, *args, **kw):
        return original(query, key, value, output, cumulative_to_lengths(seq_lens.tolist() if hasattr(seq_lens, "tolist") else seq_lens),
                        *args, **kw)

    fixed._aucspec_patched = True
    m._sdpa_attention = fixed
    return True
