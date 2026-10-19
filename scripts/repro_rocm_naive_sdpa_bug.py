"""Minimal, CPU-only reproduction of the vLLM 0.9.1 ROCm naive-SDPA prefill defect (needs only torch).

vllm/attention/backends/rocm_flash_attn.py calls
    self.sdpa_attn_func(query, key, value, output[:num_prefill_tokens], query_seq_start_loc, ...)
i.e. it passes the CUMULATIVE offsets [0, l0, l0+l1, ...] where `_sdpa_attention` expects per-sequence lengths
(`for i, seq_len in enumerate(seq_lens): end = start + seq_len`). For three or more sequences the
slices overlap several sequences, so later sequences attend to earlier sequences' tokens.

The function below is a verbatim copy of vLLM's `_sdpa_attention` (checked against the installed source when
--vllm-source is given). Output: per-sequence max error against a per-sequence causal reference.
"""

import argparse
import ast
import sys
from pathlib import Path

import torch


def _sdpa_attention(query, key, value, output, seq_lens, num_tokens, num_heads, head_size, scale, attn_masks=None):
    start = 0
    assert output.shape == (num_tokens, num_heads, head_size)
    assert output.dtype == query.dtype
    assert output.device == query.device

    for i, seq_len in enumerate(seq_lens):
        end = start + seq_len
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            sub_out = torch.nn.functional.scaled_dot_product_attention(
                query[:, start:end, :],
                key[:, start:end, :],
                value[:, start:end, :],
                dropout_p=0.0,
                is_causal=attn_masks is None,
                attn_mask=attn_masks[i] if attn_masks else None,
                scale=scale).movedim(query.dim() - 2, 0)
            output[start:end, :, :] = sub_out
            start = end

    return output


def reference(q, k, v, lens, scale):
    outs = []
    s = 0
    for n in lens:
        o = torch.nn.functional.scaled_dot_product_attention(
            q[s:s + n].transpose(0, 1), k[s:s + n].transpose(0, 1), v[s:s + n].transpose(0, 1),
            is_causal=True, scale=scale).transpose(0, 1)
        outs.append(o)
        s += n
    return torch.cat(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-source", default=None, help="path to vllm/attention/backends/rocm_flash_attn.py")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 3, 4, 16])
    a = ap.parse_args()
    if a.vllm_source and Path(a.vllm_source).exists():
        src = Path(a.vllm_source).read_text()
        tree = ast.parse(src)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_sdpa_attention")
        mine = next(n for n in ast.parse(Path(__file__).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_sdpa_attention")
        strip = lambda n: ast.dump(ast.Module(body=[n], type_ignores=[]), annotate_fields=False).replace("Optional", "")
        print("function body identical to installed vLLM source:",
              [ast.dump(x) for x in fn.body] == [ast.dump(x) for x in mine.body])
    torch.manual_seed(0)
    H, D, L = 4, 32, 40
    scale = D ** -0.5
    print(f"{'B':>3} {'per-sequence max |error| (slots)':<}")
    bad_slots = {}
    for B in a.batch:
        lens = [L] * B
        T = sum(lens)
        q, k, v = (torch.randn(T, H, D) for _ in range(3))
        ref = reference(q, k, v, lens, scale)
        cum = [0]
        for n in lens:
            cum.append(cum[-1] + n)
        out = torch.zeros(T, H, D)
        # exactly what vLLM passes: query/key/value moved to [heads, tokens, dim] and the CUMULATIVE start locations
        _sdpa_attention(q.movedim(0, q.dim() - 2), k.movedim(0, k.dim() - 2), v.movedim(0, v.dim() - 2), out, cum, T, H, D, scale)
        errs = [float((out[sum(lens[:i]):sum(lens[:i + 1])] - ref[sum(lens[:i]):sum(lens[:i + 1])]).abs().max()) for i in range(B)]
        bad_slots[B] = [i for i, e in enumerate(errs) if e > 1e-4]
        print(f"{B:>3} wrong slots {bad_slots[B]}   max errors {[round(e, 3) for e in errs]}")
        # fixed call: per-sequence lengths
        out2 = torch.zeros(T, H, D)
        _sdpa_attention(q.movedim(0, q.dim() - 2), k.movedim(0, k.dim() - 2), v.movedim(0, v.dim() - 2), out2, lens, T, H, D, scale)
        assert float((out2 - ref).abs().max()) < 1e-4, "per-sequence lengths reproduce the reference"
    print("with per-sequence lengths as argument: all slots match the reference")
    return 0 if bad_slots.get(1) == [] and bad_slots.get(2) == [] else 1


if __name__ == "__main__":
    sys.exit(main())
