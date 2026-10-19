"""One vLLM configuration (fresh process): measure batched vs single-sequence vs fp32 correctness.

Diagnostic engine (max_logprobs=2). Frozen token ids come from rocm_fi_prepare.py. Every llm.generate
call counts against --call-budget. Environment variables (attention backend etc.) are set by the caller.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "legacy" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rocm_fi_lib as lib  # noqa: E402

SHUF_SEED = 20261019
B1_PREFILL_N, B1_DECODE_N, B4_DECODE_N = 12, 4, 16


def env_snapshot() -> dict:
    keys = [k for k in os.environ if k.startswith(("VLLM_", "HIP_", "ROCM_", "HSA_", "PYTORCH_", "CUDA_", "TORCH_", "NCCL_", "RCCL_"))]
    return {k: os.environ[k] for k in sorted(keys)}


def describe_attention(llm) -> dict:
    info = {"engine_class": f"{type(llm.llm_engine).__module__}.{type(llm.llm_engine).__name__}"}
    try:
        ex = llm.llm_engine.model_executor
        runner = ex.driver_worker.model_runner
        info["attn_backend_class"] = type(runner.attn_backend).__name__ if hasattr(runner, "attn_backend") else None
        model = runner.model
        impls = []
        for m in model.modules():
            impl = getattr(m, "impl", None)
            if impl is not None and hasattr(impl, "use_naive_attn"):
                impls.append({"impl": type(impl).__name__, "use_naive_attn": bool(impl.use_naive_attn),
                              "use_triton_flash_attn": bool(getattr(impl, "use_triton_flash_attn", False)),
                              "sliding_window": str(getattr(impl, "sliding_window", None))})
        info["attention_impls_first"] = impls[:1]
        info["attention_impl_count"] = len(impls)
        cfgs = llm.llm_engine.scheduler_config
        info["chunked_prefill"] = bool(cfgs.chunked_prefill_enabled)
        info["max_num_batched_tokens"] = int(cfgs.max_num_batched_tokens)
        info["max_num_seqs"] = int(cfgs.max_num_seqs)
        info["dtype"] = str(llm.llm_engine.model_config.dtype)
        info["enforce_eager"] = bool(llm.llm_engine.model_config.enforce_eager)
        info["prefix_caching"] = bool(llm.llm_engine.cache_config.enable_prefix_caching)
    except Exception as exc:  # noqa: BLE001
        info["introspection_error"] = repr(exc)
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--call-budget", type=int, required=True)
    a = ap.parse_args()

    result = {"name": a.name, "status": "started", "env": env_snapshot(), "dtype_requested": a.dtype,
              "eager_requested": a.eager, "calls": 0, "latency": {}}
    out = Path(a.out)
    try:
        import torch
        import yaml
        import canonical_guard as guard
        import vllm
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from edge_specsim.target_client import VLLMCandidateVerifier
        from edge_specsim.provenance import collect_provenance

        cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
        guard.assert_clean_source()
        frozen = json.load(open(a.frozen))
        m, sim = cfg["models"], cfg["simulation"]
        result["versions"] = {"vllm": vllm.__version__, "torch": torch.__version__, "hip": getattr(torch.version, "hip", None)}
        result["provenance"] = collect_provenance(REPO_ROOT, cfg)
        t0 = time.time()
        ver = VLLMCandidateVerifier(
            model_name=m["target"], gpu_memory_utilization=float(m.get("target_gpu_memory_utilization", 0.7)),
            max_model_len=int(m.get("target_max_model_len", 4096)), dtype=a.dtype, enforce_eager=a.eager,
            enable_prefix_caching=False, sampling_temperature=0.0, seed=int(sim["seed"]), max_logprobs=2)
        llm = ver.llm
        result["init_s"] = time.time() - t0
        result["engine"] = describe_attention(llm)

        lat: dict[str, list[float]] = {}

        def gen(prompts, sp, kind):
            if result["calls"] >= a.call_budget:
                raise RuntimeError("call budget exhausted")
            result["calls"] += 1
            t = time.perf_counter()
            r = llm.generate(prompts, sp, use_tqdm=False)
            lat.setdefault(kind, []).append((time.perf_counter() - t) * 1000.0)
            return r

        sp_pre = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, prompt_logprobs=2, logprobs=2,
                                detokenize=False, skip_special_tokens=False)
        sp_dec = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4, logprobs=2, detokenize=False)

        def ser(d):
            s = sorted(((float(v.logprob), int(k)) for k, v in d.items()), reverse=True)
            return [[k, lp] for lp, k in s[:3]]

        def prefill_run(S, idxs, B, tag):
            """idxs: context indices in batch order (slot = position in that order). Returns {ctx: record}."""
            res = {}
            for s in range(0, len(idxs), B):
                grp = idxs[s: s + B]
                ps = [TokensPrompt(prompt_token_ids=S[i]["context"] + S[i]["proposal"]) for i in grp]
                outs = gen(ps, sp_pre, f"prefill_B{B}")
                for slot, (i, o) in enumerate(zip(grp, outs)):
                    n = len(S[i]["context"])
                    ent = [ser(o.prompt_logprobs[n + j]) for j in range(4)]
                    res[i] = {"slot": s + slot, "group_pos": slot, "entries": ent,
                              "argmax": [e[0][0] for e in ent],
                              "proposal_lp": [float(o.prompt_logprobs[n + j][S[i]["proposal"][j]].logprob) for j in range(4)],
                              "sampler_next": int(o.outputs[0].token_ids[0])}
            return res

        def decode_run(S, idxs, B):
            res = {}
            for s in range(0, len(idxs), B):
                grp = idxs[s: s + B]
                outs = gen([TokensPrompt(prompt_token_ids=list(S[i]["context"])) for i in grp], sp_dec, f"decode_B{B}")
                for slot, (i, o) in enumerate(zip(grp, outs)):
                    res[i] = {"slot": s + slot, "tokens": [int(t) for t in o.outputs[0].token_ids],
                              "top": [ser(d) for d in o.outputs[0].logprobs]}
            return res

        def correct_flags(S, run, gap=lib.GAP_OK):
            """per-context: all positions with fp32 gap >= gap agree with fp32 argmax."""
            fl = {}
            for i, r in run.items():
                fl[i] = all(r["argmax"][j] == S[i]["fp32"][j]["top_ids"][0] for j in range(4) if S[i]["fp32"][j]["gap"] >= gap)
            return fl

        def rate(S, run, gap):
            rows = [{"match": r["argmax"][j] == S[i]["fp32"][j]["top_ids"][0], "fp32_gap": S[i]["fp32"][j]["gap"]}
                    for i, r in run.items() for j in range(4)]
            return lib.agreement(rows, gap)

        def measure(S, with_shuffle=True, heavy=True):
            n = len(S)
            ident = list(range(n))
            perm = lib.make_perm(n, SHUF_SEED)
            M = {"prefill": {}, "decode": {}}
            M["prefill"]["B1"] = prefill_run(S, ident[:B1_PREFILL_N], 1, "B1")
            M["prefill"]["B4_ident"] = prefill_run(S, ident, 4, "B4")
            M["prefill"]["B16_ident"] = prefill_run(S, ident, 16, "B16")
            M["prefill"]["B16_perm"] = prefill_run(S, perm, 16, "B16")
            if with_shuffle:
                M["prefill"]["B4_perm"] = prefill_run(S, perm, 4, "B4")
            M["decode"]["B16_ident"] = decode_run(S, ident, 16)
            if heavy:
                M["decode"]["B16_perm"] = decode_run(S, perm, 16)
            M["decode"]["B4_ident"] = decode_run(S, ident[:B4_DECODE_N], 4)
            M["decode"]["B1"] = decode_run(S, ident[:B1_DECODE_N], 1)
            M["perm"] = perm
            return M

        def dev_gate(S, M):
            r = {}
            for B in ("B4_ident", "B16_ident", "B16_perm"):
                r[f"{B}_vs_fp32_gap>=0.5"] = rate(S, M["prefill"][B], lib.GAP_OK)["rate"]
            # vs single-sequence vLLM on the contexts that have a B=1 run
            b1 = M["prefill"]["B1"]
            for B in ("B4_ident", "B16_ident"):
                eq = [M["prefill"][B][i]["argmax"][j] == b1[i]["argmax"][j] for i in b1 for j in range(4)]
                r[f"{B}_vs_B1_all_positions"] = sum(eq) / len(eq)
            return r

        # warm-up (untimed in analysis; counted)
        gen([TokensPrompt(prompt_token_ids=frozen["sets"]["dev"]["contexts"][0]["context"])], sp_dec, "warmup")
        gen([TokensPrompt(prompt_token_ids=c["context"]) for c in frozen["sets"]["dev"]["contexts"][:16]], sp_dec, "warmup")

        dev = frozen["sets"]["dev"]["contexts"]
        M = measure(dev)
        result["dev"] = {"measurements": M, "gate_inputs": dev_gate(dev, M)}
        result["dev_gate_pass"] = lib.passes_gate(result["dev"]["gate_inputs"])
        if result["dev_gate_pass"] and a.call_budget - result["calls"] >= 34:
            H = frozen["sets"]["heldout"]["contexts"]
            MH = measure(H, with_shuffle=False, heavy=False)
            result["heldout"] = {"measurements": MH, "gate_inputs": dev_gate(H, MH)}
            result["heldout_gate_pass"] = lib.passes_gate(
                {k: v for k, v in result["heldout"]["gate_inputs"].items() if "B16_perm" not in k})
        result["latency_ms"] = {k: {"n": len(v), "mean": sum(v) / len(v), "max": max(v)} for k, v in lat.items()}
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - record failures, never raise
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()[-3000:]
    out.write_text(json.dumps(result))
    print(json.dumps({"name": a.name, "status": result["status"], "calls": result["calls"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
