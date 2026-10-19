"""Validate one candidate runtime (fresh process): G = naive-SDPA runtime fix, H = Triton FA with torch-compatible Triton.

Uses the frozen B4 contexts and fp32 oracle (no HF calls here). Every llm.generate call counts against --call-budget.
Records, for every prefill run, per-position argmax / proposal logprob for B in {1,4,16}, gamma in {1,2,4},
identity and shuffled order, plus batched decode, and the activation evidence of the candidate.
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
GAMMAS = (4, 2, 1)


def impl_list(llm):
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    out = []
    for m in runner.model.modules():
        impl = getattr(m, "impl", None)
        if impl is not None and hasattr(impl, "use_naive_attn"):
            out.append(impl)
    return out


def plan(level: str, n: int):
    """list of (kind, B, gamma, order, n_ctx)."""
    p = []
    if level == "dev":
        p += [("prefill", 1, 4, "ident", 12)]
        for g in GAMMAS:
            p += [("prefill", 4, g, "ident", n)]
        p += [("prefill", 4, 4, "perm", n), ("prefill", 4, 2, "perm", n)]
        for g in GAMMAS:
            p += [("prefill", 16, g, "ident", n), ("prefill", 16, g, "perm", n)]
        p += [("decode", 16, 4, "ident", n), ("decode", 16, 4, "perm", n), ("decode", 4, 4, "ident", 16), ("decode", 1, 4, "ident", 4)]
    elif level == "heldout_full":
        p += [("prefill", 1, 4, "ident", 12), ("prefill", 4, 4, "ident", n), ("prefill", 4, 4, "perm", n)]
        for g in GAMMAS:
            p += [("prefill", 16, g, "ident", n)]
        p += [("prefill", 16, 4, "perm", n), ("decode", 16, 4, "ident", n), ("decode", 4, 4, "ident", 16), ("decode", 1, 4, "ident", 4)]
    elif level == "dev_light":
        p += [("prefill", 1, 4, "ident", 12), ("prefill", 4, 4, "ident", n), ("prefill", 4, 4, "perm", n),
              ("prefill", 16, 4, "ident", n), ("prefill", 16, 4, "perm", n), ("prefill", 16, 2, "ident", n), ("prefill", 16, 1, "ident", n),
              ("decode", 16, 4, "ident", n), ("decode", 4, 4, "ident", 16), ("decode", 1, 4, "ident", 4)]
    elif level == "heldout_light":
        p += [("prefill", 1, 4, "ident", 8), ("prefill", 4, 4, "ident", n), ("prefill", 16, 4, "ident", n), ("prefill", 16, 4, "perm", n),
              ("decode", 16, 4, "ident", n)]
    return p


def n_calls(pl):
    import math
    return sum(math.ceil(nc / B) for (_, B, _, _, nc) in pl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["G", "H"], required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--call-budget", type=int, required=True)
    ap.add_argument("--dev-level", default="dev")
    ap.add_argument("--heldout-level", default="heldout_full")
    a = ap.parse_args()
    out = Path(a.out)
    res = {"name": a.name, "mode": a.mode, "status": "started", "pid": os.getpid(), "calls": 0,
           "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("VLLM_", "HIP_", "ROCM_", "PYTHONPATH", "PYTHONUSERBASE", "PYTHONNOUSERSITE"))}}
    try:
        import torch
        import triton
        res["versions"] = {"torch": torch.__version__, "torch_file": torch.__file__, "triton": triton.__version__,
                           "triton_file": triton.__file__, "hip": getattr(torch.version, "hip", None)}
        if a.mode == "H" and not triton.__version__.startswith("3.3"):
            raise RuntimeError(f"H requires the torch-compatible Triton 3.3.x; imported {triton.__version__} from {triton.__file__}")
        import yaml
        import canonical_guard as guard
        import vllm
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from edge_specsim.target_client import VLLMCandidateVerifier
        from edge_specsim.provenance import collect_provenance

        res["versions"]["vllm"] = vllm.__version__
        cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
        guard.assert_clean_source()
        res["provenance"] = collect_provenance(REPO_ROOT, cfg)
        frozen = json.load(open(a.frozen))

        patch_state = None
        if a.mode == "G":
            import rocm_sdpa_runtime_patch as rp

            patch_state = rp.apply()  # BEFORE engine construction
        m, sim = cfg["models"], cfg["simulation"]
        t0 = time.time()
        ver = VLLMCandidateVerifier(
            model_name=m["target"], gpu_memory_utilization=float(m.get("target_gpu_memory_utilization", 0.7)),
            max_model_len=int(m.get("target_max_model_len", 4096)), dtype="auto", enforce_eager=False,
            enable_prefix_caching=False, sampling_temperature=0.0, seed=int(sim["seed"]), max_logprobs=2)
        llm = ver.llm
        res["init_s"] = time.time() - t0
        impls = impl_list(llm)
        ev = {"engine_class": f"{type(llm.llm_engine).__module__}.{type(llm.llm_engine).__name__}",
              "executor_class": type(llm.llm_engine.model_executor).__name__, "n_attention_layers": len(impls),
              "layers_use_naive_attn": sum(bool(i.use_naive_attn) for i in impls),
              "layers_use_triton_flash_attn": sum(bool(getattr(i, "use_triton_flash_attn", False)) for i in impls),
              "dtype": str(llm.llm_engine.model_config.dtype), "enforce_eager": bool(llm.llm_engine.model_config.enforce_eager),
              "chunked_prefill": bool(llm.llm_engine.scheduler_config.chunked_prefill_enabled),
              "prefix_caching": bool(llm.llm_engine.cache_config.enable_prefix_caching)}
        triton_counter = {"calls": 0, "pids": set()}
        if a.mode == "G":
            ev["layers_bound_to_patched_function"] = sum(bool(getattr(getattr(i, "sdpa_attn_func", None), "_aucspec_patched", False)) for i in impls)
            patch_state.strict = True  # from now on a violation raises instead of passing through
        else:
            for i in impls:
                orig = i.triton_attn_func

                def counted(*args, _o=orig, **kw):
                    triton_counter["calls"] += 1
                    triton_counter["pids"].add(os.getpid())
                    return _o(*args, **kw)

                i.triton_attn_func = counted
            ev["layers_have_naive_sdpa_func"] = sum(hasattr(i, "sdpa_attn_func") for i in impls)
        res["engine"] = ev

        lat: dict[str, list[float]] = {}

        def gen(prompts, sp, kind):
            if res["calls"] >= a.call_budget:
                raise RuntimeError("call budget exhausted")
            res["calls"] += 1
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

        def run_item(S, item):
            kind, B, g, order, nc = item
            n = len(S)
            idxs = list(range(n)) if order == "ident" else lib.make_perm(n, SHUF_SEED)
            idxs = idxs[:nc] if order == "ident" else idxs
            if kind == "prefill" and B == 1:
                idxs = list(range(nc))
            r = {}
            for s in range(0, len(idxs), B):
                grp = idxs[s: s + B]
                if kind == "prefill":
                    ps = [TokensPrompt(prompt_token_ids=S[i]["context"] + S[i]["proposal"][:g]) for i in grp]
                    outs = gen(ps, sp_pre, f"prefill_B{B}")
                    for pos, (i, o) in enumerate(zip(grp, outs)):
                        nn = len(S[i]["context"])
                        ent = [ser(o.prompt_logprobs[nn + j]) for j in range(g)]
                        r[i] = {"slot": s + pos, "group_pos": pos, "argmax": [e[0][0] for e in ent],
                                "top": ent, "proposal_lp": [float(o.prompt_logprobs[nn + j][S[i]["proposal"][j]].logprob) for j in range(g)]}
                else:
                    outs = gen([TokensPrompt(prompt_token_ids=list(S[i]["context"])) for i in grp], sp_dec, f"decode_B{B}")
                    for pos, (i, o) in enumerate(zip(grp, outs)):
                        r[i] = {"slot": s + pos, "group_pos": pos, "tokens": [int(t) for t in o.outputs[0].token_ids],
                                "top": [ser(d) for d in o.outputs[0].logprobs]}
            return f"{kind}_B{B}_g{g}_{order}", r

        def run_level(setname, level):
            S = frozen["sets"][setname]["contexts"]
            pl = plan(level, len(S))
            if res["calls"] + n_calls(pl) > a.call_budget:
                raise RuntimeError(f"plan {level} needs {n_calls(pl)} calls, budget left {a.call_budget - res['calls']}")
            M = {}
            for item in pl:
                k, r = run_item(S, item)
                M[k] = r
            return M

        def rate_vs_fp32(S, run, g):
            ok = tot = 0
            for i, r in run.items():
                for j in range(g):
                    f = S[i]["fp32"][j]
                    if f["gap"] >= lib.GAP_OK:
                        tot += 1
                        ok += r["argmax"][j] == f["top_ids"][0]
            return ok, tot

        def gates(S, M):
            b1 = M["prefill_B1_g4_ident"]
            out_ = {}
            for k, run in M.items():
                if not k.startswith("prefill") or k.startswith("prefill_B1_"):
                    continue
                g = int(k.split("_g")[1].split("_")[0])
                ok, tot = rate_vs_fp32(S, run, g)
                common = [i for i in run if i in b1]
                eq = sum(run[i]["argmax"][j] == b1[i]["argmax"][j] for i in common for j in range(g))
                out_[k] = {"vs_fp32_gap>=0.5": [ok, tot], "vs_B1": [eq, len(common) * g]}
            return out_

        def passes(gt):
            return all(v["vs_fp32_gap>=0.5"][1] > 0 and v["vs_fp32_gap>=0.5"][0] / v["vs_fp32_gap>=0.5"][1] >= lib.GATE
                       and (v["vs_B1"][1] == 0 or v["vs_B1"][0] / v["vs_B1"][1] >= lib.GATE) for v in gt.values())

        gen([TokensPrompt(prompt_token_ids=frozen["sets"]["dev"]["contexts"][0]["context"])], sp_dec, "warmup")
        gen([TokensPrompt(prompt_token_ids=c["context"]) for c in frozen["sets"]["dev"]["contexts"][:16]], sp_dec, "warmup")
        dev = frozen["sets"]["dev"]["contexts"]
        Md = run_level("dev", a.dev_level)
        gd = gates(dev, Md)
        res["dev"] = {"measurements": Md, "gates": gd, "pass": passes(gd)}
        if res["dev"]["pass"]:
            hl = a.heldout_level
            try:
                Mh = run_level("heldout", hl)
                gh = gates(frozen["sets"]["heldout"]["contexts"], Mh)
                res["heldout"] = {"level": hl, "measurements": Mh, "gates": gh, "pass": passes(gh)}
            except RuntimeError as exc:
                res["heldout_error"] = repr(exc)
        res["latency_ms"] = {k: {"n": len(v), "mean": sum(v) / len(v), "max": max(v)} for k, v in lat.items()}
        if a.mode == "G":
            res["patch_stats"] = patch_state.summary()
        else:
            res["triton_fa_stats"] = {"calls": triton_counter["calls"], "pids": sorted(triton_counter["pids"]), "pid_now": os.getpid()}
        res["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        res["status"] = "failed"
        res["error"] = repr(exc)[:1500]
        res["traceback"] = traceback.format_exc()[-2500:]
        try:
            if a.mode == "G":
                import rocm_sdpa_runtime_patch as rp

                res["patch_stats"] = rp.STATE.summary()
        except Exception:  # noqa: BLE001
            pass
    out.write_text(json.dumps(res))
    print(json.dumps({"name": a.name, "status": res["status"], "calls": res["calls"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
