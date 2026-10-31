"""Does vLLM's prompt_logprobs argmax agree with the same engine's own greedy
generation? One engine (prefix caching off). For each client: generate gamma
greedy tokens one at a time (recording top-2 logprobs), then score
context + those tokens once with prompt_logprobs, and once more with the
prefix_fanout prompts. All three should name the same token at every
position; the top1-top2 margin tells numerical near-ties from a real bug.
Then the same prompts are scored again in one batched call (all clients'
prompt_logprobs prompts together; each client's fanout prompts one by one)
to separate a batching problem from a fanout problem.

  python legacy/scripts/selfcheck_prompt_logprobs.py --config configs/profile_target.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _top2(entry):
    items = sorted(((float(getattr(v, "logprob", v)), int(t)) for t, v in entry.items()), reverse=True)
    return [(t, round(lp, 4)) for lp, t in items[:2]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/profile_target.yaml")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context-lengths", type=lambda s: [int(x) for x in s.split(",")], default=[300])
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--disable-sliding-window", action="store_true",
                   help="vLLM disable_sliding_window=True (Qwen2.5 sets sliding_window=131072 "
                        "with use_sliding_window=false)")
    p.add_argument("--block-flash-attn", action="store_true",
                   help="make `import flash_attn` fail so ROCmFlashAttention falls back to naive SDPA")
    p.add_argument("--dtype")
    p.add_argument("--hf-no-sliding-window", action="store_true",
                   help="hf_overrides use_sliding_window=False, sliding_window=None")
    p.add_argument("--output", default="results/section3/selfcheck_prompt_logprobs.json")
    args = p.parse_args()

    if args.block_flash_attn:
        sys.modules["flash_attn"] = None  # ImportError on import
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    from profile_target import _build_context_token_ids

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    m = cfg["models"]
    llm = LLM(model=m["target"], gpu_memory_utilization=float(m.get("target_gpu_memory_utilization", 0.45)),
              max_model_len=int(m.get("target_max_model_len", 8192)), dtype=args.dtype or str(m.get("target_dtype", "auto")),
              max_logprobs=2, enable_prefix_caching=False, generation_config="vllm",
              enforce_eager=args.enforce_eager,
              disable_sliding_window=args.disable_sliding_window,
              **({"hf_overrides": {"use_sliding_window": False, "sliding_window": None}}
                 if args.hf_no_sliding_window else {}))
    tok = AutoTokenizer.from_pretrained(m["target"])
    ids = tok.encode(" alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron",
                     add_special_tokens=False)
    pool = list(dict.fromkeys(int(t) for t in ids))
    one = SamplingParams(temperature=0.0, max_tokens=1, logprobs=2, detokenize=False)
    scored = SamplingParams(temperature=0.0, max_tokens=1, logprobs=2, prompt_logprobs=2, detokenize=False)

    import os
    print(f"VLLM_USE_TRITON_FLASH_ATTN={os.environ.get('VLLM_USE_TRITON_FLASH_ATTN')} "
          f"enforce_eager={args.enforce_eager} disable_sliding_window={args.disable_sliding_window} "
          f"hf_no_sliding_window={args.hf_no_sliding_window} block_flash_attn={args.block_flash_attn} "
          f"dtype={args.dtype} VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}", flush=True)
    try:
        impl = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[0].self_attn.attn.impl
        print("attention impl:", type(impl).__name__,
              {k: getattr(impl, k) for k in ("use_naive_attn", "use_triton_flash_attn", "sliding_window")
               if hasattr(impl, k)}, flush=True)
    except Exception as exc:  # noqa: BLE001
        print("attention impl: unavailable", repr(exc)[:200], flush=True)
    report = []
    disagree = 0
    total = 0
    batch_bad = {"prompt_logprobs_batched": 0, "fanout_single": 0}
    for L in args.context_lengths:
        singles = []
        for b in range(args.batch_size):
            ctx = _build_context_token_ids(pool, L, variant=L + 17 * b)
            gen, gen_top = [], []
            for _ in range(args.gamma):
                out = llm.generate(TokensPrompt(prompt_token_ids=ctx + gen), one, use_tqdm=False)[0]
                gen.append(int(out.outputs[0].token_ids[0]))
                gen_top.append(_top2(out.outputs[0].logprobs[0]))
            full = llm.generate(TokensPrompt(prompt_token_ids=ctx + gen), scored, use_tqdm=False)[0]
            plp_top = [_top2(full.prompt_logprobs[L + k]) for k in range(args.gamma)]
            fan = llm.generate([TokensPrompt(prompt_token_ids=ctx + gen[:k]) for k in range(args.gamma)],
                               one, use_tqdm=False)
            fan_top = [_top2(o.outputs[0].logprobs[0]) for o in fan]
            for k in range(args.gamma):
                total += 1
                ok = gen_top[k][0][0] == plp_top[k][0][0] == fan_top[k][0][0]
                disagree += not ok
                row = {"L": L, "client": b, "pos": k, "generated": gen[k], "gen_top2": gen_top[k],
                       "prompt_logprobs_top2": plp_top[k], "fanout_top2": fan_top[k], "agree": ok}
                report.append(row)
                if not ok:
                    print("DISAGREE", json.dumps(row), flush=True)
            singles.append((ctx, gen, gen_top))
            for k in range(args.gamma):
                one_out = llm.generate(TokensPrompt(prompt_token_ids=ctx + gen[:k]), one, use_tqdm=False)[0]
                top = _top2(one_out.outputs[0].logprobs[0])
                if top[0][0] != gen_top[k][0][0]:
                    batch_bad["fanout_single"] += 1
                    print("FANOUT_SINGLE_DISAGREE", json.dumps({"L": L, "client": b, "pos": k,
                          "gen_top2": gen_top[k], "single_top2": top}), flush=True)
        batched = llm.generate([TokensPrompt(prompt_token_ids=c + g) for c, g, _ in singles], scored,
                               use_tqdm=False)
        for b, ((c, g, gt), out) in enumerate(zip(singles, batched)):
            for k in range(args.gamma):
                top = _top2(out.prompt_logprobs[L + k])
                if top[0][0] != gt[k][0][0]:
                    batch_bad["prompt_logprobs_batched"] += 1
                    print("BATCHED_PLP_DISAGREE", json.dumps({"L": L, "client": b, "pos": k,
                          "gen_top2": gt[k], "batched_top2": top}), flush=True)
    print(f"{disagree}/{total} positions disagree (single-prompt prompt_logprobs and batched fanout vs generation)")
    print(f"batched prompt_logprobs (B={args.batch_size} in one call) disagree: "
          f"{batch_bad['prompt_logprobs_batched']}/{total}")
    print(f"fanout prompts scored one at a time disagree: {batch_bad['fanout_single']}/{total}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
