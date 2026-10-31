"""Does vLLM's prompt_logprobs argmax agree with the same engine's own greedy
generation? One engine (prefix caching off). For each client: generate gamma
greedy tokens one at a time (recording top-2 logprobs), then score
context + those tokens once with prompt_logprobs, and once more with the
prefix_fanout prompts. All three should name the same token at every
position; the top1-top2 margin tells numerical near-ties from a real bug.

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
    p.add_argument("--output", default="results/section3/selfcheck_prompt_logprobs.json")
    args = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    from profile_target import _build_context_token_ids

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    m = cfg["models"]
    llm = LLM(model=m["target"], gpu_memory_utilization=float(m.get("target_gpu_memory_utilization", 0.45)),
              max_model_len=int(m.get("target_max_model_len", 8192)), dtype=str(m.get("target_dtype", "auto")),
              max_logprobs=2, enable_prefix_caching=False, generation_config="vllm")
    tok = AutoTokenizer.from_pretrained(m["target"])
    ids = tok.encode(" alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron",
                     add_special_tokens=False)
    pool = list(dict.fromkeys(int(t) for t in ids))
    one = SamplingParams(temperature=0.0, max_tokens=1, logprobs=2, detokenize=False)
    scored = SamplingParams(temperature=0.0, max_tokens=1, logprobs=2, prompt_logprobs=2, detokenize=False)

    report = []
    disagree = 0
    total = 0
    for L in args.context_lengths:
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
    print(f"{disagree}/{total} positions disagree")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
