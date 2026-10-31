"""Does target-side KV reuse work with prompt_logprobs verification?

Runs R speculative rounds for B clients against VLLMCandidateVerifier,
growing each context by the committed tokens, and records per-round verify
wall time plus every committed token and candidate-position logprob. Run it
once with --prefix-caching on and once with off, then --compare the two JSON
files: the committed tokens must be identical and the round-2+ verify time
should drop with caching on if vLLM reuses the context KV.

Proposals are deterministic: the first `accept` tokens are the target's own
greedy continuation (generated outside the timed call), the rest filler, so
each round exercises acceptance, rejection and the correction token.

  python legacy/scripts/probe_verifier_cache.py --config configs/profile_target.yaml \
      --prefix-caching on --output results/section3/probe_cache_on.json
  python legacy/scripts/probe_verifier_cache.py --compare A.json B.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _run(args: argparse.Namespace) -> None:
    import vllm

    from edge_specsim.models import DraftResult
    from edge_specsim.target_client import VLLMCandidateVerifier
    from profile_target import _build_context_token_ids, _non_special_token_pool

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    m = cfg["models"]
    caching = args.prefix_caching == "on"
    verifier = VLLMCandidateVerifier(
        model_name=m["target"],
        gpu_memory_utilization=float(m.get("target_gpu_memory_utilization", 0.45)),
        max_model_len=int(m.get("target_max_model_len", 8192)),
        dtype=str(m.get("target_dtype", "auto")),
        enforce_eager=bool(m.get("target_enforce_eager", False)),
        enable_prefix_caching=caching,
        sampling_temperature=0.0,
        seed=int(cfg["simulation"]["seed"]),
    )
    try:
        from vllm import envs
        use_v1 = bool(envs.VLLM_USE_V1)
    except Exception:  # noqa: BLE001
        use_v1 = None
    pool = _non_special_token_pool(verifier)
    out = {"vllm_version": vllm.__version__, "vllm_use_v1": use_v1,
           "prefix_caching": caching, "config": args.config, "cases": []}

    for context_length in args.context_lengths:
        contexts = [_build_context_token_ids(pool, context_length, variant=context_length + 17 * b)
                    for b in range(args.batch_size)]
        case = {"batch_size": args.batch_size, "context_length": context_length,
                "gamma": args.gamma, "accept": args.accept, "rounds": []}
        for r in range(args.rounds):
            requests = []
            for b, ctx in enumerate(contexts):
                prefix, proposal = list(ctx), []
                for _ in range(min(args.accept, args.gamma)):
                    tok = verifier._generate_one_greedy_token(prefix)
                    proposal.append(int(tok)); prefix.append(int(tok))
                while len(proposal) < args.gamma:
                    proposal.append(pool[(r + b + len(proposal)) % len(pool)])
                draft = DraftResult(client_id=b, token_ids=proposal, draft_logprobs=[None] * len(proposal),
                                    text="", latency_ms=0.0, worker_id=-1,
                                    distribution_payload="delta_proposal")
                requests.append((list(ctx), draft, 10 ** 6))
            error = None
            start = time.perf_counter()
            try:
                results = verifier.verify_batch_sync(requests)
            except RuntimeError as exc:
                results, error = [], str(exc)
            wall_ms = (time.perf_counter() - start) * 1000.0
            row = {"round": r, "wall_ms": wall_ms, "error": error,
                   "accepted": [res.accepted_length for res in results],
                   "committed": [res.committed_token_ids for res in results],
                   "candidate_logprobs": [[max(d.values()) for d in res.target_logprobs] for res in results]}
            case["rounds"].append(row)
            print(f"caching={args.prefix_caching} L={context_length} round={r} "
                  f"wall={wall_ms:8.1f} ms accepted={row['accepted']} error={error}", flush=True)
            if error:
                break
            for ctx, res in zip(contexts, results):
                ctx.extend(res.committed_token_ids)
        out["cases"].append(case)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.output}  vllm={out['vllm_version']} V1={use_v1}")


def _compare(paths: list[str]) -> None:
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in paths)
    print(f"vllm {a['vllm_version']} V1={a['vllm_use_v1']}")
    ok = True
    for ca, cb in zip(a["cases"], b["cases"]):
        for ra, rb in zip(ca["rounds"], cb["rounds"]):
            same = ra["committed"] == rb["committed"] and not ra["error"] and not rb["error"]
            dlp = max((abs(x - y) for la, lb in zip(ra["candidate_logprobs"], rb["candidate_logprobs"])
                       for x, y in zip(la, lb)), default=0.0)
            ok &= same
            print(f"L={ca['context_length']:5d} round={ra['round']} "
                  f"wall {a['prefix_caching']}={ra['wall_ms']:8.1f} {b['prefix_caching']}={rb['wall_ms']:8.1f} ms "
                  f"same_tokens={same} max|dlogp|={dlp:.2e} err={ra['error'] or rb['error']}")
    print("IDENTICAL COMMITS" if ok else "!! COMMITS DIFFER OR ERRORS")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/profile_target.yaml")
    p.add_argument("--prefix-caching", choices=["on", "off"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context-lengths", type=lambda s: [int(x) for x in s.split(",")], default=[300, 2048])
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--accept", type=int, default=2)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--output")
    p.add_argument("--compare", nargs=2)
    args = p.parse_args()
    if args.compare:
        _compare(args.compare)
    else:
        if not (args.prefix_caching and args.output):
            p.error("--prefix-caching and --output are required unless --compare")
        _run(args)


if __name__ == "__main__":
    main()
