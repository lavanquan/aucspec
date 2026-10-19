"""Bounded GPU diagnostic: why do target-generated greedy proposals fail V1 verification?

DIAGNOSTIC configuration, not the canonical verifier: the vLLM engine is built with max_logprobs=2 so that the
top-2 gap can be observed. V1 behaviour is reproduced by (a) requesting prompt_logprobs=1/logprobs=1 and
(b) calling the UNMODIFIED VLLMCandidateVerifier.verify_batch_sync with that k=1 sampling.
An independent fp32 HuggingFace forward pass is used as a numerical oracle (not the verifier itself).

Hard cap on target-model calls (every llm.generate call and every HF forward counts as one).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "legacy" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

CALL_CAP = 200
GAMMAS = (1, 2, 4)
BATCHES = (16, 4, 1)
POOL_LENGTHS = (158, 205)
POOL_SIZE = 16
V1_B1_CONTEXTS = (0, 7, 8, 9, 10, 11, 12, 15)  # mix of previously accepted/rejected L=158 slots
DECODE_B1_CONTEXTS = (0, 7, 8, 11)


def ser(d):
    return {int(t): {"lp": float(v.logprob), "rank": getattr(v, "rank", None)} for t, v in d.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--calib-raw", required=True, help="calibration_raw.jsonl, to check the pool is reproduced")
    ap.add_argument("--seed", type=int, default=20261019)
    a = ap.parse_args()

    import yaml
    import canonical_guard as guard
    from edge_specsim.models import DraftResult
    from edge_specsim.provenance import collect_provenance, config_sha256
    from edge_specsim.target_client import VLLMCandidateVerifier
    from edge_specsim.dataset import load_samples
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    guard.assert_clean_source()
    guard.validate_canonical_config(cfg)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    calls = {"n": 0, "log": []}

    def count(kind: str, n_seq: int):
        calls["n"] += 1
        calls["log"].append((kind, n_seq))
        if calls["n"] > CALL_CAP:
            raise RuntimeError(f"target-model call cap {CALL_CAP} exceeded")

    m, sim = cfg["models"], cfg["simulation"]
    t0 = time.time()
    verifier = VLLMCandidateVerifier(
        model_name=m["target"],
        gpu_memory_utilization=float(m.get("target_gpu_memory_utilization", 0.88)),
        max_model_len=int(m.get("target_max_model_len", 8192)),
        dtype=str(m.get("target_dtype", "auto")),
        enforce_eager=bool(m.get("target_enforce_eager", False)),
        enable_prefix_caching=bool(m.get("target_enable_prefix_caching", True)),
        sampling_temperature=0.0,
        seed=int(sim["seed"]),
        max_logprobs=2,  # DIAGNOSTIC: canonical V1 is max_logprobs=1
    )
    llm = verifier.llm
    eos = verifier.eos_token_id

    def sp_prefill(k: int):
        return SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, prompt_logprobs=k, logprobs=k,
                              detokenize=False, skip_special_tokens=False)

    def gen(prompts, sps, kind):
        count(kind, len(prompts))
        return llm.generate(prompts, sps, use_tqdm=False)

    # ---- replicate the calibration pool (same RNG, same selection, one batched decode call)
    samples = load_samples(cfg["dataset"])
    prompts = [(s.prompt, verifier.build_context_token_ids(s.prompt, [])) for s in samples]
    prng = random.Random(a.seed + 7)
    lengths = [L for L in (113, 158, 205) for _ in range(20)]
    used: set[int] = set()
    plans = []
    for L in lengths:
        cand = [i for i, (_, ids) in enumerate(prompts) if i not in used and L - 70 <= len(ids) <= L]
        if not cand:
            cand = [i for i, (_, ids) in enumerate(prompts) if i not in used and len(ids) <= L]
        i = prng.choice(cand)
        used.add(i)
        plans.append((i, L, L - len(prompts[i][1])))
    sps = [SamplingParams(temperature=0.0, top_p=1.0, max_tokens=h + 5, logprobs=2, detokenize=False)
           for (_, _, h) in plans]
    outs = gen([TokensPrompt(prompt_token_ids=list(prompts[i][1])) for (i, _, _) in plans], sps, "pool_decode_B60")
    pool_all = []
    for (i, L, h), o in zip(plans, outs):
        toks = [int(t) for t in o.outputs[0].token_ids]
        if len(toks) < h + 4 or (eos is not None and eos in toks[: h + 4]):
            continue
        lps = [ser(d) for d in o.outputs[0].logprobs[h: h + 4]]
        pool_all.append({"q_index": i, "L": L, "h": h, "context": list(prompts[i][1]) + toks[:h],
                         "proposal": toks[h: h + 4], "decode_logprobs": lps, "prompt_len": len(prompts[i][1])})
    pools = {L: [e for e in pool_all if e["L"] == L][:POOL_SIZE] for L in POOL_LENGTHS}
    # reproducibility check against the calibration job's per-slot first-token outcomes
    prev = {}
    for line in Path(a.calib_raw).read_text().splitlines():
        r = json.loads(line)
        if r.get("phase") in ("timed", "warmup") and r["gamma_requested"][0] > 0:
            for slot, acc in enumerate(r["gamma_realized_accepted"]):
                prev.setdefault((r["context_lens"][0], slot), set()).add(acc >= 1)
    contexts = []
    for L in POOL_LENGTHS:
        for slot, e in enumerate(pools[L]):
            e["calibration_first_token_accepted"] = (sorted(prev[(L, slot)])[0] if len(prev.get((L, slot), ())) == 1 else None)
            e["pool_slot"] = slot
            contexts.append(e)

    # ---- D2: decode at B=1 (batch invariance of the decode path)
    decode_b1 = []
    for slot in DECODE_B1_CONTEXTS:
        e = pools[158][slot]
        o = gen([TokensPrompt(prompt_token_ids=list(e["context"][: e["prompt_len"]]))],
                [SamplingParams(temperature=0.0, top_p=1.0, max_tokens=e["h"] + 5, logprobs=2, detokenize=False)],
                "decode_B1")[0]
        toks = [int(t) for t in o.outputs[0].token_ids]
        decode_b1.append({"slot": slot, "proposal": toks[e["h"]: e["h"] + 4],
                          "same_as_B60": toks[e["h"]: e["h"] + 4] == e["proposal"],
                          "history_same": toks[: e["h"]] == e["context"][e["prompt_len"]:],
                          "logprobs": [ser(d) for d in o.outputs[0].logprobs[e["h"]: e["h"] + 4]]})

    # ---- prefill with top-2 dictionaries across B and gamma
    def prefill_records(k: int, batch: int, gamma: int, ctx_ids):
        recs = {}
        sp = sp_prefill(k)
        for s in range(0, len(ctx_ids), batch):
            group = ctx_ids[s: s + batch]
            ps = [TokensPrompt(prompt_token_ids=contexts[c]["context"] + contexts[c]["proposal"][:gamma]) for c in group]
            os_ = gen(ps, sp, f"prefill_k{k}_B{batch}_g{gamma}")
            for c, o in zip(group, os_):
                n = len(contexts[c]["context"])
                plp = o.prompt_logprobs
                recs[c] = {
                    "entries": [ser(plp[n + j]) for j in range(gamma)],
                    "entries_shift_minus1": [ser(plp[n + j - 1]) for j in range(gamma)],
                    "entries_shift_plus1": [ser(plp[n + j + 1]) if n + j + 1 < len(plp) else None for j in range(gamma)],
                    "sampler_next_token": int(o.outputs[0].token_ids[0]) if o.outputs[0].token_ids else None,
                    "sampler_next_logprobs": ser(o.outputs[0].logprobs[0]) if o.outputs[0].logprobs else None,
                }
        return recs

    all_ctx = list(range(len(contexts)))
    prefill = {}
    for B in BATCHES:
        for g in GAMMAS:
            prefill[f"k2_B{B}_g{g}"] = prefill_records(2, B, g, all_ctx)

    # ---- UNMODIFIED V1 verify_batch_sync with k=1 sampling (exact V1 request settings)
    verifier.sampling = sp_prefill(1)

    def draft_of(tokens):
        return DraftResult(client_id=0, token_ids=list(tokens), draft_logprobs=[], text="", latency_ms=0.0,
                           worker_id=-1, distribution_payload="delta_proposal")

    v1_runs = {}
    b1_set = [pools[158][s]["pool_slot"] for s in V1_B1_CONTEXTS]
    b1_idx = [contexts.index(pools[158][s]) for s in V1_B1_CONTEXTS]
    for B, idxs in ((16, all_ctx), (4, all_ctx), (1, b1_idx)):
        for g in GAMMAS:
            rec = {}
            for s in range(0, len(idxs), B):
                group = idxs[s: s + B]
                count(f"v1_verify_B{B}_g{g}", len(group))
                res = verifier.verify_batch_sync(
                    [(contexts[c]["context"], draft_of(contexts[c]["proposal"][:g]), 96) for c in group])
                for c, r in zip(group, res):
                    rec[c] = {"accepted_length": r.accepted_length, "correction": r.correction_token_id,
                              "bonus": r.bonus_token_id, "target_greedy": r.target_greedy_token_ids,
                              "reached_eos": r.reached_eos}
            v1_runs[f"V1k1_B{B}_g{g}"] = rec

    # also the exact-V1 k=1 dictionaries (to compare with k=2 argmax)
    prefill["k1_B16_g4"] = prefill_records(1, 16, 4, all_ctx) if calls["n"] + 2 <= CALL_CAP else None

    meta = {
        "diagnostic_config": "vLLM max_logprobs=2 (canonical V1 uses 1); V1 requests emulated with k=1",
        "provenance": collect_provenance(REPO_ROOT, cfg), "config_sha256": config_sha256(cfg),
        "calls_total_before_oracle": calls["n"], "call_log": calls["log"], "seed": a.seed,
        "contexts": [{k: v for k, v in e.items() if k not in ("context",)} | {"context_len": len(e["context"])}
                     for e in contexts],
        "vllm_load_seconds": time.time() - t0,
    }
    raw = {"meta": meta, "contexts": contexts, "decode_b1": decode_b1, "prefill": prefill, "v1_runs": v1_runs}
    (out / "diag_raw.json").write_text(json.dumps(raw))
    print("vLLM phase done; calls so far", calls["n"], flush=True)

    # ---- independent fp32 HuggingFace oracle (2 forward calls: equal-length groups, no padding)
    oracle = {}
    try:
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(m["target"], torch_dtype=torch.float32).to("cuda:1").eval()
        for L in POOL_LENGTHS:
            idxs = [c for c, e in enumerate(contexts) if e["L"] == L]
            ids = torch.tensor([contexts[c]["context"] + contexts[c]["proposal"] for c in idxs], device="cuda:1")
            count(f"hf_fp32_forward_L{L}", len(idxs))
            with torch.no_grad():
                logits = model(ids).logits.float()
            for row, c in enumerate(idxs):
                n = len(contexts[c]["context"])
                per_pos = []
                for j in range(4):
                    lg = logits[row, n + j - 1]  # logits[i] predicts token i+1
                    lsm = torch.log_softmax(lg, dim=-1)
                    top = torch.topk(lsm, 2)
                    prop = contexts[c]["proposal"][j]
                    per_pos.append({"top_ids": [int(x) for x in top.indices], "top_logprobs": [float(x) for x in top.values],
                                    "proposal_logprob": float(lsm[prop]),
                                    "proposal_rank": int((lsm > lsm[prop]).sum()) + 1})
                oracle[c] = per_pos
        raw["oracle_fp32"] = oracle
        meta["calls_total"] = calls["n"]
    except Exception as exc:  # noqa: BLE001 - keep the vLLM data even if the oracle fails
        raw["oracle_error"] = repr(exc)
    (out / "diag_raw.json").write_text(json.dumps(raw))
    print(json.dumps({"calls_total": calls["n"], "oracle": "oracle_fp32" in raw}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
