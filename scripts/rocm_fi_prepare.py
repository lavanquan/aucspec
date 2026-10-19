"""Freeze tokenised contexts and the independent fp32 HuggingFace reference (job-side, one process).

Contexts = chat-template prompt + the first (L - p) tokens of the GSM8K gold solution text (natural text,
no model generation involved), so every comparison uses identical frozen token ids.
Dev set: 32 contexts of length 158. Held-out set: 32 other questions, length 205.
fp32 reference: greedy continuation of 4 tokens by repeated forward passes (equal lengths, no padding),
then one forward over context+continuation for top-2 logprobs at the 4 candidate positions.
HF forward passes: 5 per set, 10 in total.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "legacy" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rocm_fi_lib as lib  # noqa: E402

SETS = {"dev": 158, "heldout": 205}
N_PER_SET = 32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from edge_specsim.dataset import load_samples

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    name = cfg["models"]["target"]
    tok = AutoTokenizer.from_pretrained(name)
    samples = load_samples(cfg["dataset"])
    cands = []
    for i, s in enumerate(samples):
        msgs = [{"role": "user", "content": s.prompt}]
        p_ids = list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
        ref_ids = tok(s.reference or "", add_special_tokens=False)["input_ids"]
        cands.append((i, p_ids, ref_ids))

    used: set[int] = set()
    out = {"model": name, "sets": {}, "hf_forward_calls": 0}
    plan = {}
    for sname, L in SETS.items():
        picked = []
        for i, p_ids, ref_ids in cands:
            if i in used:
                continue
            h = L - len(p_ids)
            if h < 0 or len(ref_ids) < h + 8:
                continue
            picked.append((i, p_ids + ref_ids[:h]))
            used.add(i)
            if len(picked) == N_PER_SET:
                break
        if len(picked) < N_PER_SET:
            raise RuntimeError(f"only {len(picked)} contexts for set {sname}")
        plan[sname] = picked

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, attn_implementation="eager").to("cuda:0").eval()
    out["hf_load_s"] = time.time() - t0
    for sname, picked in plan.items():
        ids = torch.tensor([c for _, c in picked], device="cuda:0")
        L = ids.shape[1]
        for _ in range(4):
            out["hf_forward_calls"] += 1
            with torch.no_grad():
                nxt = model(ids).logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        out["hf_forward_calls"] += 1
        with torch.no_grad():
            logits = model(ids).logits.float()
        recs = []
        for row, (qi, ctx) in enumerate(picked):
            props = [int(x) for x in ids[row, L:L + 4]]
            pos = []
            for j in range(4):
                lsm = torch.log_softmax(logits[row, L - 1 + j], dim=-1)
                top = torch.topk(lsm, 2)
                tid = [int(x) for x in top.indices]
                tlp = [float(x) for x in top.values]
                pos.append({"top_ids": tid, "top_logprobs": tlp, "gap": tlp[0] - tlp[1],
                            "proposal_logprob": float(lsm[props[j]]), "consistent_with_greedy_step": tid[0] == props[j]})
            recs.append({"q_index": qi, "context": ctx, "context_len": len(ctx), "context_sha": lib.token_hash(ctx),
                         "proposal": props, "fp32": pos})
        out["sets"][sname] = {"L": L, "contexts": recs}
    Path(a.out).write_text(json.dumps(out))
    inc = sum(not p["consistent_with_greedy_step"] for s in out["sets"].values() for r in s["contexts"] for p in r["fp32"])
    print(json.dumps({"hf_forward_calls": out["hf_forward_calls"], "inconsistent_positions": inc, "load_s": out["hf_load_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
