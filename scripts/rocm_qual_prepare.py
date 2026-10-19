"""Fresh qualification contexts + independent HF fp32 reference (job-side).

Contexts: chat-template prompt + a model-generated history (HF fp32 greedy, left-padded batch) cut to an exact total length,
so each length group is an equal-length batch (no padding) for the fp32 reference. Questions are disjoint from the B4/B5 sets.
Reference per context: greedy 4-token continuation and fp32 top-2 at 5 positions (4 candidates + the position after them).
HF forward/generate calls: 6 per group (1 generate + 4 continuation forwards + 1 final forward).
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

GROUPS = {"dev": [108, 138, 168, 205], "heldout": [125, 190]}
PER_GROUP, CANDIDATES = 16, 20
EOS_IDS = {151643, 151645}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exclude-frozen", required=True, help="B4 frozen_contexts.json (its questions are excluded)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from edge_specsim.dataset import load_samples
    import rocm_fi_lib as lib

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    name = cfg["models"]["target"]
    tok = AutoTokenizer.from_pretrained(name)
    used = {c["q_index"] for s in json.load(open(a.exclude_frozen))["sets"].values() for c in s["contexts"]}
    samples = load_samples(cfg["dataset"])
    prompts = []
    for i, s in enumerate(samples):
        ids = list(tok.apply_chat_template([{"role": "user", "content": s.prompt}], tokenize=True, add_generation_prompt=True))
        prompts.append((i, ids))
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, attn_implementation="eager").to("cuda:0").eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = {"model": name, "hf_calls": 0, "hf_load_s": time.time() - t0, "sets": {}}

    for sname, lengths in GROUPS.items():
        flat = []
        for gi, L in enumerate(lengths):
            cand = [(i, p) for i, p in prompts if i not in used and L - 60 <= len(p) <= L - 4][:CANDIDATES]
            if len(cand) < PER_GROUP:
                raise RuntimeError(f"only {len(cand)} candidate questions for L={L}")
            for i, _ in cand:
                used.add(i)
            hs = [L - len(p) for _, p in cand]
            maxh = max(hs)
            width = max(len(p) for _, p in cand)
            ids = torch.full((len(cand), width), pad, dtype=torch.long)
            mask = torch.zeros((len(cand), width), dtype=torch.long)
            for r, (_, p) in enumerate(cand):
                ids[r, width - len(p):] = torch.tensor(p)
                mask[r, width - len(p):] = 1
            gc = GenerationConfig(do_sample=False, max_new_tokens=maxh, pad_token_id=pad, repetition_penalty=1.0, top_k=None,
                                  top_p=None, temperature=None)
            out["hf_calls"] += 1
            with torch.no_grad():
                gen = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), generation_config=gc)[:, width:].cpu()
            ctxs, meta = [], []
            for r, ((qi, p), h) in enumerate(zip(cand, hs)):
                hist = [int(x) for x in gen[r, :h]]
                if EOS_IDS & set(hist):
                    continue
                ctxs.append(p + hist)
                meta.append((qi, len(p), h))
            ids2 = torch.tensor(ctxs, device="cuda:0")
            n = ids2.shape[0]
            for _ in range(4):
                out["hf_calls"] += 1
                with torch.no_grad():
                    nxt = model(ids2).logits[:, -1].argmax(-1, keepdim=True)
                ids2 = torch.cat([ids2, nxt], dim=1)
            out["hf_calls"] += 1
            with torch.no_grad():
                logits = model(ids2).logits.float()
            kept = 0
            for r in range(n):
                props = [int(x) for x in ids2[r, L:L + 4]]
                pos = []
                for j in range(5):
                    lsm = torch.log_softmax(logits[r, L - 1 + j], dim=-1)
                    top = torch.topk(lsm, 2)
                    tid = [int(x) for x in top.indices]
                    tlp = [float(x) for x in top.values]
                    pos.append({"top_ids": tid, "top_logprobs": tlp, "gap": tlp[0] - tlp[1]})
                if EOS_IDS & (set(props) | {p["top_ids"][0] for p in pos}):
                    continue
                if kept == PER_GROUP:
                    break
                qi, plen, h = meta[r]
                flat.append({"q_index": qi, "group": gi, "L": L, "prompt_len": plen, "history_len": h, "context": ctxs[r],
                             "context_sha": lib.token_hash(ctxs[r]), "proposal": props, "fp32": pos})
                kept += 1
            if kept < PER_GROUP:
                raise RuntimeError(f"group L={L}: only {kept} valid contexts")
            del logits, ids2
            torch.cuda.empty_cache()
        out["sets"][sname] = {"lengths": lengths, "contexts": flat}
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps({"hf_calls": out["hf_calls"], "n": {k: len(v["contexts"]) for k, v in out["sets"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
