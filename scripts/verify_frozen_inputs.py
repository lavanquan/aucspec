"""Integrity check of the frozen qualification inputs (CPU only): sha256, structure, hashes and fp32 consistency."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rocm_fi_lib as fl  # noqa: E402

EOS_IDS = {151643, 151645}
EXPECTED = {"dev": (64, [108, 138, 168, 205]), "heldout": (32, [125, 190])}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_frozen(path, expect_sha256=None, expected=EXPECTED) -> dict:
    """Returns {"ok": bool, "problems": [...], "sha256": ..., "counts": {...}}; never raises on bad content."""
    problems = []
    p = Path(path)
    if not p.exists():
        return {"ok": False, "problems": ["file missing"], "sha256": None, "counts": {}}
    sha = sha256_file(p)
    if expect_sha256 and sha != expect_sha256:
        problems.append(f"sha256 {sha} != expected {expect_sha256}")
    try:
        d = json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "problems": [f"unreadable json: {exc!r}"], "sha256": sha, "counts": {}}
    counts = {}
    for s, (n, lengths) in expected.items():
        ctxs = (d.get("sets", {}).get(s) or {}).get("contexts") or []
        counts[s] = len(ctxs)
        if len(ctxs) != n:
            problems.append(f"{s}: {len(ctxs)} contexts, expected {n}")
        seen_q = set()
        for c in ctxs:
            if fl.token_hash(c["context"]) != c["context_sha"]:
                problems.append(f"{s}: context hash mismatch for q {c.get('q_index')}")
            if len(c["context"]) != c["L"] or c["L"] not in lengths:
                problems.append(f"{s}: unexpected length {len(c['context'])} / {c['L']}")
            if len(c["proposal"]) != 4 or len(c["fp32"]) != 5:
                problems.append(f"{s}: wrong proposal/fp32 shape for q {c.get('q_index')}")
                continue
            for j, f in enumerate(c["fp32"]):
                if len(f["top_ids"]) != 2 or len(f["top_logprobs"]) != 2 or abs((f["top_logprobs"][0] - f["top_logprobs"][1]) - f["gap"]) > 1e-4 \
                        or f["top_logprobs"][0] < f["top_logprobs"][1]:
                    problems.append(f"{s}: bad fp32 top-2 at q {c['q_index']} pos {j}")
            if any(c["proposal"][j] != c["fp32"][j]["top_ids"][0] for j in range(4)):
                problems.append(f"{s}: proposal is not the fp32 greedy chain for q {c['q_index']}")
            if EOS_IDS & (set(c["proposal"]) | {f["top_ids"][0] for f in c["fp32"]} | set(c["context"][c["prompt_len"]:])):
                problems.append(f"{s}: EOS inside context/proposal for q {c['q_index']}")
            if c["q_index"] in seen_q:
                problems.append(f"{s}: duplicate question {c['q_index']}")
            seen_q.add(c["q_index"])
        groups = sorted({c["L"] for c in ctxs})
        if groups != sorted(lengths):
            problems.append(f"{s}: length groups {groups}")
    dev_q = {c["q_index"] for c in (d.get("sets", {}).get("dev") or {}).get("contexts", [])}
    ho_q = {c["q_index"] for c in (d.get("sets", {}).get("heldout") or {}).get("contexts", [])}
    if dev_q & ho_q:
        problems.append("dev and held-out share questions")
    return {"ok": not problems, "problems": problems[:20], "n_problems": len(problems), "sha256": sha, "counts": counts}


def check_never_seen_by_vllm(qual_result_path) -> dict:
    """The earlier job must have aborted before any vLLM call (calls == 0, no engine evidence)."""
    r = json.loads(Path(qual_result_path).read_text())
    ok = r.get("calls") == 0 and r.get("status") == "failed" and "runtime_evidence" not in r and "dev" not in r
    return {"ok": bool(ok), "calls": r.get("calls"), "status": r.get("status"), "error": str(r.get("error"))[:160]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("frozen")
    ap.add_argument("--expect-sha256", default=None)
    ap.add_argument("--previous-qual-result", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = check_frozen(a.frozen, a.expect_sha256)
    if a.previous_qual_result:
        rep["previous_job_never_ran_vllm"] = check_never_seen_by_vllm(a.previous_qual_result)
        rep["ok"] = rep["ok"] and rep["previous_job_never_ran_vllm"]["ok"]
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    sys.exit(0 if rep["ok"] else 3)
