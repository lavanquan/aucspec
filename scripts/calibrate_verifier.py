"""Bounded verifier-cost calibration through the real verify_batch_sync path.

One GPU job, one node. Measures the full-batch wall time of the exact verifier the
simulator uses (V1 greedy semantics, same model/vLLM settings), over a randomised grid
of batch size x context length x speculative depth, plus a heterogeneous held-out set.
No regime (memory/compute bound) is assumed here; points are recorded raw and any
regime analysis is done afterwards from the data.

CPU-testable core: build_schedule / run_schedule / check_realized (no torch import).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "legacy" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

BATCH_SIZES = (1, 2, 4, 8, 16)
GAMMAS = (0, 1, 2, 3, 4)
CONTEXT_QUANTILES = {"p10": 113, "p50": 158, "p90": 205}  # canonical N=10 smoke traces, all rounds (see report)
WARMUP_REPS = 3
TIMED_REPS = 20
VAL_CELLS = 30
VAL_REPS = 10
ENGINE_WARM_CALLS = 10


# ----------------------------------------------------------------- schedule
def build_schedule(seed: int, warmup_reps: int = WARMUP_REPS, timed_reps: int = TIMED_REPS,
                   val_cells: int = VAL_CELLS, val_reps: int = VAL_REPS) -> dict:
    """Deterministic randomised order. Warm-ups first (all cells, shuffled, `warmup_reps` passes),
    then `timed_reps` rounds, each a fresh permutation of all primary cells, then the held-out set."""
    rng = random.Random(seed)
    cells = [{"cell": f"B{b}_L{lq}_g{g}", "B": b, "Lq": lq, "gamma": g}
             for b in BATCH_SIZES for lq in CONTEXT_QUANTILES for g in GAMMAS]
    items: list[dict] = []
    for r in range(warmup_reps):
        order = cells[:]
        rng.shuffle(order)
        items += [{"phase": "warmup", "rep": r, **c} for c in order]
    for r in range(timed_reps):
        order = cells[:]
        rng.shuffle(order)
        items += [{"phase": "timed", "rep": r, **c} for c in order]
    vrng = random.Random(seed + 1)
    val = []
    sizes = [2, 4, 8, 16]
    for i in range(val_cells):
        b = sizes[i % len(sizes)]
        val.append({"cell": f"V{i:02d}_B{b}", "B": b,
                    "pool_idx": sorted(vrng.sample(range(32), b)),
                    "gammas": [vrng.randint(0, 4) for _ in range(b)]})
    for r in range(val_reps):
        order = val[:]
        vrng.shuffle(order)
        items += [{"phase": "val", "rep": r, **c} for c in order]
    for k, it in enumerate(items):
        it["order"] = k
        it["item_id"] = f"{it['phase']}:{it['rep']}:{it['cell']}"
    return {"cells": cells, "val_cells": val, "items": items}


def expected_call_counts(sched: dict) -> dict:
    n = {"warmup": 0, "timed": 0, "val": 0}
    for it in sched["items"]:
        n[it["phase"]] += 1
    n["engine_warm"] = ENGINE_WARM_CALLS
    n["pool_generate_calls"] = 2
    n["total_verifier_and_prep_calls"] = n["warmup"] + n["timed"] + n["val"] + n["engine_warm"] + n["pool_generate_calls"]
    return n


def check_realized(gammas: list[int], results) -> tuple[bool, list[int]]:
    """A cell is well formed when every request had all of its gamma proposed tokens accepted
    (greedy proposals taken from the target's own continuation) and no EOS truncated it."""
    acc = [int(r.accepted_length) for r in results]
    ok = all(a == g for a, g in zip(acc, gammas)) and not any(r.reached_eos for r in results)
    return ok, acc


def run_schedule(items, make_payload, verify_fn, gen_timer, sink, done_ids, deadline_s=None, clock=time.perf_counter):
    """Execute items in order, appending one JSON record per call via `sink`. `make_payload(item)` returns
    (payload, meta); `verify_fn(payload)` is the real verify_batch_sync; `gen_timer()` returns the
    last llm.generate duration in ms. Returns (n_done, stopped_early)."""
    t_start = clock()
    n = 0
    for it in items:
        if it["item_id"] in done_ids:
            continue
        if deadline_s is not None and (clock() - t_start) > deadline_s:
            return n, True
        payload, meta = make_payload(it)
        t0 = clock()
        results = verify_fn(payload)
        wall_ms = (clock() - t0) * 1000.0
        gen_ms = gen_timer()
        ok, acc = check_realized(meta["gammas"], results)
        rec = {
            **{k: it[k] for k in ("item_id", "phase", "rep", "cell", "order")},
            "B": len(payload),
            "gamma_requested": meta["gammas"],
            "gamma_realized_accepted": acc,
            "well_formed": ok,
            "context_lens": meta["context_lens"],
            "sum_context": int(sum(meta["context_lens"])),
            "sum_gamma": int(sum(meta["gammas"])),
            "logical_tokens": int(sum(g + 1 for g in meta["gammas"])),
            "processed_tokens": int(sum(meta["context_lens"]) + sum(meta["gammas"])),
            "wall_ms": wall_ms,
            "stamped_batch_wall_ms": float(results[0].target_batch_wall_ms) if results else None,
            "generate_ms": gen_ms,
            "host_ms": wall_ms - gen_ms,
            "t_unix": time.time(),
        }
        sink(rec)
        n += 1
    return n, False


# -------------------------------------------------------------- GPU side
def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            out.add(json.loads(line)["item_id"])
        except Exception:  # noqa: BLE001 - tolerate a torn last line after a kill
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True, help="scratch directory for raw JSONL (checkpointed)")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--deadline-min", type=float, default=38.0, help="stop starting new calls after this")
    ap.add_argument("--dry-schedule", action="store_true", help="print call counts and exit (CPU)")
    a = ap.parse_args()

    sched = build_schedule(a.seed)
    counts = expected_call_counts(sched)
    if a.dry_schedule:
        print(json.dumps(counts, indent=1))
        return 0

    import yaml
    import canonical_guard as guard
    from edge_specsim.provenance import collect_provenance, config_sha256, verifier_semantics_record

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    prov_git = guard.assert_clean_source()
    guard.validate_canonical_config(cfg)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "calibration_raw.jsonl"
    done = _load_done(raw)

    # --- exact simulator target settings
    from edge_specsim.models import DraftResult
    from edge_specsim.target_client import VLLMCandidateVerifier
    from edge_specsim.dataset import load_samples
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    m, sim = cfg["models"], cfg["simulation"]
    target = VLLMCandidateVerifier(
        model_name=m["target"],
        gpu_memory_utilization=float(m.get("target_gpu_memory_utilization", 0.88)),
        max_model_len=int(m.get("target_max_model_len", 8192)),
        dtype=str(m.get("target_dtype", "auto")),
        enforce_eager=bool(m.get("target_enforce_eager", False)),
        enable_prefix_caching=bool(m.get("target_enable_prefix_caching", True)),
        sampling_temperature=0.0,
        seed=int(sim["seed"]),
    )
    # time llm.generate separately (non-invasive wrapper)
    last = {"ms": 0.0}
    orig_generate = target.llm.generate

    def timed_generate(*args, **kw):
        t0 = time.perf_counter()
        r = orig_generate(*args, **kw)
        last["ms"] = (time.perf_counter() - t0) * 1000.0
        return r

    target.llm.generate = timed_generate

    # --- context pools from real GSM8K prompts + target-greedy history (preparation: 2 generate calls)
    samples = load_samples(cfg["dataset"])
    prng = random.Random(a.seed + 7)
    eos = target.eos_token_id
    prompts = []
    for s in samples:
        ids = target.build_context_token_ids(s.prompt, [])
        prompts.append((s.prompt, ids))

    def make_pool(lengths: list[int]) -> list[dict]:
        used: set[int] = set()
        reqs, plans = [], []
        for L in lengths:
            cand = [i for i, (_, ids) in enumerate(prompts) if i not in used and L - 70 <= len(ids) <= L]
            if not cand:
                cand = [i for i, (_, ids) in enumerate(prompts) if i not in used and len(ids) <= L]
            i = prng.choice(cand)
            used.add(i)
            h = L - len(prompts[i][1])
            plans.append((i, L, h))
            reqs.append(TokensPrompt(prompt_token_ids=list(prompts[i][1])))
        sps = [SamplingParams(temperature=0.0, top_p=1.0, max_tokens=h + 5, detokenize=False) for (_, _, h) in plans]
        outs = target.llm.generate(reqs, sps, use_tqdm=False)
        pool = []
        for (i, L, h), o in zip(plans, outs):
            toks = [int(t) for t in o.outputs[0].token_ids]
            if len(toks) < h + 4 or (eos is not None and eos in toks[: h + 4]):
                continue  # cannot build 4 accepted proposal tokens; skip (counted below)
            ctx = list(prompts[i][1]) + toks[:h]
            assert len(ctx) == L
            pool.append({"q_index": i, "L": L, "context": ctx, "proposal": toks[h: h + 4]})
        return pool

    t_prep = time.perf_counter()
    prim_lengths = [L for L in CONTEXT_QUANTILES.values() for _ in range(20)]  # 20 per class, >=16 survive
    prim_pool_all = make_pool(prim_lengths)
    pools: dict[int, list[dict]] = {}
    for L in CONTEXT_QUANTILES.values():
        p = [e for e in prim_pool_all if e["L"] == L][:16]
        if len(p) < 16:
            raise RuntimeError(f"only {len(p)} valid contexts for L={L}")
        pools[L] = p
    vrng = random.Random(a.seed + 3)
    val_lengths = [vrng.randint(100, 230) for _ in range(40)]
    val_pool = make_pool(val_lengths)[:32]
    if len(val_pool) < 32:
        raise RuntimeError(f"only {len(val_pool)} valid validation contexts")
    prep_s = time.perf_counter() - t_prep

    pool_hash = hashlib.sha256(json.dumps([[e["context"], e["proposal"]] for L in sorted(pools) for e in pools[L]]
                                          + [[e["context"], e["proposal"]] for e in val_pool]).encode()).hexdigest()

    def draft_of(tokens: list[int]) -> DraftResult:
        return DraftResult(client_id=0, token_ids=list(tokens), draft_logprobs=[], text="", latency_ms=0.0,
                           worker_id=-1, distribution_payload="delta_proposal")

    def make_payload(it):
        if it["phase"] == "val":
            ents = [val_pool[j] for j in it["pool_idx"]]
            gammas = it["gammas"]
        else:
            L = CONTEXT_QUANTILES[it["Lq"]]
            ents = pools[L][: it["B"]]
            gammas = [it["gamma"]] * it["B"]
        payload = [(e["context"], draft_of(e["proposal"][:g]), 96) for e, g in zip(ents, gammas)]
        return payload, {"gammas": gammas, "context_lens": [len(e["context"]) for e in ents]}

    # --- provenance (written before any measurement)
    meta = {
        "provenance": collect_provenance(REPO_ROOT, cfg),
        "verifier_semantics": verifier_semantics_record(target, cfg.get("verification_batching", {}), cfg.get("controller", {})),
        "config_path": str(a.config), "config_sha256": config_sha256(cfg),
        "target_runtime": {k: m.get(k) for k in ("target", "target_dtype", "target_enable_prefix_caching",
                                                 "target_enforce_eager", "target_gpu_memory_utilization",
                                                 "target_max_model_len")},
        "seed": a.seed, "context_quantiles": CONTEXT_QUANTILES, "expected_counts": counts,
        "pool_sha256": pool_hash, "prep_seconds": prep_s,
        "pool_lengths": {str(L): [len(e["context"]) for e in pools[L]] for L in pools},
        "val_pool_lengths": [len(e["context"]) for e in val_pool],
        "job_start_unix": time.time(),
    }
    (out / "calibration_meta.json").write_text(json.dumps(meta, indent=1, default=str))

    fh = open(raw, "a", buffering=1)

    def sink(rec):
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    # --- engine warm-up (untimed, counted)
    if "engine_warm" not in done:
        for k in range(ENGINE_WARM_CALLS):
            payload, _ = make_payload({"phase": "warmup", "Lq": "p50", "B": 1 + (k % 4), "gamma": k % 5})
            target.verify_batch_sync(payload)
        sink({"item_id": "engine_warm", "phase": "engine_warm", "calls": ENGINE_WARM_CALLS})

    n, early = run_schedule(sched["items"], make_payload, target.verify_batch_sync, lambda: last["ms"], sink, done,
                            deadline_s=a.deadline_min * 60.0)
    end = {"item_id": "END", "phase": "end", "calls_this_run": n, "stopped_early": early, "t_unix": time.time()}
    sink(end)
    fh.close()
    print(json.dumps(end))
    return 3 if early else 0


if __name__ == "__main__":
    raise SystemExit(main())
