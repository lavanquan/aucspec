"""Expanded correctness qualification of runtime "V1-batchwall-tritonfa33" with the CANONICAL verifier configuration.

Engine = VLLMCandidateVerifier exactly as the simulator builds it (max_logprobs=1, V1 tie-break, prefix caching off).
All verification goes through the unmodified `verify_batch_sync`. The runtime guard proves Triton FA in all layers.
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

import rocm_qual_lib as q  # noqa: E402

CALL_CAP = 400


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"status": "started", "pid": os.getpid(), "calls": 0, "latency_ms": {}}
    out = Path(a.out)
    try:
        import yaml
        import canonical_guard as guard
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from edge_specsim import runtime_guard as rg
        from edge_specsim.models import DraftResult
        from edge_specsim.provenance import collect_provenance
        from edge_specsim.target_client import VLLMCandidateVerifier

        cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
        guard.assert_clean_source()
        guard.validate_canonical_config(cfg)
        guard.assert_runtime_env()
        res["provenance"] = collect_provenance(REPO_ROOT, cfg)
        frozen = json.load(open(a.frozen))
        m, sim = cfg["models"], cfg["simulation"]
        t0 = time.time()
        ver = VLLMCandidateVerifier(                      # canonical constructor arguments, max_logprobs left at its default (1)
            model_name=m["target"], gpu_memory_utilization=float(m.get("target_gpu_memory_utilization", 0.88)),
            max_model_len=int(m.get("target_max_model_len", 8192)), dtype=str(m.get("target_dtype", "auto")),
            enforce_eager=bool(m.get("target_enforce_eager", False)),
            enable_prefix_caching=bool(m.get("target_enable_prefix_caching", True)),
            sampling_temperature=0.0, seed=int(sim["seed"]))
        res["init_s"] = time.time() - t0
        evidence = rg.enforce_runtime(ver, cfg["runtime"], REPO_ROOT)
        res["runtime_evidence"] = rg.serialisable(evidence)
        llm = ver.llm
        sched = llm.llm_engine.scheduler[0]
        steps: list[tuple[int, int]] = []
        orig_schedule = sched.schedule

        def schedule():
            r = orig_schedule()
            so = r[1]
            steps.append((int(getattr(so, "num_prefill_groups", -1)), int(getattr(so, "num_batched_tokens", -1))))
            return r

        sched.schedule = schedule
        orig_generate = llm.generate
        lat: dict[str, list[float]] = {}

        def tick(kind, fn, *args, **kw):
            if res["calls"] >= CALL_CAP:
                raise RuntimeError("call budget exhausted")
            res["calls"] += 1
            n0 = len(steps)
            t = time.perf_counter()
            r = fn(*args, **kw)
            lat.setdefault(kind, []).append((time.perf_counter() - t) * 1000.0)
            return r, steps[n0:]

        sp_dec = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4, logprobs=1, detokenize=False)

        def draft(tokens):
            return DraftResult(client_id=0, token_ids=list(tokens), draft_logprobs=[], text="", latency_ms=0.0, worker_id=-1,
                               distribution_payload="delta_proposal")

        def run_item(S, item):
            kind, B, g = item["kind"], item["B"], item["gamma"]
            rec = {}
            calls_meta = []
            for s in range(0, len(item["ctx"]), B):
                grp = item["ctx"][s: s + B]
                if kind == "decode":
                    outs, st = tick(f"decode_B{B}", orig_generate, [TokensPrompt(prompt_token_ids=list(S[i]["context"])) for i in grp],
                                    sp_dec, use_tqdm=False)
                    for pos, (i, o) in enumerate(zip(grp, outs)):
                        rec[i] = {"slot": s + pos, "group_pos": pos, "tokens": [int(t) for t in o.outputs[0].token_ids]}
                else:
                    payload, cpos = [], {}
                    for i in grp:
                        props = S[i]["proposal"][:g]
                        if kind == "corrupt":
                            props, cpos[i] = q.corrupt_proposal(S[i]["fp32"], S[i]["proposal"], g, i)
                        payload.append((S[i]["context"], draft(props), 96))
                    results, st = tick(f"verify_B{B}", ver.verify_batch_sync, payload)
                    for pos, (i, r_) in enumerate(zip(grp, results)):
                        tl = []
                        for j, d in enumerate(r_.target_logprobs):
                            mx = max(d.values())
                            tok = payload[pos][1].token_ids[j] if j < len(payload[pos][1].token_ids) else None
                            tl.append([mx, d.get(tok) if tok is not None else None])
                        rec[i] = {"slot": s + pos, "group_pos": pos, "accepted": int(r_.accepted_length),
                                  "committed": [int(t) for t in r_.committed_token_ids], "correction": r_.correction_token_id,
                                  "bonus": r_.bonus_token_id, "target_greedy": [int(t) for t in r_.target_greedy_token_ids],
                                  "reached_eos": bool(r_.reached_eos), "lp_max_and_proposed": tl,
                                  "corrupt_pos": cpos.get(i), "payload": list(payload[pos][1].token_ids)}
                calls_meta.append({"n_seq": len(grp), "steps": st})
            return {"item": {k: v for k, v in item.items() if k != "ctx"}, "ctx_order": item["ctx"], "records": rec,
                    "calls": calls_meta}

        # warm-up (counted)
        dev = frozen["sets"]["dev"]["contexts"]
        tick("warmup", orig_generate, [TokensPrompt(prompt_token_ids=dev[0]["context"])], sp_dec, use_tqdm=False)
        tick("warmup", orig_generate, [TokensPrompt(prompt_token_ids=c["context"]) for c in dev[:16]], sp_dec, use_tqdm=False)

        def margin_rate(S, run):
            ok = tot = 0
            for i, r_ in run["records"].items():
                g = run["item"]["gamma"]
                for j in range(min(g, len(r_["target_greedy"]))):
                    gap = S[i]["fp32"][j]["gap"]
                    if gap >= q.GAP_OK:
                        tot += 1
                        ok += r_["target_greedy"][j] == S[i]["fp32"][j]["top_ids"][0]
            return ok, tot

        def run_level(setname, level):
            S = frozen["sets"][setname]["contexts"]
            plan = q.build_plan(len(S), level)
            need = q.n_calls(plan)
            if res["calls"] + need > CALL_CAP:
                raise RuntimeError(f"plan {setname}/{level} needs {need} calls; {CALL_CAP - res['calls']} left")
            R = {}
            for item in plan:
                R[q.key(item)] = run_item(S, item)
            gate = {}
            for k, run in R.items():
                if run["item"]["kind"] == "true" and run["item"]["gamma"] >= 1 and run["item"]["B"] >= 2:
                    ok, tot = margin_rate(S, run)
                    gate[k] = [ok, tot]
            return R, gate, need

        res["dev"] = {}
        R, gate, need = run_level("dev", "dev")
        res["dev"] = {"runs": R, "gate": gate, "planned_calls": need,
                      "pass": all(q.passes(ok, tot) for ok, tot in gate.values())}
        if res["dev"]["pass"]:
            try:
                R2, gate2, need2 = run_level("heldout", "heldout")
                res["heldout"] = {"runs": R2, "gate": gate2, "planned_calls": need2,
                                  "pass": all(q.passes(ok, tot) for ok, tot in gate2.values())}
            except RuntimeError as exc:
                res["heldout_error"] = repr(exc)
        res["latency_ms"] = {k: {"n": len(v), "mean": sum(v) / len(v), "max": max(v)} for k, v in lat.items()}
        res["runtime_evidence_end"] = rg.serialisable(evidence)       # Triton-FA call counter at the end
        res["scheduler_steps_total"] = len(steps)
        res["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        res["status"] = "failed"
        res["error"] = repr(exc)[:2000]
        res["traceback"] = traceback.format_exc()[-3000:]
    out.write_text(json.dumps(res))
    print(json.dumps({"status": res["status"], "calls": res["calls"], "error": res.get("error")}))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
