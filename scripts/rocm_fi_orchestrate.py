"""Sequential fault isolation driver (runs inside the single Slurm job).

prepare (HF fp32 reference, 10 forwards) -> A -> B -> C -> D -> E -> F, each vLLM configuration in a FRESH process.
Choices of the base for C/D/E/F are adaptive (best dev score so far), never a Cartesian product.
Hard limits: 400 target-model calls in total, 6 vLLM configurations, wall-clock deadline.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CALL_CAP, MAX_CONFIGS = 400, 6
DEADLINE_LAUNCH_S = 34 * 60
PER_CONFIG_TIMEOUT_S = 15 * 60
CLEAN_ENV_KEYS = ("VLLM_USE_TRITON_FLASH_ATTN", "VLLM_USE_V1", "VLLM_ATTENTION_BACKEND", "VLLM_ROCM_CUSTOM_PAGED_ATTN",
                  "VLLM_ENABLE_V1_MULTIPROCESSING")


def score(res: dict) -> float:
    if res.get("status") != "ok":
        return -1.0
    g = res["dev"]["gate_inputs"]
    vals = [g.get("B16_ident_vs_fp32_gap>=0.5"), g.get("B4_ident_vs_fp32_gap>=0.5")]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    used = 0
    log = {"events": [], "configs": {}}

    def sh(cmd, env, logfile, timeout):
        with open(logfile, "w") as fh:
            p = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode

    base_env = {k: v for k, v in os.environ.items() if k not in CLEAN_ENV_KEYS}
    frozen = out / "frozen_contexts.json"
    rc = sh([sys.executable, str(HERE / "rocm_fi_prepare.py"), "--config", a.config, "--out", str(frozen)],
            base_env, out / "prepare.log", 20 * 60)
    log["events"].append({"step": "prepare", "rc": rc, "elapsed_s": time.time() - t_start})
    if rc != 0 or not frozen.exists():
        (out / "orchestration.json").write_text(json.dumps(log, indent=1))
        return 2
    used += 10

    results: dict[str, dict] = {}

    def run(name, env_extra, dtype="auto", eager=False):
        nonlocal used
        if len(results) >= MAX_CONFIGS or time.time() - t_start > DEADLINE_LAUNCH_S or used >= CALL_CAP - 46:
            log["events"].append({"step": name, "skipped": "budget/deadline", "used_calls": used})
            return None
        env = dict(base_env)
        env.update(env_extra)
        cmd = [sys.executable, str(HERE / "rocm_fi_run_config.py"), "--name", name, "--frozen", str(frozen), "--config", a.config,
               "--out", str(out / f"result_{name}.json"), "--dtype", dtype, "--call-budget", str(min(80, CALL_CAP - used))]
        if eager:
            cmd.append("--eager")
        t = time.time()
        try:
            rc = sh(cmd, env, out / f"log_{name}.txt",
                    max(60, min(PER_CONFIG_TIMEOUT_S, 43 * 60 - (time.time() - t_start))))
        except subprocess.TimeoutExpired:
            rc = -9
        resf = out / f"result_{name}.json"
        res = json.loads(resf.read_text()) if resf.exists() else {"status": "no_result", "calls": 0}
        res["rc"], res["wall_s"] = rc, time.time() - t
        res["spec"] = {"env": env_extra, "dtype": dtype, "eager": eager}
        used += int(res.get("calls", 0))
        results[name] = res
        log["configs"][name] = {"status": res["status"], "rc": rc, "calls": res.get("calls"), "wall_s": res["wall_s"],
                                "score": score(res), "spec": res["spec"], "error": res.get("error")}
        log["events"].append({"step": name, "elapsed_s": time.time() - t_start, "used_calls": used})
        (out / "orchestration.json").write_text(json.dumps(log, indent=1))
        return res

    def best():
        ok = {k: v for k, v in results.items() if v.get("status") == "ok"}
        if not ok:
            return None
        return max(ok.items(), key=lambda kv: (score(kv[1]), kv[0] != "A_triton0"))

    run("A_triton0", {"VLLM_USE_TRITON_FLASH_ATTN": "0"})
    run("B_triton1", {"VLLM_USE_TRITON_FLASH_ATTN": "1"})
    b = best()
    if b:
        name, res = b
        be = dict(res["spec"]["env"])
        run("C_eager", be, eager=True)
        run("D_no_custom_paged_attn", {**be, "VLLM_ROCM_CUSTOM_PAGED_ATTN": "0"})
        b = best()
        be = dict(b[1]["spec"]["env"]) if b else be
        run("E_fp16", be, dtype="float16")
        run("F_v1_engine", {**be, "VLLM_USE_V1": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
    log["total_calls_used"] = used
    log["elapsed_s"] = time.time() - t_start
    log["not_tested"] = {
        "VLLM_ATTENTION_BACKEND alternatives (TORCH_SDPA, XFORMERS, FLASH_ATTN)":
            "vLLM 0.9.1 rocm.py maps every non-ROCM_FLASH request to ROCmFlashAttention with only an info line, "
            "so these would silently fall back to the same kernel; not tested as separate configurations",
        "AITER backends": "not installed/enabled in this build (VLLM_ROCM_USE_AITER=False)"}
    (out / "orchestration.json").write_text(json.dumps(log, indent=1))
    print(json.dumps({"configs": {k: (v["status"], round(score(v), 3)) for k, v in results.items()}, "calls": used}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
