"""Single-job driver for candidates G and H (fresh process each). Budget: 200 target-model calls, ~20 min wall."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAP = 200
T_LIMIT = 18 * 60


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    used = 0
    log = {"events": []}
    base = {k: v for k, v in os.environ.items() if k not in ("VLLM_USE_TRITON_FLASH_ATTN", "VLLM_USE_V1", "VLLM_ATTENTION_BACKEND")}

    def run(name, mode, env_extra, budget, **kw):
        nonlocal used
        env = dict(base)
        env.update(env_extra)
        cmd = [sys.executable, str(HERE / "rocm_fi_validate.py"), "--name", name, "--mode", mode, "--frozen", a.frozen,
               "--config", a.config, "--out", str(out / f"result_{name}.json"), "--call-budget", str(budget)]
        for k, v in kw.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]
        tmo = max(60, min(9 * 60, T_LIMIT - (time.time() - t0)))
        t = time.time()
        try:
            with open(out / f"log_{name}.txt", "w") as fh:
                rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=tmo).returncode
        except subprocess.TimeoutExpired:
            rc = -9
        f = out / f"result_{name}.json"
        r = json.loads(f.read_text()) if f.exists() else {"status": "no_result", "calls": 0}
        used += int(r.get("calls", 0))
        log["events"].append({"name": name, "rc": rc, "status": r["status"], "calls": r.get("calls"), "wall_s": time.time() - t,
                              "error": r.get("error"), "dev_pass": (r.get("dev") or {}).get("pass"),
                              "heldout_pass": (r.get("heldout") or {}).get("pass"), "elapsed_s": time.time() - t0})
        (out / "orchestration.json").write_text(json.dumps(log, indent=1))
        return r

    # G: the naive-SDPA runtime fix on the current environment
    run("G_runtime_patch", "G", {"VLLM_USE_TRITON_FLASH_ATTN": "0"}, 130)
    # H: torch-compatible Triton (3.3.1 shipped with torch) via a symlink shim ahead of the user-site Triton 3.8.0
    shim = out / "triton_shim"
    shim.mkdir(exist_ok=True)
    target = Path("/usr/local/lib/python3.12/dist-packages/triton")
    link = shim / "triton"
    if target.exists() and not link.exists():
        link.symlink_to(target, target_is_directory=True)
    if not target.exists():
        log["events"].append({"name": "H_triton331", "status": "unexecuted", "error": f"{target} not found"})
    elif time.time() - t0 < T_LIMIT - 120 and CAP - used >= 48:
        pp = f"{shim}{os.pathsep}{base.get('PYTHONPATH', '')}".rstrip(os.pathsep)
        run("H_triton331", "H", {"VLLM_USE_TRITON_FLASH_ATTN": "1", "PYTHONPATH": pp}, CAP - used,
            dev_level="dev_light", heldout_level="heldout_light")
    else:
        log["events"].append({"name": "H_triton331", "status": "skipped", "error": "budget/time"})
    log["total_calls_used"] = used
    log["elapsed_s"] = time.time() - t0
    (out / "orchestration.json").write_text(json.dumps(log, indent=1))
    print(json.dumps({"calls": used, "events": [(e["name"], e["status"]) for e in log["events"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
