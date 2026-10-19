"""One-job driver: preflight -> HF fp32 reference -> qualification -> (only if gates pass) canonical-launcher wiring smoke."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exclude-frozen", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke-run-tag", required=True)
    ap.add_argument("--smoke-n", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = {"steps": []}

    def step(name, cmd, timeout):
        t = time.time()
        with open(out / f"log_{name}.txt", "w") as fh:
            try:
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                rc = -9
        log["steps"].append({"step": name, "rc": rc, "wall_s": time.time() - t, "elapsed_s": time.time() - t0})
        (out / "orchestration.json").write_text(json.dumps(log, indent=1))
        return rc

    py = sys.executable
    if step("preflight", [py, str(HERE / "preflight_runtime.py"), "--out", str(out / "preflight.json")], 300) != 0:
        return 65
    frozen = out / "frozen_qualification_contexts.json"
    if step("prepare", [py, str(HERE / "rocm_qual_prepare.py"), "--config", a.config, "--exclude-frozen", a.exclude_frozen,
                        "--out", str(frozen)], 12 * 60) != 0:
        return 2
    res = out / "result_qualification.json"
    rc_q = step("qualification", [py, str(HERE / "rocm_qual_run.py"), "--config", a.config, "--frozen", str(frozen), "--out", str(res)],
                14 * 60)
    gate = None
    if res.exists():
        r = json.loads(res.read_text())
        gate = bool(r.get("status") == "ok" and (r.get("dev") or {}).get("pass") and (r.get("heldout") or {}).get("pass"))
    log["qualification_gate_pass"] = gate
    rc_s = None
    if gate:
        rc_s = step("integration_smoke", [py, str(HERE / "canonical_launch.py"), "run", "--run-tag", a.smoke_run_tag, "--n",
                                          str(a.smoke_n), "--seed", "1"], max(120, 33 * 60 - (time.time() - t0)))
    else:
        log["steps"].append({"step": "integration_smoke", "skipped": "qualification gate not passed"})
    log["total_wall_s"] = time.time() - t0
    (out / "orchestration.json").write_text(json.dumps(log, indent=1))
    print(json.dumps({"gate": gate, "qual_rc": rc_q, "smoke_rc": rc_s}))
    return 0 if rc_q == 0 and (rc_s in (None, 0)) else 1


if __name__ == "__main__":
    sys.exit(main())
