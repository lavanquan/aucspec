"""One-job driver, FAIL-CLOSED: preflight -> frozen-input integrity -> qualification -> (only if every gate passes) wiring smoke.

Exit codes: 0 only when qualification (dev AND held-out) AND the integration smoke both succeed.
  65 preflight failed | 3 frozen-input integrity failed | 10 qualification process failed or output missing
  11 correctness gate failed / incomplete / zero observations | 12 integration smoke or its checks failed
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rocm_qual_lib as q  # noqa: E402
import verify_frozen_inputs as vfi  # noqa: E402


def default_runner(name, cmd, timeout, logfile):
    with open(logfile, "w") as fh:
        try:
            return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return -9


def run_pipeline(a, runner=default_runner, py=None, clock=time.time) -> int:
    py = py or sys.executable
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = clock()
    log = {"steps": [], "verdict": None}

    def save():
        (out / "orchestration.json").write_text(json.dumps(log, indent=1))

    def step(name, cmd, timeout):
        t = clock()
        rc = runner(name, cmd, timeout, out / f"log_{name}.txt")
        log["steps"].append({"step": name, "rc": rc, "wall_s": clock() - t, "elapsed_s": clock() - t0})
        save()
        return rc

    def finish(code, why):
        log["exit_code"], log["why"], log["total_wall_s"] = code, why, clock() - t0
        save()
        return code

    if step("preflight", [py, str(HERE / "preflight_runtime.py"), "--out", str(out / "preflight.json")], 300) != 0:
        return finish(65, "preflight failed")
    frozen = out / "frozen_qualification_contexts.json"
    if not a.reuse_frozen:
        return finish(3, "this retry must reuse the frozen contexts (--reuse-frozen)")
    shutil.copyfile(a.reuse_frozen, frozen)
    integ = vfi.check_frozen(frozen, a.expect_frozen_sha256)
    log["frozen_integrity"] = integ
    if not integ["ok"]:
        return finish(3, f"frozen input integrity failed: {integ['problems'][:3]}")
    res_path = out / "result_qualification.json"
    rc_q = step("qualification", [py, str(HERE / "rocm_qual_run.py"), "--config", a.config, "--frozen", str(frozen), "--out", str(res_path)], 14 * 60)
    res = json.loads(res_path.read_text()) if res_path.exists() else None
    if rc_q != 0 or res is None:
        return finish(10, f"qualification process failed (rc={rc_q}) or output missing")
    verdict = q.qualification_verdict(res, json.loads(frozen.read_text()))
    log["verdict"] = verdict
    save()
    if not verdict["ok"]:
        log["steps"].append({"step": "integration_smoke", "skipped": "qualification gate not passed"})
        return finish(11, "; ".join(verdict["reasons"])[:600])
    if step("integration_smoke", [py, str(HERE / "canonical_launch.py"), "run", "--run-tag", a.smoke_run_tag, "--n", str(a.smoke_n), "--seed", "1"],
            max(120, 33 * 60 - (clock() - t0))) != 0:
        return finish(12, "integration smoke process failed")
    run_dir = HERE.parent / "results" / "capacity_frontier" / a.smoke_run_tag
    rc_c = step("integration_checks", [py, str(HERE / "integration_checks.py"), "--run-dir", str(run_dir), "--n", str(a.smoke_n),
                                       "--expected-sha", a.expected_sha or "", "--out", str(out / "integration_checks.json")], 300)
    if rc_c != 0:
        return finish(12, "integration smoke checks failed")
    return finish(0, "qualification and integration smoke passed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exclude-frozen", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke-run-tag", required=True)
    ap.add_argument("--smoke-n", type=int, default=8)
    ap.add_argument("--reuse-frozen", default=None)
    ap.add_argument("--expect-frozen-sha256", default=None)
    ap.add_argument("--expected-sha", default=None)
    a = ap.parse_args()
    code = run_pipeline(a)
    print(json.dumps({"exit": code}))
    return code


if __name__ == "__main__":
    sys.exit(main())
