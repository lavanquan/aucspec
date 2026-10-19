"""Hard budget guard for the RQ1 pilot (A3).  The total cap is a LIMIT, not a forecast.

ledger JSON: {"cap_node_hours": 3.2, "authorized_jobs": 1, "jobs": [{"job_id", "seed", "elapsed_s", "node_hours"}, ...]}
  check  <ledger> <job_limit_minutes>   exit 0 only if (a) fewer jobs than `authorized_jobs` were submitted so far and
                                        (b) actual node-hours so far + this job's MAXIMUM (its walltime limit, 1 node) <= cap
  record <ledger> <job_id> <seed> <elapsed_seconds>   append the ACTUAL consumption (from sacct) of a finished job
"""
import json
import sys
from pathlib import Path


def spent(ledger: dict) -> float:
    return sum(float(j["node_hours"]) for j in ledger["jobs"])


def check(ledger: dict, job_limit_minutes: float) -> tuple[bool, str]:
    cap = float(ledger["cap_node_hours"])
    if len(ledger["jobs"]) >= int(ledger["authorized_jobs"]):
        return False, (f"REFUSED: {len(ledger['jobs'])} job(s) already submitted, only {ledger['authorized_jobs']} authorized "
                       "(further seed jobs need explicit review)")
    need = job_limit_minutes / 60.0
    if spent(ledger) + need > cap + 1e-12:
        return False, f"REFUSED: spent {spent(ledger):.3f} + max {need:.3f} node-h > cap {cap}"
    return True, f"OK: spent {spent(ledger):.3f} + max {need:.3f} <= cap {cap}"


def record(ledger: dict, job_id: str, seed: int, elapsed_s: float) -> dict:
    ledger["jobs"].append({"job_id": str(job_id), "seed": int(seed), "elapsed_s": float(elapsed_s),
                           "node_hours": float(elapsed_s) / 3600.0})
    return ledger


if __name__ == "__main__":
    cmd, path = sys.argv[1], Path(sys.argv[2])
    led = json.loads(path.read_text())
    if cmd == "check":
        ok, msg = check(led, float(sys.argv[3]))
        print(msg)
        sys.exit(0 if ok else 1)
    if cmd == "record":
        path.write_text(json.dumps(record(led, sys.argv[3], int(sys.argv[4]), float(sys.argv[5])), indent=1))
        print("recorded; total node-h =", round(spent(json.loads(path.read_text())), 4))
