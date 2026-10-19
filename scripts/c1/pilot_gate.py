"""Abort gate for the RQ1 pilot: evaluated after the FIRST run (N=8) of the first job, and cheaply after every run.
Exit 0 = continue, 3 = stop the job.  CPU only, no GPU."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "c1"))
import analyze_rq1_pilot as an  # noqa: E402

MAX_REAL_OVER_VIRTUAL = 3.0


def gate(csv: Path, n: int, ledger: Path | None) -> tuple[bool, list[str]]:
    df = pd.read_csv(csv, low_memory=False)
    m = an.run_metrics(df, n)
    reasons = []
    if m["invalid"]:
        reasons.append(f"INVALID: {m['invalid_reason']}")
    if ledger is not None and ledger.exists():
        chk = json.loads(ledger.read_text())
        if chk["n_violations"]:
            reasons.append(f"{chk['n_violations']} causality violation(s)")
    elif ledger is not None:
        reasons.append("causality_check.json missing")
    if not (m["real_over_virtual"] <= MAX_REAL_OVER_VIRTUAL):
        reasons.append(f"real/virtual time ratio {m['real_over_virtual']:.2f} > {MAX_REAL_OVER_VIRTUAL}")
    if m["rounds_per_client_min"] < 20:
        reasons.append(f"fewer than 20 in-window rounds for some client ({m['rounds_per_client_min']})")
    return (not reasons), reasons


if __name__ == "__main__":
    csv, n = Path(sys.argv[1]), int(sys.argv[2])
    ledger = Path(str(csv).replace(".csv", "_artifacts")) / "causality_check.json"
    ok, why = gate(csv, n, ledger)
    print(json.dumps({"csv": str(csv), "n": n, "ok": ok, "reasons": why}))
    sys.exit(0 if ok else 3)
