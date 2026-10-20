"""Derive the NEW exploratory serving configuration `batch_cap_16` from the batch_cap_32 extension config (C1.3).
Serving change: verification_batching.max_batch_size 32 -> 16, nothing else.  Plus one OBSERVATION-ONLY flag,
simulation.failure_diagnostics (writes metadata only if a run raises; no per-batch I/O, no change to verifier behaviour).
usage: make_rq1_batchcap16_config.py [out.yaml]"""
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "configs" / "rq1_extension_fixed_gamma2_tritonfa33.yaml"
OUT = ROOT / "configs" / "rq1_batchcap16_fixed_gamma2_tritonfa33.yaml"
N_ORDER = [8, 16, 24, 32]        # N=8 is a diagnostic burn-in run


def build() -> dict:
    cfg = copy.deepcopy(yaml.safe_load(SRC.read_text(encoding="utf-8")))
    cfg["verification_batching"]["max_batch_size"] = 16
    cfg["simulation"]["failure_diagnostics"] = True
    return cfg


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    out.write_text("# NEW serving configuration `batch_cap_16` (C1.3, exploratory). Same correctness-qualified verifier backend (runtime H) as batch_cap_32;\n"
                   "# only verification_batching.max_batch_size differs (32 -> 16). batch_cap_32 data (incl. the N=24 HIP OOM) are preserved unchanged.\n"
                   + yaml.safe_dump(build(), sort_keys=True), encoding="utf-8")
    print("wrote", out)
