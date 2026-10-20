"""C1.4 final bounded probe config: the batch_cap_16 config plus ONE observation-only flag, simulation.memory_summary (snapshots at engine
init / run start / run end / failure; no per-batch sampling, no change to inference).  usage: make_rq1_cap16_probe_config.py [out.yaml]"""
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "configs" / "rq1_batchcap16_fixed_gamma2_tritonfa33.yaml"
OUT = ROOT / "configs" / "rq1_cap16_probe_fixed_gamma2_tritonfa33.yaml"
N_ORDER = [28, 32]


def build() -> dict:
    cfg = copy.deepcopy(yaml.safe_load(SRC.read_text(encoding="utf-8")))
    cfg["simulation"]["memory_summary"] = True
    return cfg


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    out.write_text("# C1.4 final bounded saturation probe (N=28, 32) on the batch_cap_16 serving configuration; EXPLORATORY protocol (see scripts/c1/pilot_gate.py).\n"
                   "# Differs from the C1.3 batch_cap_16 config only by the observation-only flag simulation.memory_summary.\n"
                   + yaml.safe_dump(build(), sort_keys=True), encoding="utf-8")
    print("wrote", out)
