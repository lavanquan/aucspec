"""Derive the exploratory RQ1 extension config from the (immutable) pilot config.  Only the prompt catalog / population size change.
usage: make_rq1_extension_config.py <prompt_catalogs.json> [out.yaml]"""
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "configs" / "rq1_pilot_fixed_gamma2_tritonfa33.yaml"
OUT = ROOT / "configs" / "rq1_extension_fixed_gamma2_tritonfa33.yaml"
N_ORDER = [12, 8, 16, 24, 32]


def build(catalogs: dict) -> dict:
    cfg = copy.deepcopy(yaml.safe_load(PILOT.read_text(encoding="utf-8")))
    cfg["dataset"]["split"] = "train"
    cfg["dataset"]["num_questions"] = 32 * 64
    cfg["simulation"]["population_n_max"] = 32
    cfg["simulation"]["questions_per_client"] = 64
    cfg["simulation"]["expected_catalog_hash"] = catalogs["extension_catalog"]["catalog_hash"]
    return cfg


if __name__ == "__main__":
    cat = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    out.write_text("# EXPLORATORY RQ1 saturation-extension config (C1.2), derived from the immutable pilot config; GSM8K train catalog, n_max 32.\n"
                   + yaml.safe_dump(build(cat), sort_keys=True), encoding="utf-8")
    print("wrote", out)
