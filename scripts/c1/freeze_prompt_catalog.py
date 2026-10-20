"""Freeze the prompt catalogs (CPU; needs the `datasets` cache).  usage: freeze_prompt_catalog.py <out_dir>
Writes catalog records for (a) the ORIGINAL pilot catalog (GSM8K test, n_max 16 x 64) and (b) the EXTENSION catalog (GSM8K train, n_max 32 x 64),
plus their overlap, so the incompatibility is documented rather than assumed."""
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legacy" / "src"))
from edge_specsim.dataset import load_samples  # noqa: E402
from edge_specsim.prompt_catalog import catalog_record  # noqa: E402


def load(split: str, n: int, base: dict):
    cfg = copy.deepcopy(base["dataset"])
    cfg.update({"split": split, "num_questions": n})
    return list(load_samples(cfg))


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((ROOT / "configs" / "rq1_pilot_fixed_gamma2_tritonfa33.yaml").read_text(encoding="utf-8"))
    old = load("test", 16 * 64, base)
    new = load("train", 32 * 64, base)
    rec_old, rec_new = catalog_record(old, 16, 64), catalog_record(new, 32, 64)
    ov = len({s.prompt for s in old[:1024]} & {s.prompt for s in new})
    res = {"original_pilot_catalog": {"split": "test", **rec_old}, "extension_catalog": {"split": "train", **rec_new},
           "prompt_overlap_between_catalogs": ov,
           "compatibility": "DIFFERENT populations of prompts (test vs train split); static hardware profiles identical; the original "
                            "N=8..16 seed-1 results are a separate diagnostic and are NOT directly comparable with the extension"}
    (out / "prompt_catalogs.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ("split", "catalog_hash", "n_slots", "duplicates")}) for k, v in res.items()}, indent=1))
