"""Collect compact smoke outputs (no raw traces) into <out_dir>: provenance table, context quantiles."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
prov = {}
for tag in sys.argv[3:]:
    for art in sorted((R / tag / "raw").glob("*_artifacts")):
        cfg = json.loads((art / "run_config.json").read_text())
        p, v = cfg["provenance"], cfg["verifier_semantics"]
        sm = json.loads((art / "summary.json").read_text()) if (art / "summary.json").exists() else {}
        prov[f"{tag}/{art.name}"] = {
            "git_sha": p["git_sha"], "git_dirty": p["git_dirty"],
            "slurm_job_id": p["node"]["slurm"]["SLURM_JOB_ID"], "node": p["node"]["slurm"]["SLURMD_NODENAME"],
            "gpus": p["node"]["gpu"]["device_names"][:1] + [f"x{p['node']['gpu']['device_count']}"],
            "hip": p["node"]["gpu"]["hip_version"], "rocm": p["node"]["rocm_version_file"],
            "campaign_config_hash": p["campaign_config_hash"], "resolved_config_sha256": p["resolved_config_sha256"],
            "population_fingerprint": p["population_fingerprint"], "verifier_semantics": v,
            "software_versions": cfg.get("software_versions"),
        }
(out / "provenance.json").write_text(json.dumps(prov, indent=2))

# per-request context lengths from the N=1 batch-wall run (B=1 batches: context_sum == one request's context)
csv = next((R / sys.argv[3] / "raw").glob("capacity_dpp_n1_x3.0_s1.csv"))
d = pd.read_csv(csv, usecols=["verification_batch_size", "verification_batch_context_tokens_sum"])
ctx = d[d["verification_batch_size"] == 1]["verification_batch_context_tokens_sum"].to_numpy()
q = {f"p{k}": float(np.percentile(ctx, k)) for k in (0, 10, 50, 90, 100)}
(out / "context_length_quantiles.json").write_text(json.dumps({"n": int(len(ctx)), **q}, indent=2))
print(json.dumps(q), len(ctx))
