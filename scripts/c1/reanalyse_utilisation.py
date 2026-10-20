"""Read-only re-analysis of verifier utilisation for existing raw traces (C1.4).  usage: reanalyse_utilisation.py <scratch_root> <out.json> <tag> [<tag> ...]
Reports the ORIGINAL accounting (full wall of batches reached through in-window receives / window) next to the CORRECTED exact occupancy
(overlap of every unique service interval with the window), plus serialisation status and, as an INFERENCE only, how many batches carry more
processed tokens than the engine's max_num_batched_tokens (such a batch would need more than one scheduler step; the number of engine steps
was not logged).  Raw files are never modified."""
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import analyze_rq1_pilot as an  # noqa: E402


def main():
    root, out, tags = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    rows = []
    for tag in tags:
        for csv in sorted(glob.glob(str(root / tag / "raw" / "fixed_gamma_n*_x3.0_s*.csv"))):
            m = re.search(r"_n(\d+)_x3\.0_s(\d+)\.csv$", csv)
            n, seed = int(m.group(1)), int(m.group(2))
            df = pd.read_csv(csv, low_memory=False)
            met = an.run_metrics(df, n)
            start, end = float(df["experiment_warmup_end_ms"].iloc[0]), float(df["experiment_measurement_end_ms"].iloc[0])
            occ = an.exact_occupancy(df, start, end)
            b = df.drop_duplicates("verification_batch_id")
            rc = Path(csv.replace(".csv", "_artifacts")) / "run_config.json"
            budget = None
            if rc.exists():
                budget = json.loads(rc.read_text()).get("runtime_identity", {}).get("live_engine", {}).get("max_num_batched_tokens")
            over = int((b["verification_batch_processed_tokens"] > budget).sum()) if budget else None
            rows.append({"tag": tag, "N": n, "seed": seed, "utilisation_original": met["verifier_utilisation_legacy"],
                         "utilisation_corrected": occ["utilisation"], "difference": occ["utilisation"] - met["verifier_utilisation_legacy"],
                         "serialisation_ok": occ["serialisation_ok"], "batches_overlapping_window": occ["n_batches_overlapping_window"],
                         "max_num_batched_tokens": budget, "batches_with_processed_tokens_over_step_budget": over,
                         "processed_tokens_max": int(b["verification_batch_processed_tokens"].max()),
                         "max_batch_size_seen": int(b["verification_batch_size"].max())})
    out.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['tag'][:18]:18s} N={r['N']:>2d} orig {r['utilisation_original']:.3f} corrected {r['utilisation_corrected']:.3f} "
              f"d={r['difference']:+.3f} serial={r['serialisation_ok']} over_budget={r['batches_with_processed_tokens_over_step_budget']} "
              f"max_tok={r['processed_tokens_max']}")


if __name__ == "__main__":
    main()
