"""C2 Phase A: compact inventory of the RQ1 evidence (C1.1-C1.4) from the committed compact results.  CPU only; nothing is re-run and no old
verdict is changed.  Writes results/c2/rq1_evidence_inventory.json and prints a markdown table."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"

CAMPAIGNS = [
    {"id": "C1.1-pilot", "tag": "RQ1_PILOT_4a7e866", "job": "51179457", "code_sha": "4a7e866", "catalog": "GSM8K-test (1,024 slots, hash 628857be)",
     "serving": "batch_cap_32, FCFS", "protocol": "original (real/virtual <= 3 gate)", "file": R / "rq1_pilot_seed1" / "pilot_seed1_compact.json",
     "note": "N=8 was the first run of the job: 45 slow batches (>600 ms), x_min depressed; unexplained"},
    {"id": "C1.2-extension", "tag": "RQ1_EXT_28907a6", "job": "51186048", "code_sha": "28907a6", "catalog": "GSM8K-train (2,048 slots, hash 02e79304)",
     "serving": "batch_cap_32, FCFS", "protocol": "original", "file": R / "rq1_extension" / "ext_seed1_compact.json",
     "note": "N=24 ABORTED with HIP OOM (cap-32 batch of 24); N=32 never reached; N=24 data lost except console log"},
    {"id": "C1.3-cap16", "tag": "RQ1_CAP16_5a37f03", "job": "51226862", "code_sha": "5a37f03", "catalog": "GSM8K-train (hash 02e79304)",
     "serving": "batch_cap_16, FCFS", "protocol": "original; N=24 FAILED the original ratio gate (3.07 > 3)", "file": R / "rq1_batchcap16" / "cap16_seed1_compact.json",
     "note": "N=8 burn-in (diagnostic); N=24 passes every validity check but not the original execution-cost gate"},
    {"id": "C1.4-probe", "tag": "RQ1_PROBE_2d58666", "job": "51263752", "code_sha": "2d58666", "catalog": "GSM8K-train (hash 02e79304)",
     "serving": "batch_cap_16, FCFS", "protocol": "exploratory_v2 (ratio = telemetry); would FAIL the original gate (3.21, 3.28)", "file": R / "rq1_final_probe" / "probe_seed1_compact.json",
     "note": "first peak-memory measurement; N=28 first run of the job"},
]


# exploratory_v2 verdicts that were established by the READ-ONLY re-evaluation in C1.4 (the C1.3 compact file predates the v2 gate)
V2_REEVALUATED = {("RQ1_CAP16_5a37f03", 8): True, ("RQ1_CAP16_5a37f03", 16): True, ("RQ1_CAP16_5a37f03", 24): True}


def main():
    util = {(u["tag"], u["N"]): u for u in json.loads((R / "rq1_final_probe" / "utilisation_reanalysis.json").read_text())}
    rows = []
    for c in CAMPAIGNS:
        d = json.loads(c["file"].read_text())
        for r in d["runs"]:
            n = r["N"]
            u = util.get((c["tag"], n))
            v2 = r.get("gate_v2_ok", V2_REEVALUATED.get((c["tag"], n)))
            orig_pass = r["real_over_virtual"] <= 3.0 and r.get("gate_ok", True)
            rows.append({
                "campaign": c["id"], "tag": c["tag"], "N": n, "x_min": round(r["x_min"], 3), "Y": round(r["Y"], 2),
                "verifier_busy_corrected": round(u["utilisation_corrected"], 3) if u else None,
                "real_over_virtual": round(r["real_over_virtual"], 2), "original_gate": "pass" if orig_pass else "FAIL",
                "validity_checks_exploratory_v2": "not evaluated" if v2 is None else ("pass" if v2 else "FAIL"),
                "catalog": c["catalog"], "serving": c["serving"], "protocol": c["protocol"],
                "descriptive_figure": ("yes (original protocol)" if orig_pass
                                       else "yes, labelled exploratory_v2 only (original gate FAILED)" if v2 else "no"),
                "statistical_claim_supported": False})
    out = {"label": "Frozen evidence inventory C1.1-C1.4; every row is ONE run-seed (seed 1), a single population per catalog; no row supports a statistical claim.",
           "campaigns": [{k: (str(v) if isinstance(v, Path) else v) for k, v in c.items() if k != "file"} for c in CAMPAIGNS],
           "points": rows,
           "not_comparable": ["pilot (GSM8K-test) vs extension/cap16/probe (GSM8K-train): different prompts, never pooled",
                              "cap-32 vs cap-16 only where the cap does not bind (N <= 16): equivalent replicates, not a policy comparison",
                              "N=24 cap-16 (original gate FAIL) must not be mixed into 'original protocol' figures"],
           "figure_supporting_descriptive_rq1": ["batch_cap_16 train-catalog curve N=8,16 (original) + N=24 (exploratory_v2, flagged) + N=28,32 (exploratory_v2)",
                                                  "pilot N=1..16 as a low-load anatomy of the floor decline, test catalog, separate panel"],
           "cannot_support": ["any confidence interval / significance (one seed)", "a certified N*(x) or ICA", "a controller-vs-baseline claim",
                              "causal attribution of the N=8 anomaly", "N>=24 under the original protocol"]}
    (R / "c2" / "rq1_evidence_inventory.json").write_text(json.dumps(out, indent=1))
    print("| campaign | N | x_min | Y | busy (corrected) | real/virtual | original gate | exploratory_v2 | figure use |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['campaign']} | {r['N']} | {r['x_min']} | {r['Y']} | {r['verifier_busy_corrected']} | {r['real_over_virtual']} | {r['original_gate']} | "
              f"{r['validity_checks_exploratory_v2']} | {r['descriptive_figure']} |")


if __name__ == "__main__":
    main()
