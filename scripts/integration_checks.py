"""Post-run checks of the canonical-launcher wiring smoke (CPU only, read-only). Fail-closed: any missing evidence fails."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "legacy" / "src"))

IDENTITY = "V1-batchwall-tritonfa33"


def check_run(run_dir, n_clients, seed, questions_per_client, max_new_tokens, expected_sha=None):
    import pandas as pd
    import smoke_invariants as si

    run_dir = Path(run_dir)
    problems, out = [], {}
    csvs = sorted((run_dir / "raw").glob(f"capacity_dpp_n{n_clients}_x*_s{seed}.csv"))
    arts = sorted((run_dir / "raw").glob(f"capacity_dpp_n{n_clients}_x*_s{seed}_artifacts"))
    if not csvs or not arts:
        return {"ok": False, "problems": ["raw csv or artifacts directory missing"], "checks": {}}
    rc = json.loads((arts[0] / "run_config.json").read_text())
    ri = rc.get("runtime_identity") or {}
    live = ri.get("live_engine") or {}
    out["runtime_identity"] = {k: ri.get(k) for k in ("identity", "vllm_version", "vllm_file")}
    out["runtime_identity"].update({k: live.get(k) for k in ("engine_class", "n_attention_layers", "layers_triton_flash_attn", "layers_naive_sdpa",
                                                           "effective_max_logprobs", "tie_break_rule", "prefix_caching", "triton_fa_calls", "pids", "pid_now")})
    out["runtime_imports"] = ri.get("imports")
    out["semantics"] = ri.get("semantics")
    if ri.get("identity") != IDENTITY:
        problems.append("runtime identity missing or wrong in run_config.json")
    if not (live.get("n_attention_layers") and live.get("layers_triton_flash_attn") == live.get("n_attention_layers")
            and live.get("layers_naive_sdpa") == 0):
        problems.append("not every attention layer used Triton FA, or a naive-SDPA layer exists")
    if not live.get("triton_fa_calls", 0) > 0:
        problems.append("zero Triton FA invocations recorded inside the simulator process")
    if live.get("effective_max_logprobs") != 1:
        problems.append("effective max_logprobs is not 1")
    prov = rc.get("provenance") or {}
    out["provenance"] = {"git_sha": prov.get("git_sha"), "git_dirty": prov.get("git_dirty"),
                         "slurm_job_id": ((prov.get("node") or {}).get("slurm") or {}).get("SLURM_JOB_ID"),
                         "campaign_config_hash": prov.get("campaign_config_hash"), "population_fingerprint": prov.get("population_fingerprint")}
    if prov.get("git_dirty") is not False or (expected_sha and prov.get("git_sha") != expected_sha):
        problems.append("provenance: dirty tree or SHA differs from the run directory")
    if not out["provenance"]["slurm_job_id"]:
        problems.append("provenance: no Slurm job id")
    vs = rc.get("verifier_semantics") or {}
    out["verifier_semantics"] = vs
    if vs.get("service_time_model") != "batch_wall" or vs.get("verifier_price_mode") != "zero":
        problems.append("service_time_model/verifier_price_mode are not batch_wall/zero")

    inv = si.check(csvs[0], "batch_wall")
    out["timing_invariants"] = {k: inv.get(k) for k in ("pass", "failures", "n_batches", "overlapping_batch_pairs", "occupancy_minus_modeled_ms_max_abs",
                                                       "round_latency_decomposition_resid_ms_max", "modeled_over_wall_mean", "batch_size",
                                                       "verifier_busy_fraction")}
    if not inv["pass"]:
        problems.append(f"timing/serialization invariants failed: {inv['failures']}")
    df = pd.read_csv(csvs[0], low_memory=False)
    b = df.drop_duplicates("verification_batch_id")
    n_gt2 = int((b["verification_batch_size"] > 2).sum())
    out["batches_with_B_gt_2"] = n_gt2
    out["b_gt2_exercised"] = n_gt2 > 0
    out["batch_size_counts"] = {int(k): int(v) for k, v in b["verification_batch_size"].value_counts().sort_index().items()}
    # acceptance / committed-token accounting
    acc_bad = int(((df["accepted_length"] < 0) | (df["accepted_length"] > df["gamma"])).sum())
    com_bad = int((df["committed_token_count"] > df["accepted_length"] + 1).sum())
    use_bad = int((df["useful_tokens"] != df["committed_token_count"]).sum())
    out["acceptance_accounting"] = {"rows": int(len(df)), "accepted_outside_[0,gamma]": acc_bad, "committed>accepted+1": com_bad,
                                    "useful!=committed": use_bad, "mean_accepted_over_gamma": float(df["accepted_length"].sum() / max(1, df["gamma"].sum()))}
    if acc_bad or com_bad or use_bad:
        problems.append(f"acceptance accounting inconsistent: {acc_bad}/{com_bad}/{use_bad}")
    per_sample = df.groupby(["client_id", "sample_id"]).agg(tokens=("committed_token_count", "sum"), last=("generated_for_sample", "max"))
    token_mismatch = int((per_sample["tokens"] != per_sample["last"]).sum())
    over = int((per_sample["tokens"] > max_new_tokens).sum())
    per_client = df.groupby("client_id")["sample_id"].nunique()
    out["workload_accounting"] = {"samples": int(len(per_sample)), "per_client_samples_min": int(per_client.min()), "expected_per_client": questions_per_client,
                                  "clients": int(per_client.size), "sample_token_sum_mismatch": token_mismatch, "samples_over_max_new_tokens": over}
    if token_mismatch or over or int(per_client.size) != n_clients or int(per_client.min()) != questions_per_client:
        problems.append("workload accounting inconsistent (token sums, sample counts or client count)")
    return {"ok": not problems, "problems": problems, "checks": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--questions-per-client", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--expected-sha", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = check_run(a.run_dir, a.n, a.seed, a.questions_per_client, a.max_new_tokens, a.expected_sha)
    print(json.dumps(rep, indent=1, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    sys.exit(0 if rep["ok"] else 4)
