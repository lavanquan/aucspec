"""Abort gates for the RQ1 runs.  Exit 0 = continue, 3 = stop the job.  CPU only, no GPU.

Two protocols, both available and tested:
  * ORIGINAL (default): the pre-registered gate, including the real/virtual execution-cost ratio <= 3.  Unchanged; the C1.3 verdict
    "N=24 failed the original gate (3.07 > 3)" stays valid.
  * EXPLORATORY_V2 (explicit, env GATE_PROTOCOL=exploratory_v2, C1.4 only): the real/virtual ratio is recorded as OPERATIONAL TELEMETRY
    (still bounded by Slurm walltime and the GPU budget) and is not a validity failure.  Everything that is a scientific-validity
    requirement stays fail-closed and is checked here: runtime identity / no attention fallback, causality ledger, verifier serialisation,
    no dropped or duplicated requests, workload non-exhaustion, enough in-window rounds, batch-wall and token accounting,
    effective max_batch_size, finite measurements.  It must not be used to promote results obtained under the original protocol."""
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "c1"))
import analyze_rq1_pilot as an  # noqa: E402

MAX_REAL_OVER_VIRTUAL = 3.0
FROZEN_MANIFEST_SHA = "564571978a95c23c11136bf7bc66822357cd0b13eca2110c32348412f549bfed"
FROZEN_IDENTITY = "V1-batchwall-tritonfa33"


def csv_path(scratch_root, tag: str, n: int, seed: int, policy: str = "fixed_gamma", x: float = 3.0) -> Path:
    """Where run_capacity_frontier.run_one writes the rounds CSV: <scratch>/<tag>/raw/<policy>_n<N>_x<x>_s<seed>.csv"""
    return Path(scratch_root) / tag / "raw" / f"{policy}_n{n}_x{float(x)}_s{seed}.csv"


def gate(csv: Path, n: int, ledger: Path | None, expect_max_batch: int | None = None, *, check_ratio: bool = True) -> tuple[bool, list[str]]:
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
    if check_ratio and not (m["real_over_virtual"] <= MAX_REAL_OVER_VIRTUAL):
        reasons.append(f"real/virtual time ratio {m['real_over_virtual']:.2f} > {MAX_REAL_OVER_VIRTUAL}")
    if expect_max_batch is not None:
        if "verifier_max_batch_size" not in df.columns or set(df["verifier_max_batch_size"].unique()) != {expect_max_batch}:
            reasons.append(f"effective max_batch_size != {expect_max_batch}")
        if m["batch_size_max"] > expect_max_batch:
            reasons.append(f"a batch of {m['batch_size_max']} > cap {expect_max_batch}")
    if m["rounds_per_client_min"] < 20:
        reasons.append(f"fewer than 20 in-window rounds for some client ({m['rounds_per_client_min']})")
    return (not reasons), reasons


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def gate_exploratory_v2(csv: Path, n: int, ledger: Path | None, expect_max_batch: int | None, run_config: Path | None
                        ) -> tuple[bool, list[str], dict]:
    """Fail-closed scientific-validity checks; the real/virtual ratio is returned as telemetry only."""
    ok, reasons = gate(csv, n, ledger, expect_max_batch, check_ratio=False)
    reasons = list(reasons)
    df = pd.read_csv(csv, low_memory=False)
    m = an.run_metrics(df, n)
    telemetry = {"real_over_virtual": m["real_over_virtual"], "ratio_is_telemetry_only": True,
                 "verifier_utilisation": m["verifier_utilisation"], "verifier_utilisation_legacy": m.get("verifier_utilisation_legacy")}
    # finite, valid measurements
    if not all(_finite(v) for v in list(m["rates"].values()) + [m["x_min"], m["Y"], m["x_mean"]]):
        reasons.append("non-finite rates")
    for col in ("batch_start_ms", "verify_finish_ms", "client_receive_ms", "verification_batch_wall_ms", "committed_token_count"):
        if col not in df.columns or not df[col].map(lambda v: math.isfinite(float(v))).all():
            reasons.append(f"missing or non-finite column {col}")
    if not _finite(m["real_over_virtual"]):
        reasons.append("real/virtual telemetry is not finite (wall clock missing)")
    # verifier serialisation (no overlapping service intervals)
    if m.get("verifier_serialisation_ok") is not True:
        reasons.append("verifier service intervals overlap or finish times are missing")
    # batch-wall accounting under the batch_wall clock
    if {"verification_batch_service_ms", "verification_batch_wall_ms", "batch_start_ms", "verify_finish_ms"} <= set(df.columns):
        b = df.drop_duplicates("verification_batch_id")
        if not ((b["verification_batch_service_ms"] - b["verification_batch_wall_ms"]).abs() < 1e-6).all():
            reasons.append("modeled service time != measured batch wall (batch_wall accounting)")
        if not (((b["verify_finish_ms"] - b["batch_start_ms"]) - b["verification_batch_service_ms"]).abs() < 1e-3).all():
            reasons.append("verify_finish - batch_start != modeled service time")
        if not (b["verification_batch_wall_ms"] > 0).all():
            reasons.append("non-positive batch wall")
    else:
        reasons.append("batch-wall accounting columns missing")
    # dropped / duplicated requests (the ledger covers the event level; here the record level)
    if df.duplicated(["client_id", "sample_id", "round_id"]).any():
        reasons.append("duplicated (client, sample, round) records")
    sizes = df.groupby("verification_batch_id")["client_id"].count()
    declared = df.drop_duplicates("verification_batch_id").set_index("verification_batch_id")["verification_batch_size"]
    if not (sizes.reindex(declared.index) == declared).all():
        reasons.append("batch member count != declared batch size (dropped or duplicated request)")
    # token accounting
    if "useful_tokens" in df.columns and not (df["useful_tokens"] == df["committed_token_count"]).all():
        reasons.append("useful_tokens != committed_token_count")
    # runtime identity / no attention-backend fallback
    if run_config is None or not Path(run_config).exists():
        reasons.append("run_config.json missing (runtime identity cannot be verified)")
    else:
        rc = json.loads(Path(run_config).read_text())
        ri = rc.get("runtime_identity", {})
        live = ri.get("live_engine", {})
        if ri.get("identity") != FROZEN_IDENTITY or ri.get("manifest_sha256") != FROZEN_MANIFEST_SHA:
            reasons.append("runtime identity / manifest hash differs from the frozen record")
        if not (live.get("layers_triton_flash_attn") == live.get("n_expected_layers") == live.get("n_attention_layers")
                and live.get("layers_naive_sdpa") == 0):
            reasons.append("attention layers are not all Triton FlashAttention (fallback)")
        if live.get("effective_max_logprobs") != 1 or live.get("prefix_caching") is not False:
            reasons.append("verifier semantics fields differ from the frozen record")
    return (not reasons), reasons, telemetry


if __name__ == "__main__":
    if sys.argv[1] == "--path":                       # --path <scratch_root> <tag> <n> <seed>
        print(csv_path(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])))
        sys.exit(0)
    csv, n = Path(sys.argv[1]), int(sys.argv[2])
    art = Path(str(csv).replace(".csv", "_artifacts"))
    cap = os.environ.get("EXPECT_MAX_BATCH")
    cap_i = int(cap) if cap else None
    if os.environ.get("GATE_PROTOCOL") == "exploratory_v2":
        ok, why, tel = gate_exploratory_v2(csv, n, art / "causality_check.json", cap_i, art / "run_config.json")
        print(json.dumps({"csv": str(csv), "n": n, "protocol": "exploratory_v2", "ok": ok, "reasons": why, "telemetry": tel}, default=str))
    else:
        ok, why = gate(csv, n, art / "causality_check.json", cap_i)
        print(json.dumps({"csv": str(csv), "n": n, "protocol": "original", "ok": ok, "reasons": why}))
    sys.exit(0 if ok else 3)
