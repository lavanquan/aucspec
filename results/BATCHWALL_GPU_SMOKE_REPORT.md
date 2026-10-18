# Batch-wall GPU smoke test report (Task B1)

Branch `research/verifier-batchwall`. Run code revision `04d0bda` (clean worktree, `git_dirty=false` recorded job-side). This is a correctness/sanity test, **not capacity certification**: one seed, 60 s measurement window. No frontier, baseline sweep, ablation or heterogeneity experiment was run, nothing was merged, and calibrated pricing was not enabled.

## 1. Storage actions (Stage 1) and quotas

`v2dev_dirty_backup_2026-09-28` (approved Option A), all checks passed before removal:

1. Scratch tar re-verified: sha256 `5d17a5bee9883a78d3877c14c1523c7caf59e61f2067f330c9bf25768a2bdc15` matches `SHA256SUMS_b0.txt`.
2. Copied to `/software/projects/pawsey1257/quanla/archives/v2dev_dirty_backup_2026-09-28.tar` (1,777,960,960 B); sha256 of the copy identical.
3. `tar --compare` of the `/software` copy against the live directory: clean. Sorted member names equal the live `find` listing exactly (12,096 entries = the directory plus 12,095 below it).
4. Restore test to a temporary directory: 11 representative files restored and md5-identical (W&B `.wandb`, `.syncstate`, `debug.log`, `requirements.txt` of one run; two `.jsonl` and three `.json` results; one `.py`); temporary directory removed.
5. No Slurm job and no open file under the directory (`squeue` empty for the user, `lsof` empty) at removal time; directory removed.
6. Both archives retained: `/scratch/pawsey1257/quanla/aucspec_archive_b0/v2dev_dirty_backup_2026-09-28.tar` (purgeable) and the `/software` copy above.

Restore: `mkdir -p /scratch/pawsey1257/quanla/restore && tar -xf /software/projects/pawsey1257/quanla/archives/v2dev_dirty_backup_2026-09-28.tar -C /scratch/pawsey1257/quanla/restore` (single file: add the member path after the archive name). Verify first with `sha256sum` against the value above.

Also done under your separate instruction to compress the specdiff projects: `specdiff_cqb`, `specdiff_cqb_v3dev`, `specdiff_pylibs` were each tarred+gzipped to `/software/projects/pawsey1257/quanla/archives_specdiff/*.tar.gz` (2.27 GB, 145 MB, 136 MB), verified with `tar -dzf` (clean), `gzip -t` and `SHA256SUMS.txt` (`sha256sum -c` OK), then the three directories were removed. Both git histories are inside the archives, including the 36 uncommitted changes in `specdiff_cqb` and the 328 commits of `specdiff_cqb_v3dev` branch `v3-dev` that had not been pushed. Restore: `tar -xzf …/specdiff_cqb.tar.gz -C <dest>`.

Not removed (as instructed): `vllm_rocm_build` (including its `tests`, `docs`, `examples`, `benchmarks`), caches, models, other users' or projects' environments, other worktrees.

**Quota after cleanup (`pawseyAccountBalance`)**

| FS | Files (user) | Share | Space |
|---|---|---|---|
| /software | 192,058 | 76.8% | 155.92 GiB user, 72.2% of project |
| /scratch | 241,346 | 24.1% | 128.49 GiB user |
| /home | 3,124 | 31.2% | 1024 MiB, 100% full |

`/software` file headroom is about 57,900 entries (limit about 250,000). The gate of 10,000 is met.

Remaining blockers: Acacia is still not configured (no `~/.config/rclone`, no keys; see the earlier report for the setup steps), so every raw archive, including the 12 raw-trace tars, exists only on purgeable `/scratch`, with the two new `/software` copies being the exception. `/home` stays 100% full.

## 2. CPU verification (Stage 2)

- Clean worktree on the branch; configuration `service_time_model: batch_wall` and `controller.verifier_price_mode: zero` are explicit in `configs/canonical_batchwall_v1.yaml` and the smoke config; the launcher guard refuses a missing/other value of either (new commit `9862e66`; behaviour of the controller is unchanged, the key is only validated and recorded).
- Verifier semantics recorded job-side: `V1`, tie-break `first_max_in_vllm_logprob_dict_order`, `effective_max_logprobs=1`, greedy only, budget 100, max batch 32. V2 edits are not in this branch.
- Regression tests: **251 passed** (`PYTHONPATH=legacy/src`, `.venv-sim`), including the 13 batch-wall timing tests, the 9 verifier-price characterization tests, guard/provenance tests. Three torch-dependent modules still fail collection on the login node, as before.
- Launcher refusal verified for real: with an untracked `scripts/zz_dirty_probe.py` present, `canonical_launch.py init` printed `GUARD REFUSED: source tree is not clean … untracked source files: ['scripts/zz_dirty_probe.py']` and created no run directory.
- Job-side provenance verified in the three run configs (section 3).
- Inspection of the batch accounting: `batch_start = max(batch_ready, server_available_at)`, `verify_finish = batch_start + modeled`, `server_available_at = verify_finish`, `client_receive = verify_finish + rtt/2 + download`; `target_latency_ms` is only logged.

First attempt note: the first set of smoke jobs (`bw_smoke60_20261019`, `v1ctl_smoke60_20261019`) failed after 13 s at the quota preflight because `pawseyAccountBalance` does not exist on compute nodes; no simulation ran. The run-time check now records a warning when the tool is missing (init-time login-node check stays the gate), commit `04d0bda`, and the runs were repeated with fresh immutable tags `*_r2_20261019`. The failed tags contain only logs.

## 3. Provenance of the smoke runs

| | Batch-wall N=1 | Batch-wall N=10 | Legacy-V1 control N=10 |
|---|---|---|---|
| Slurm job | 51097012 | 51097013 | 51097014 |
| Tag | `bw_smoke60_r2_20261019` | same | `v1ctl_smoke60_r2_20261019` |
| Code | `04d0bda`, dirty=false | same | same |
| Config | `configs/canonical_batchwall_smoke60.yaml` (hash `56644278e61c7a09`) | same | `configs/legacy_v1_control_smoke60.yaml` (hash `e0856ddc296ef415`), differs only in `service_time_model: amortized_per_request` |
| Seed / x / policy | 1 / 3.0 / capacity_dpp | same | same |
| Window | 30 s warm-up + 60 s measurement (virtual) | same | same |
| Population fingerprint | `7d873c2e94ab447b` (same as the certified runs) | same | same |
| Elapsed (wall) | 6:00 | 28:17 | 29:23 |

Hardware/software from the job process (nodes: N=1 `nid002126`, batch-wall N=10 `nid002128`, V1 control `nid003012`), each with 8 × AMD Instinct MI250X, ROCm 6.3.3-74, HIP 6.3.42134, amdgpu driver 6.12.12, python 3.12.3, torch 2.7.1a0, transformers 4.51.3, vllm 0.9.1+rocm633, module `pytorch/2.7.1-rocm6.3.3`. Full records: `results/smoke_batchwall_20261019/provenance.json`.

Compact outputs in the repo are under `results/smoke_batchwall_20261019/` with `bw_batchwall/` and `v1_control/` kept in separate directories. Raw traces/logs stay in `/scratch/pawsey1257/quanla/canonical_raw/<tag>/`.

## 4. Measured versus modeled batch timing

| | Batch-wall N=1 | Batch-wall N=10 | V1 control N=10 |
|---|---|---|---|
| Batches (whole run) | 1,466 | 7,460 | 7,460 |
| Batch size mean / max | 1.0 / 1 | 1.91 / 7 | 2.00 / 7 |
| Batch wall mean / p50 / p95 / max (ms) | 78.5 / 78.5 / 92.6 / 104 | 198.7 / 193.6 / 288 / 1,824 | 207.4 / 200.7 / 311 / 1,307 |
| modeled service / wall | 1.000 | 1.000 | 0.657 |
| event-loop measured / wall (mean) | 1.002 | 1.003 (max 1.036) | 1.003 |
| wall vs B linear fit (diagnostic only) | n/a | 118 ms + 42 ms per request | |

The event-loop measurement is slightly above the stamped wall time (thread hand-off); the stamped wall is what drives the clock.

## 5. Virtual scheduling invariants (all computed from the round CSV by `scripts/smoke_invariants.py`)

All pass in the batch-wall runs (tolerance 1e-3 ms; observed errors about 1e-10):

1. Virtual occupancy (`verify_finish − batch_start`) equals modeled service time; modeled equals the full batch wall (max error 0.0).
2. Requests of one batch share identical `verify_finish_ms`, `batch_start_ms`, `batch_ready_ms`, wall and service values (0 batches with disagreement); rows per batch equal batch size.
3. Consecutive batches never overlap on the virtual verifier: 0 overlapping pairs, minimum gap 168 ms (N=1) and 0.0 ms (N=10, back-to-back service); batch ids are in start order; no batch starts before it is ready; no request arrives after its batch is ready.
4. No double counting: `virtual_round_latency = draft + rtt/2 + upload + queue wait + service + rtt/2 + download` with residual below 1e-9 ms; adding the logged amortized `target_latency_ms` would break the identity by 78.5 ms (N=1) and 103.8 ms (N=10) on average, so it is correctly not part of the timeline.
5. Control run under V1: same checks pass with `modeled = wall/B` (modeled/wall = 0.657), confirming the V1 path is intact.

Workload exhaustion: none (`exhausted_clients = []` in all three runs; `exhausted=0` in the logs).

## 6. Logical versus processed tokens

| | N=1 | N=10 |
|---|---|---|
| Logical verifier tokens per batch (Σ(γ+1)) | 2.99 | 6.69 |
| Processed tokens per batch (context + draft) | 164.7 | 308.7 |
| Processed / logical (mean of per-batch ratios) | 98.6 | 66.3 |
| Context tokens per request (mean) | 162.8 | 158.4 |
| Batches where the token budget (100) binds | 0 | 0 |

The logical budget never binds (consistent with the earlier analysis), and the GPU work is dominated by context recomputation by roughly two orders of magnitude.

## 7. Change relative to legacy V1 (non-certification diagnostics, N=10, seed 1, 60 s, measurement window only)

| | V1-batchwall | V1 control |
|---|---|---|
| min client rate (tok/s) | 2.150 | 2.717 |
| median client rate | 2.483 | 3.350 |
| total goodput (tok/s) | 25.05 | 32.55 |
| mean round latency (ms) | 1,017 | 750 |
| verifier busy fraction (modeled service / span) | 0.999 | 0.798 |
| mean γ | 2.46 | 2.43 |
| mean batch wall (ms) | 211 | 216 |

Under batch-wall the single verifier is saturated at N=10 (busy about 100% of the window) and goodput is about 23% lower than the V1 control from the same code revision. Both values are single-seed, 60-s numbers; the certified 480-s V1 runs gave a min rate around 3.0 for N=10, so the 60-s control (2.72) also shows that this window is too short for threshold conclusions. N=1 min rate is 8.75 tok/s; no V1 control was run for N=1 because B=1 equality is covered by the unit test (and by modeled/wall = 1.000 here). Do not read these as a new N*(x).

## 8. Controller causality and noncausal observations

- With `verifier_price_mode=zero` the server price coefficient is 0, so Q_s cannot influence depth or batch selection, whatever the ordering of Q_s updates.
- The remaining cross-client input to a depth decision is the uplink rate signal. Its age in rounds is logged: the minimum is 0 and the value −1 appears exactly 10 times, once per client at round 0 (no signal yet); no negative ages beyond that sentinel. This shows the signal comes from the client's own earlier rounds, but it does not show that the *content* of the shared allocation respects virtual time across clients.
- Not validated (**unresolved**): (a) whether the shared allocator/controller state read by a client at its virtual decision time `draft_start_ms` includes events with later virtual time, because controller state is updated in real processing order while clients have different virtual clocks; (b) the `server_queue_price` and `theta_f` stamped on each request are not logged, so Q_s as seen at submission cannot be reconstructed from the event logs. This matters once pricing is enabled; it does not affect the zero-price runs above.

## 9. Remaining instrumentation issues

1. Log the stamped `server_queue_price`, `theta_f`, and the maximum virtual finish time of the batches already incorporated into controller state at each decision (needed for the causality check before pricing).
2. CSV rows cover warm-up as well as the measurement window; summaries must filter `in_measurement_window` (the scripts do).
3. `mean_batch_size` in the cache record is row-weighted; the batch-weighted value differs (2.58 vs 2.08 in the window).
4. A single batch of 1.8 s was observed (p95 0.29 s): heavy tail to be examined in the calibration data.
5. Unexplained coincidence to inspect: the batch-wall and V1-control N=10 runs both contain exactly 7,460 batches although their batch-size distributions (and row counts, 14,276 vs 14,931) differ. Every invariant holds in both, but I have not established why the totals are equal (a fixed-workload effect is the most likely reason; not verified).
6. `batch_wait_ms=150` contributes 150 ms of virtual queue wait even for one request (51% of the N=1 round latency of 294 ms), which is part of the configuration, not a bug, but dominates low-load latency.

## 10. Stage 4: pricing preparation (design only, nothing activated)

**Calibration target and model.** Fit full-batch wall time `T_b` (not wall/B) against processed context tokens, request count and draft tokens:
`T̂_b = θ0 + max{ m0 + m1·ΣL_i + m2·B,  f1·ΣP_i + f2·B }`, `P_i = L_i + γ_i`; compare with the simpler affine form `θ0 + c_req_ctx·ΣL_i + c_req·B + b·Σγ_i` on held-out data. The three requested components are separate parameters: fixed batch overhead `θ0`, per-request context cost (`c_req_ctx·L_i + c_req`), incremental draft-token cost `b`. Output `verifier_cost_model.json` with coefficients, confidence intervals, fit and data hashes.

**Controller price (still zero).** `verifier_price_mode ∈ {zero, calibrated_marginal}`; `zero` is the default and stays the only allowed mode in the guard. Unit-consistent form: `Q̃ = Q_s/Δ_s`; per-request marginal `ĉ_i = c_req_ctx·L_i + c_req + b·γ`; depth score `(V+Z)φ − κ_s·Q̃·ĉ_i/Δ_s`; knapsack value likewise plus `κ_s·Q̃·θ0/Δ_s` once per non-empty batch; Q_s update with the measured `T_b`.

**Sweep grid `rho_s = κ_s/V ∈ {0, 0.1, 0.3, 1, 3}` – magnitude check with smoke numbers.** With V=50, φ about 2–3 (benefit about 100–150), Δ_s = 100 ms, Q_s up to 1,724 ms (Q̃ = 17) in the N=10 run and a per-request slope of about 42 ms (ĉ/Δ_s about 0.4):
- request-level penalty ≈ ρ_s·V·Q̃·0.4: ρ_s=0.1 → about 34 at Q̃=17 and about 2 at Q̃=1 (modest); ρ_s=0.3 → about 100 at Q̃=17 (comparable to the whole benefit); ρ_s ≥ 1 → penalty exceeds the benefit whenever Q̃ ≳ 3, i.e. it would close the batch and rely on the forced-service escape hatch most of the time at saturation;
- depth-only penalty: if the per-draft-token cost were as large as the context-proportional upper bound (about 42 ms / 158 tokens ≈ 0.27 ms/token, an assumption to be tested), one extra draft token costs about 0.0027 slots; at Q̃=17 and ρ_s=3 this is about 7 against a marginal benefit of about 25, so the pricing mostly acts through admission/batching (the per-request and batch overhead terms), not through depth.
Recommendation: keep the grid but treat ρ_s ≥ 1 as a stress regime; add 0.03 as a lower point only if 0.1 already bites; decide after calibration provides ĉ_i.

**Baselines stay independent of DPP pricing.** Today `_round_time_estimate_ms` uses the same `current_verifier_theta_f` as DPP, so TurboSpec, FixedSLO and GELATO see a zero verifier-latency estimate in fixed-budget mode. Plan: a separate `baseline_verifier_latency_ms` input, defaulting to a causal running estimate of completed-batch service time (EMA of modeled service of batches with `finish_ms ≤ t_decision`, no constant invented) and later replaced by `T̂_b(L, γ, expected B)` from the calibration; it is unaffected by `verifier_price_mode`. The zero-price ablation then changes only DPP's price term.

**No future leakage.** Frozen offline coefficients (hash recorded); ledger of `(finish_ms, T_b)`; a decision at virtual time `t_d` uses only entries with `finish_ms ≤ t_d`; log the horizon used and count late-arriving ledger entries (section 8 items); a conservative virtual-time barrier is the exact alternative if late entries are frequent.

**Bounded GPU calibration plan (not submitted).** Grid: `B ∈ {1,2,4,8,16}` × three context lengths × `γ ∈ {0,1,2,3,4}` = 75 cells. Context lengths from the smoke run (B=1 batches, n=1,466): p10 = 112, p50 = 163, p90 = 215 tokens (range 79–272; these are early-in-sample contexts of the first 60 s, so the final choice should also use the full dataset's prompt+generation length quantiles computed on CPU first).
Measurements: 3 warm-up + 20 timed repetitions per cell = 75 × 23 = 1,725 verifier calls (1,500 timed); plus a heterogeneous-context validation set of 30 cells × 10 repetitions = 300 calls; total about 2,000 calls. Using the smoke fit (118 ms + 42 ms per request at about 160 tokens), one repetition over the five batch sizes costs about 1.9 s, so one repetition of the 15 (L, γ) cells costs about 28 s; the grid takes about 11 min, the validation set about 2 min, model load and start-up about 4–6 min (the N=1 job took 6 min in total). Expected total about 20 min in one job; request a 45-minute limit on one node (only the target GCD is used, but Setonix GPU jobs reserve the 8-GCD node; roughly 3–4 GCD-hours of allocation). Randomised cell order, fixed seed, results written as one compact CSV per run to `/scratch` plus a JSON fit summary in the repository.

## 11. GO / NO-GO for the calibration sweep

**GO (conditional)** for the bounded calibration above, independent of the controller:
- Stage 1 gate met (about 57.9k free file entries); Stage 2 passed; Stage 3 timing invariants all pass; no workload exhaustion; provenance complete; V1 and V1-batchwall kept separate.
- Conditions before submitting: (i) approval of the grid and time estimate; (ii) the logging additions of section 9 item 1 (non-behavioural) – not required for the timing calibration itself but required before any priced run; (iii) context quantiles refined from the full dataset.

**NO-GO** for enabling calibrated pricing or running any capacity frontier under V1-batchwall until the calibration is reviewed, the causality instrumentation exists, and a multi-seed, 480-s protocol is agreed; the smoke numbers must not be used as N*(x) estimates. Observed: at N=10 the corrected simulator saturates the verifier, so earlier V1 capacity numbers will not carry over.
