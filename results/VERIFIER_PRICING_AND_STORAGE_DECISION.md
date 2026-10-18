# Verifier pricing and storage: decision report (Task B0)

Branch `research/verifier-batchwall` (base `47292cf`). No GPU job was run, no frontier launched, nothing merged to main, and no new verifier pricing was enabled.

Approved decisions carried in: verifier semantics stay V1 (`max_logprobs=1`, original tie-break; V2 not merged); `service_time_model=batch_wall`; zero verifier price in fixed-budget mode is a limitation to fix; `/scratch` for temporary raw data and Acacia for persistent archives; historical V1 certified results stay separate (they live on `main` and in the tar archives; I recommend a tag such as `v1-certified-6e0e7e7` on main, which I have not created because it is an outward-facing push).

## 1. Storage actions completed

| Item | Result |
|---|---|
| `aucspec_wt_921dedd` | **Removed.** Checks: no Slurm jobs or processes for the user; `921dedd` is an ancestor of `main` and other pushed branches, working tree had only two untracked items (a run-dir symlink pointing back into the main dir, and the wrapper `run_x3_cert_job_wt.sh`), both inside the archive; no script in the repo references the path. Archived to `/scratch/pawsey1257/quanla/aucspec_archive_b0/aucspec_wt_921dedd.tar` (169,379,840 B, 709 entries), `tar --compare` clean, sha256 `8a6564ba…98b8` (full value in `SHA256SUMS_b0.txt` next to it). Removed with `git worktree remove --force` + `git worktree prune`; `git worktree list` no longer lists it. |
| `aucspec/vllm_rocm_build` | **Not removed: condition failed.** The installed vLLM is an editable install whose finder maps `vllm` to `/software/projects/pawsey1257/quanla/aucspec/vllm_rocm_build/vllm/vllm`, and the built `_C`, `_moe_C`, `_rocm_C` `.so` files live inside it. Deleting the directory would break the vLLM runtime used by every simulator run. It is a clean git checkout (`b6553be`), but the `.so` files are build products, so it is not "safely reproducible" without a rebuild. Safe partial option (needs your approval): remove only the non-runtime subtrees `tests` (719 files), `docs` (217), `examples` (175), `benchmarks` (58), about 1,170 files, all recoverable from git; `csrc` (245) should stay if a rebuild is ever needed. |
| `v2dev_dirty_backup_2026-09-28` | **Not deleted. Inventoried and backed up.** 1.7 GB, 12,095 files: `wandb_dirty/` 11,863 files (1,080 offline W&B runs, 1.5 GB: logs, txt, syncstate; not in git; I did not check whether they were ever synced), `depth_campaign/` 232 files, 217 MB: 58 `.py/.md/.sbatch` files (every one has a byte-identical, same-named tracked copy in `specdiff_cqb`, `specdiff_cqb_v3dev`, `SpecPlan` or `pluto`, so recoverable from Git) and 137 `.json/.jsonl` outputs (not in git, most likely irreplaceable). Full-directory tar written to `/scratch/…/aucspec_archive_b0/v2dev_dirty_backup_2026-09-28.tar` (1,777,960,960 B), `tar --compare` clean, sha256 `5d17a5be…dc15`. This copy is on purgeable `/scratch` only. |

**Proposed durable plan for `v2dev_dirty_backup` (needs your approval):**
Option A (frees about 12k files without waiting for Acacia): copy the verified tar to `/software/projects/pawsey1257/quanla/archives/` (1 file, 1.7 GB; `/software` is not purged), re-verify sha256 against the manifest, then replace the directory with the tar. Option B: same, but delete only after the Acacia upload and read-back also succeed. I recommend A now and the Acacia copy as a second copy.

**Headroom target not met.** `/software` file-count headroom is about 1,870 (248,131 of about 250,000 used) so the 10,000 target is not reachable with the approved removals alone. With Option A it becomes about 14,000. The launcher guard (margin 2,000 files) is unchanged.

## 2. Quotas now (`pawseyAccountBalance`, counter lags deletions slightly)

| Filesystem | Files (user) | Space |
|---|---|---|
| /software | 248,131 (99.3%) | 165.65 GiB user, 76.0% of project quota |
| /scratch | 225,790 (22.6%) | 127.57 GiB user |
| /home | 3,124 (31.2%) | 1024 MiB, 100% full |

## 3. Acacia readiness and archive durability

Status: **not ready; nothing uploaded.** `rclone/1.68.1` is available as a module; there is no `~/.config/rclone`, no `RCLONE_CONFIG`, no AWS/S3 variables; `/home` is full so the default config location cannot be written. I did not look for credentials.

Setup for you (do not paste keys into chat, logs, reports or Git):
1. In the Pawsey portal create an Acacia access key pair for project `pawsey1257` and a bucket (for example `aucspec-archive`) in the project's Acacia allocation (S3 endpoint `https://projects.pawsey.org.au`, per Pawsey's S3-client documentation).
2. On Setonix: `mkdir -p /scratch/pawsey1257/quanla/.rclone && chmod 700 /scratch/pawsey1257/quanla/.rclone`, then `export RCLONE_CONFIG=/scratch/pawsey1257/quanla/.rclone/rclone.conf`, `module load rclone/1.68.1`, `rclone config` (new remote `acacia`, type `s3`, provider `Other`, enter the keys interactively, endpoint as above), `chmod 600 $RCLONE_CONFIG`. Keep this file out of Git. Because `/scratch` is purged, keep a copy of the config in a private location of your choice (not the repo).
3. Test: `rclone lsd acacia:`.

Prepared (not run): `scripts/acacia_transfer.sh <bucket> <prefix> <files…>`: dry run by default, `APPLY=1` uploads with `--local-no-set-modtime`, runs `rclone check` for sizes/hashes, then downloads each file to `/scratch` and compares sha256 (read-back). Intended payload: the 12 raw-trace tars in `aucspec_raw_archive/` (19.88 GiB) with `SHA256SUMS.txt`, plus `aucspec_archive_b0/*.tar` and `SHA256SUMS_b0.txt`, plus a small verification-metadata file (remote sizes, check output, read-back result). Multipart S3 ETags are not MD5, so the sha256 read-back is the authoritative integrity check. No scratch archive will be deleted without separate approval. Durability today: only single copies on `/scratch` (21-day inactivity purge per the Pawsey documentation excerpt cited in the earlier report; not re-verified against the live page).

## 4. Confirmed zero-price code path

Causal chain as implemented (file references are in the worktree):

1. Profiling: `RooflineBudgetModel.signal_for` returns `(budget, regime, theta_f)` only when a roofline model is configured. In fixed-budget mode `VerificationBatcher.estimate_verify_budget_signal` returns `(verify_token_budget, "memory_bound", 0.0)`.
2. `simulator._run_round` (about line 628) calls `controller.set_verifier_theta_f_signal(0.0)` before every depth decision. The controller's constructor default of 1.0 is therefore always overwritten with 0.0.
3. Depth: `capacity_dpp_score = (V+Z_i)·phi − Q_s·theta_f·(γ+1) − Q_i·γ·c_dev`; with theta_f = 0 the Q_s term is exactly 0.
4. Submission: the request is stamped with `server_queue_price = Q_s` and `theta_f = 0`, so `capacity_dpp_value = weight·E[useful] − Q_s·0·(γ+1)`; the `capacity_knapsack` selection reduces to maximising weighted expected tokens.
5. Batch completion: batch metadata computes a *different* `theta_f` = modeled batch service / logical verifier tokens (V1-batchwall: full batch wall / Σ(γ+1)), and `observe_server_batch` uses `theta_f·cost` = the modeled batch service in ms to update Q_s with a slot-based Lindley recursion (`slot_ms = 100`). So Q_s grows with real verifier busy time, but nothing reads it with a nonzero coefficient.
6. Side effects found: (a) the baselines (`turbospec`, `fixed_slo`, `gelato`) estimate round time via `_round_time_estimate_ms`, which also uses `current_verifier_theta_f_ms_per_token·(γ+1)`, so their verifier-latency estimate is also zero in fixed mode; changing the shared signal would silently change the baselines. (b) `server_price_scale` is read from config and stored on the controller but never used. (c) Q_s is updated in the order batches are processed in real time, while client decisions happen at their own virtual times, so a decision can see a Q_s that includes batches whose virtual finish lies after the decision time. This (c) is derived from reading the code and has not been measured.

These are pinned by the CPU tests in section 7.

## 5. Proposed verifier-cost model and estimator (design only)

**Quantities to keep apart**
- Batch service time `T_b`: measured wall time of one batched call; the single verifier is occupied for `T_b` (V1-batchwall).
- Fixed per-batch overhead `θ0`: the intercept of `T_b` as workload goes to zero.
- Marginal cost of extra work: `∂T_b/∂(one more draft token)` at fixed batch and contexts `= b`, and the per-request cost `c_req(L) = ∂T_b/∂(one more request)` which includes prefilling its context of length `L`. `c_req` does not depend on γ.
- Logical verifier positions `N_i = γ_i + 1` (what the scheduler budget counts today).
- Processed tokens `P_i = L_i + γ_i` (what the GPU actually computes: stateless prefill of context plus draft, no prefix cache). Earlier measurements showed `P` is about an order of magnitude or more above `N`, so `θ_f = T_b/ΣN` mixes these and must not be used as a marginal price. `T_b/B` is likewise an average, not a marginal cost (it is the V1 error).

**Model.** `T̂_b = θ0 + max{ m0 + m1·ΣL_i + m2·B ,  f1·ΣP_i + f2·B }` (roofline form, with memory and compute branches over processed tokens). Existing data (wall roughly 129.6 + 44.4·B ms) show an intercept and a per-request slope but do **not** identify `b`, because context dominates `P`; I am not assuming a value. Candidate simpler forms (affine in `ΣP` and `B`) are fitted and compared on held-out data.

**Calibration (GPU, later, bounded).** Run the real `verify_batch_sync` path over a grid: `B ∈ {1,2,4,8,16}`, `L` at observed context percentiles (p10/p50/p90 from the GSM8K runs), `γ ∈ {0,…,γmax}`, several repetitions, warm-up discarded. Fit by least squares with a train/validation split, select the form by held-out error, and write `verifier_cost_model.json` (coefficients, confidence intervals, fit and data hashes). If the confidence interval of `b` includes 0, the depth price is legitimately negligible and that must be reported as a measured fact; the per-request term `c_req` and `θ0` would still act on batching and on Q_s.

**Prices used by the controller (explicit units).** Let `Δ_s` be the slot (ms), `Q̃ = Q_s/Δ_s` (dimensionless backlog in slots). Per-request marginal cost in ms: `ĉ_i(γ) = c_req(L_i) + b·γ`. Then
- depth: `S_i(γ) = (V+Z_i)·phi_i(γ) − κ_s·Q̃·ĉ_i(γ)/Δ_s − Q_i·γ·c_dev`,
- knapsack item value: `v_i = (V+Z_i)·phi_i(γ_i) − κ_s·Q̃·ĉ_i(γ_i)/Δ_s`, with a batch-level start cost `κ_s·Q̃·θ0/Δ_s` charged once if the selected set is non-empty,
- queue: `Q_s ← [Q_s + Σ_batches T_b − Δ_s]^+` per slot, using the **measured** `T_b` (the same quantity that advances the verifier clock).
`κ_s` (tokens per unit of normalised backlog cost) is an explicit Lyapunov weight, a tunable like `V`; it is not a physical constant and must be swept and reported rather than chosen to fit curves. The paper's `λθ_f` form mixes seconds and tokens unless such a weight is implicit.

**Causality in asynchronous operation.**
1. The coefficients are frozen offline before a run and their hash is recorded: no online learning leakage.
2. Keep an append-only ledger of `(finish_ms, T_b)` per batch. A decision at virtual time `t_d` uses only ledger entries with `finish_ms ≤ t_d` and recomputes Q_s by replaying the Lindley recursion over them.
3. Log, for each decision, the horizon `h = max finish_ms` used. If an entry later arrives with `finish_ms < h` of a past decision, count it in `causality_late_entries` and report it; this measures how often real-order execution violates virtual-time causality.
4. If that count is non-negligible, the only exact fix is a conservative barrier (a client decision at virtual time `t_d` waits until no unprocessed batch can finish before `t_d`); this is stated as an option, not implemented.
5. At decision time only `L_i`, own history and the candidate `γ` are used; batch composition is unknown, so the price uses the per-request marginal `ĉ_i`, not the realised batch size.

**Ablation.** `controller.verifier_price_mode ∈ {zero, calibrated_marginal}`. `zero` must reproduce the current numerics exactly (price 0, Q_s still logged, baselines unchanged) and stays the default until a reviewed enable. Baselines keep their own, explicitly recorded latency proxy so that changing DPP pricing never changes them.

## 6. Necessary implementation changes (not made)

1. New `verifier_cost.py`: `VerifierCostModel` (load JSON, hash, `marginal_ms(L, γ)`, `batch_ms(...)`), plus a calibration script reusing `verify_batch_sync` and the new batch-wall metrics.
2. Config: `controller.verifier_price_mode`, `controller.kappa_s`, `verification_batching.verifier_cost_model_path`; wire `server_price_scale` or remove it.
3. `simulator._run_round`: supply `ĉ_i` (not a shared `theta_f`) to the controller; keep baselines on a separate recorded proxy.
4. `controller`: price from `Q̃`, `κ_s`, `ĉ_i`; ledger-based causal Q_s; late-entry counter.
5. `VerificationRequest`: carry `verifier_cost_ms`; `capacity_dpp_value` uses it; batch start cost in the DP; optional processed-token budget alongside the logical one.
6. Logging: price, `Q̃`, `ĉ_i`, predicted vs measured `T_b` (residual), horizon, mode, model hash, κ_s.

## 7. CPU test results

`PYTHONPATH=legacy/src`, `.venv-sim`: **249 passed** (240 before + 9 new in `legacy/tests/test_verifier_price_regression.py`); the 13 batch-wall timing tests are included and pass. The three modules that need torch still fail collection exactly as before. The new tests assert: fixed-budget signal is `(100, "memory_bound", 0.0)`; the simulator wiring leaves the controller with theta_f = 0 (constructor default 1.0 overwritten); Q_s = 150 ms after a 250 ms batch while depth scores and the chosen depth equal those of a zero-queue controller; a synthetic theta_f = 0.5 lowers the depth score by exactly `Q_s·θ_f·(γ+1)` and a larger one reduces the selected depth; a synthetic price changes knapsack item values by the stated formula and flips the selected request; batch metadata satisfies `theta_f·logical tokens = modeled service`, is not the price the controller used (30 ms/token vs 0) and logical positions (4) differ from processed tokens (13); Q_s update uses ms and the 100 ms slot. These document current behaviour; no pricing algorithm was changed.

## 8. Paper-side qualifications (Sections 3–4: `thm:frontier`, `thm:dpp`)

1. `as:roofline`/`eq:Tv` assume `Γ(B)` counts newly verified positions and `θ_m` has a KV term. The implemented verifier recomputes the whole context every round, so the compute term scales with `ΣP_i = Σ(L_i+γ_i)`, not `Σ(γ_i+1)`. Theorem 1(d) (existence of `B_c` where speculation switches off because `phi/(γ+1) < 1`) then changes: with `P = L+γ` and `L ≫ γ` the ratio `phi(γ)/(L+γ)` can exceed `1/L`, so `B_c` may not exist or may be much larger. Parts (b) and (c) (unimodality, stopping rule `eq:gammastar`, monotonicity in `B`) need `ΔT` recomputed with the correct cost increment; I have not re-derived them.
2. `eq:index` prices a draft token at `λθ_f` per logical position. For this verifier the per-draft-token marginal is `b`, not `θ_f`, the context cost is a per-request term that does not affect depth, and positions are `γ+1` while processed tokens are `L+γ`.
3. Subproblem (ii): ranking by `ω_i/(γ_i+1)` and budget `Γ_bud` at the knee assume cost proportional to logical tokens. Here item cost is `c_req(L_i)+bγ_i`, `θ0` is a per-batch fixed cost (not separable across items), and the `max` in `T_v` makes batch time non-linear, so the one-slot drift bound is a linearisation, not exact.
4. `eq:vq`: `Q` accumulates `θ_fΓ(B(t))`, the true occupied time is `T_b = θ0+max{…}`; the update must use measured/predicted batch time. Also `Q` (time) and `V+Z` (tokens) need an explicit conversion weight (`κ_s` above) for `eq:index` to be dimensionally consistent.
5. `thm:dpp` is for slotted, synchronous decisions with known α. The simulator is event-driven and asynchronous, Q_s is updated in real-processing order, α is learned (UCB), and acceptance under greedy verification is deterministic given draft and target rather than i.i.d. Bernoulli. The optimality gap `C/V` and the `O(V)` backlog bound therefore must not be claimed for the implementation; the pricing proposed here is a principled heuristic and this report does not claim it restores Theorem 3.
6. V1-batchwall (single verifier busy for the full batch wall time) is the setting where `Q` and the verifier clock describe the same resource; V1 amortised timing did not.

## 9. Prerequisites for a controlled GPU smoke test

1. Free at least 10,000 `/software` files (approve Option A for `v2dev_dirty_backup`; optionally the 1,170 non-runtime vLLM files); guard margin stays at 2,000.
2. Acacia keys configured by you; upload, `rclone check` and sha256 read-back of the 12 raw tars and the two B0 tars.
3. Review of the pricing design (section 5), including `κ_s` sweep range and whether baselines get a frozen proxy.
4. Approval of a bounded calibration sweep (grid in section 5; GPU time estimate to be fixed once the grid is approved).
5. Then, with `verifier_price_mode: zero` as the default, the V1-batchwall parity smoke from the earlier report (N=1 and N=10, seed 1) via `canonical_launch.py`, before any run with `calibrated_marginal`.
