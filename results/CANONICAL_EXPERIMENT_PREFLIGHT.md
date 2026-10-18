# Canonical experiment preflight (audit only, no GPU)

Audit date: 2026-10-18. Everything below was obtained by read-only inspection of the repository, the Setonix filesystems, archived raw data and the existing N=10/11/12 certification run (`results/capacity_frontier/budget100_x3_certification/`). No GPU job was launched, no user file was modified, deleted, committed or reset. The only files created outside this report are small bookkeeping files in `/scratch/pawsey1257/quanla/aucspec_raw_archive/` (`SHA256SUMS.txt`, `SHA256SUMS.err`, `SHA256SUMS.done`, `filecounts_quanla.txt`, `filecounts.done`).

Evidence labels: **[measured]** = computed from run data in this audit; **[code]** = read from source; **[inferred]** = reasoning not yet tested.

## 0. Summary

| # | Finding | Severity |
|---|---|---|
| 1 | `main@680845e` has simulator/driver code byte-identical to the certified revision `921dedd` (only a report script and its test differ). It is a valid canonical base. | ok |
| 2 | The main working directory is detached at `cf5f434` with 2 uncommitted modifications to the verifier (`target_client.py`) and trace collection (`collect_trace.py`) and 2 local-only commits that no branch references. These must not be used for canonical runs. | decision needed |
| 3 | The uncommitted verifier edit changes the default greedy `max_logprobs` (1 -> 2) and the tie-break of the target argmax. It can change accepted tokens at logprob ties and per-call timing. Magnitude is unmeasured. | blocker for mixing semantics |
| 4 | The simulated verifier busy time is the **per-request average** of the real call time (`wall / B`), not the batch wall time. This is a modeling semantic that I did not find documented in the config or reports and that is decisive for roofline/theory validation. | decision needed |
| 5 | `/software` per-user file count is 248,754 of 250,000 (99.5%): about 1,250 files of headroom. A full campaign needs several thousand. `/home` is at 100%. | blocker |
| 6 | Raw traces exist only in tar files on purgeable `/scratch` (plus the un-archived raw of the certification run). Compact results are in git. | risk |
| 7 | Provenance records the git SHA but not dirty state, GPU model, ROCm version, node or Slurm job id; `environment.json` is written on the login node. | patch needed |

## 1. Code and worktree status

### 1.1 Worktrees, HEADs, branches

| Path | HEAD | Branch | Tracked changes | Untracked |
|---|---|---|---|---|
| `/software/projects/pawsey1257/quanla/aucspec` (main dir) | `cf5f434` | detached | `legacy/src/edge_specsim/target_client.py`, `scripts/collect_trace.py` | 49 (mostly Slurm logs, run scripts) |
| `.../aucspec_wt_921dedd` | `921dedd` | detached | none | 2 (wrapper `run_x3_cert_job_wt.sh`, symlink to the run dir) |
| `.../aucspec_wt_main` | `680845e` | `main` (= `origin/main`) | none | 0 |

- `origin/main` = `680845e` (x=3.0 N=12 certification).
- Local-only commits `cb34518` (causal HF speculative target runtime) and `cf5f434` (HF target-only qualification op) sit on top of `6ccbda0`, are reachable only from the detached HEAD and the reflog, and are not on any branch or remote. They add `hf_target_client.py`, `specplan_worker_hf.py` and tests; **no file imported by the simulator** (`grep` finds importers only in `legacy/tests/test_hf_target_client.py` and `scripts/specplan_worker_hf.py`). Nothing in git protects them from eventual garbage collection.
- No stashes.

### 1.2 Comparison with the certified simulator

- Files changed between `921dedd` and `680845e` outside `results/`: `scripts/certify_x3_report.py`, `legacy/tests/test_certify_x3.py` only. The diff over `legacy/src`, `legacy/scripts`, `scripts/run_capacity_frontier.py`, `scripts/certify_x3_run.py`, `scripts/run_x3_cert_job.sh` and the certification config is **empty** [measured].
- The same holds for `4b24977` (cache-key fix), `e5670f3` (Gate B-D report) and `ac1159d` (7-point sweep) versus `921dedd`: no simulator/driver difference. Results at or after `4b24977` therefore share one simulator code path. `aef4d9a` and earlier differ in `capacity.py` (the cache fix) and should not be pooled with later runs without re-checking.
- Committed `target_client.py` and `collect_trace.py` are identical at `921dedd` and `cf5f434`; only the working-tree edits differ.

### 1.3 The uncommitted edits (not touched)

Both files were modified after the N=10/11 runs ended (runs ended 2026-10-11 14:22; `collect_trace.py` mtime 10-11 21:17, `target_client.py` mtime 10-12 08:00), and the N=12 runs used the clean `921dedd` worktree, so **none of the certification data used these edits**. The driver guard only compares `HEAD`; it would not have detected them.

`target_client.py` (+14/-2):
1. Default `max_logprobs` for greedy decoding: `1` -> `2`. This value is passed to `LLM(max_logprobs=...)` and to `SamplingParams(prompt_logprobs=..., logprobs=...)`. The simulator constructs the verifier without an explicit value, so it is the effective default [code].
2. `_best_token_id`: from "first maximum in dict iteration order" to "smallest token id among entries with the maximal logprob" (plus an empty-dict check).

`collect_trace.py` (+13/-2): draft loading with `AutoConfig`, `sliding_window=None` if `use_sliding_window` is false, `attn_implementation="eager"`, and an explicit `attention_mask` in `generate`. It is invoked only by `scripts/run_collect_trace.sh`; the simulator's `DraftWorker` loads the draft model with default attention (earlier job logs show the `sdpa` sliding-window warning) and uses the `past_key_values` path, unchanged.

### 1.4 Could the verifier edit change results?

| Aspect | Effect | Basis |
|---|---|---|
| Target verification (accept/reject) | Changes only where the top logprobs tie; elsewhere identical | [code] |
| Token selection (correction token) | Same: tie -> smallest id instead of engine/dict order | [code] |
| Acceptance profile (alpha per gamma) | Can shift slightly if ties are not negligible; unquantified | [inferred] |
| Useful-token output | Indirectly, through accepted length and the committed token stream (which feeds later contexts) | [inferred] |
| Performance timing | `prompt_logprobs=k` is requested for **every prompt position** of every request, so returning 2 instead of 1 entries per position adds host-side Python work per call, and the new `_best_token_id` adds a per-position scan. The measured batch time is a wall time that includes this host work (section 4), so a timing change feeds directly into the simulated verifier service time | [code] + [measured] relation in section 4 |
| Tie frequency | Not measurable from existing logs: logprobs are not recorded. Detecting ties itself needs top-2 logprobs, i.e. the edited setting or an equivalent | [measured: no such column] |

The edit is the more faithful implementation of greedy argmax semantics, but it is a **different verifier semantic** from the one used for every certified number. It must not be silently mixed.

### 1.5 Recommendation: canonical revision and worktree strategy

1. **Canonical base = `main@680845e`** (simulator semantics identical to `921dedd`). Name the semantic generation `V1` (first-max tie-break, `max_logprobs=1`, amortized verifier service time).
2. Run from a **dedicated clean worktree** at that commit (new directory, not the main working directory, not mutated while jobs are queued). Tag the commit used for each campaign.
3. Leave the main directory exactly as is. Its owner should decide whether the two local commits get a branch (e.g. `hf-target-runtime`) so they are not lost, and whether the verifier edit becomes a branch `verifier-v2-tiebreak`.
4. If `V2` (the edited verifier) is wanted, treat it as a new semantic generation with its own run tags and a bridge experiment (tie-rate measurement and a paired A/B at identical seeds) before any cross-comparison. Record `verifier_semantics` in every run.
5. Add a hard guard: refuse to launch if tracked source files (`legacy/src`, `scripts`, `configs`) are dirty or HEAD differs from the run's recorded SHA.

## 2. Storage and quotas

### 2.1 Capacity and file counts (from `pawseyAccountBalance -storage`, 2026-10-18)

| Filesystem | Space | Files | Status |
|---|---|---|---|
| `/software` | project 194.8 GiB, 76.1% of project quota (user 165.8 GiB) | user **248,754 = 99.5%** (limit ~250,000) | headroom ~1,250 files |
| `/scratch` | project 122.0 GiB (user 121.3 GiB), 0.0% of project quota | user 169,505 = 17.0% | healthy, purgeable |
| `/home` | user **1024 MiB = 100%** | 3,124 = 31.2% | full |

- Earlier today the file limit was exceeded (250,378) and every job died in ~8 s because Slurm could not create its log file; no data was lost.
- Largest file consumers under `/software/projects/pawsey1257/quanla/` (files): `aucspec` 48,718 (of which ~40k are `.pythonuserbase-rocm-vllm` and `.venv-sim`), `pyuserbase` 21,817, `specdiff_cqb` 16,760, `pluto_pylibs_py310` 13,853, `pyuser` 13,790, `pluto_pylibs` 10,778, `specdiff_pylibs` 8,206, `specdiff_cqb_v3dev` 8,070 (205,762 files counted in total). Most of the count is Python library trees (not aucspec results); I did not check which project owns each.
- `/home` is full mainly because of `~/.cache/huggingface` (953 MB). Any job or tool that writes under `$HOME` can fail.

### 2.2 Per-run footprint

Each `(N, seed)` run creates 6 files in 2 directories: `*.csv` (~53 MB), `*.yaml`, and an `*_artifacts/` directory with `events.jsonl` (~40 MB), `rounds.parquet` (~8 MB), `run_config.json`, `summary.json`; about 100-150 MB and 6 files per run, plus one Slurm log (~2 MB). The certification run holds 32 runs = 192 raw files, 2.1 GB, 76 MB of logs. A multi-point frontier of several hundred runs adds thousands of files, more than the current headroom.

### 2.3 Raw-trace archives on `/scratch` (verified)

`/scratch/pawsey1257/quanla/aucspec_raw_archive/`: 12 tar files, 19.88 GiB total, created 2026-10-18 07:01-07:02, each checked at creation with `tar --compare` against the source (all `*_compare.err` empty; file counts on disk = in tar). Checked again today: all 12 were read in full to compute SHA-256 (manifest in `SHA256SUMS.txt`, first 16 hex digits below), are group-readable (`pawsey1257`) and have unchanged sizes. Total 19.88 GiB.

| Archive | SHA-256 (prefix) |
|---|---|
| gate_a_smoke_raw.tar | 8fcd72aa30ad6c6d |
| gate_b_x4p5_raw.tar | e3de0ab55bc7064f |
| gate_c_turbospec_vs_capacity_dpp_raw.tar | fb632be00382d591 |
| gate_d_small_frontier_raw.tar | 5f7308ac9a96b2b9 |
| budget100_small_frontier_raw.tar | c11ff9b9a75907e7 |
| budget100_x2p0_raw.tar | 2e5da89f546750b2 |
| budget100_x3p0_raw.tar | 5b5b50d3182ebb17 |
| budget100_x4p0_raw.tar | fc153702f3967985 |
| budget100_x4p5_raw.tar | 8210b9e890608b1f |
| budget100_x5p0_raw.tar | efe8e1f9f9e4e117 |
| budget100_x5p5_raw.tar | 9e3f310dc14a85e9 |
| budget100_x6p0_raw.tar | cc7fd0d0953d2f2d |

### 2.4 Retention and purge risk

- `/scratch` on Setonix is subject to a purge policy for files not accessed for a period; the only figure found (Pawsey workshop material and notices from 2022-2023, see Sources) is 30 days, **not confirmed against current documentation**. Treat the tars as temporary. Pawsey positions the Acacia object store for longer-term storage.
- The archive tars are the only copy of those raw traces.
- Not archived: raw of `budget100_x3_certification` (2.1 GB, still on `/software`), `results/capacity_frontier/raw` (legacy pilot, 430 MB), `results/exp1_frontier`, `exp5_alpha_hetero`, `auc_achievable_region`.
- In git (independent of scratch): configs, candidate caches (`candidate_cache.jsonl`, force-added), seed summaries, boundary CSVs, `run_config` copies for the certification run (config/hash/commit/fingerprint), and reports. **Raw per-round and per-batch data needed for roofline fitting are not in git.**

### 2.5 Proposed storage strategy for future GPU jobs (nothing applied)

1. Write raw artifacts and Slurm logs to `/scratch` (17% of file budget used, large quota), never to `/software`; keep on `/software` only compact outputs.
2. Pack each run's artifacts into a single `tar.zst` (or one parquet per run) at the end of the job: 1-2 files per run instead of 6.
3. Add a preflight check before submission: read the user file count, require `headroom >= expected_files + margin`, abort otherwise; also have the job touch its output directory first and fail loudly.
4. Mirror the compact bundle (caches, per-batch table, `run_config.json`, manifest with SHA-256) into git, and copy raw tars to Acacia (or another persistent project store) with the manifest.
5. Free file-count headroom with approval only: candidates are consolidating Python library trees (`pyuserbase`, `pyuser`, `pluto_pylibs*`, `specdiff_pylibs`) into a single archive/squashfs/container, and pruning `~/.cache/huggingface` to free `/home`. These are not aucspec files; none was touched.

## 3. Instrumentation availability

Legend: 1 = directly recorded, 2 = derivable from existing raw/compact data, 3 = missing. "Raw" = per-round CSV / `events.jsonl` / `rounds.parquet`; "compact" = `candidate_cache.jsonl` / `summary.json`.

| Quantity | Class | Where / how | Gap |
|---|---|---|---|
| Per-client useful-token rate | 1 | compact `per_client_rates`, `summary.json` `per_client_*`; raw `useful_tokens`, `client_receive_ms`, `in_measurement_window` | none |
| Aggregate goodput | 1 | compact `goodput_tps`, `summary.json` `goodput_tps` | defined as mean per-client rate x N in the cache |
| Speculative depth distribution | 2 | raw `gamma`, `selected_gamma`; compact has only mean gamma | add a gamma histogram to the compact output |
| Actual verifier batch size | 1 | raw `verification_batch_size`; compact mean | add distribution |
| Verifier-token work per batch (logical) | 1 | raw `verification_batch_token_cost` (sum of the logical per-request cost, about gamma+1 per request; definition not re-derived here) | none |
| Actual tokens processed per batch (context + gamma) | 2 | length of raw `context_token_ids` + `gamma` | not stored as a number; large list columns inflate the CSV |
| Measured verifier latency | 1 | raw `verification_batch_measured_service_ms` (wall time of the `generate` call incl. host work) | not split into GPU forward vs host post-processing |
| Simulated verifier service time | 1 | raw `verification_batch_service_ms` = `target_latency_ms` = measured / B | semantic must be documented (section 4) |
| Verifier context / sequence length | 2 | raw `context_token_ids` length | per-batch sum/max not stored |
| KV-cache usage, vLLM scheduler stats | 3 | not logged | needs instrumentation |
| Batch fill ratio | 1 | raw `batch_fill_ratio`; compact `mean_fill_ratio` | none |
| Deficit / server / device queues | 1 | raw `z_queue`, `server_queue`, `device_queue`, `server_queue_before/after`; compact slopes | quantiles not in compact |
| Accepted-token statistics | 1/2 | raw `accepted_length`, `committed_token_count`, `useful_tokens`, `alpha_hat`; `summary.json` `acceptance_rate`, `verification_tokens_per_useful_token` | per-gamma acceptance table is derivable only |
| Measurement duration | 1 | compact `extra.measurement_duration_s`; `run_config.json` `experiment` | none |
| Workload exhaustion | 1 | compact `extra.exhausted_clients` (client active span < 0.85 x window) | none |
| Per-request timeline | 1 | `events.jsonl` (draft/uplink/arrival/batch/downlink/complete, virtual and wall time) | none |
| Git SHA | 1 | `git_commit.txt`, `run_config.json` | no dirty flag, no diff hash |
| Config hash, population fingerprint | 1 | `config_hash.txt`, `population_fingerprint.txt`, per-round column | none |
| Software versions | 1 | `run_config.json`: torch 2.7.1a0, transformers 4.51.3, vLLM 0.9.1+rocm633, python 3.12.3, pandas 3.0.5, pyarrow 25.0.1 | none |
| GPU model, ROCm/driver version, node, Slurm job id | 3 | not recorded in the run files (GPU memory 63.98 GiB appears only in job logs) | needs instrumentation |
| `environment.json` | 3 (wrong host) | written at `init` on the login node (python 3.11.6, hostname `setonix-03`); no torch/vLLM/GPU | write from the job process |
| Verifier semantic flags (`max_logprobs`, tie-break rule, prefix caching, service-time model) | 3 | partly in `run_config.json` (`target_enable_prefix_caching: false`), the rest only implied by the commit | record explicitly |
| Top-1/top-2 logprob gap (tie detection) | 3 | not recorded | needs instrumentation (optional) |

### 3.1 Do the logs support independent roofline fitting and fixed-depth theory validation?

Roofline fitting from existing data: **partially, not cleanly.** Per-batch measured latency, batch size and logical token cost exist for every batch; context length is derivable. But:

1. **The simulated verifier service time is not the batch time.** In `verification_batcher.py` (fixed budget mode, no roofline model) the batch occupies the virtual server for `max(result.target_latency_ms)`, and `target_client.py` sets `target_latency_ms = elapsed_ms / len(results)`. On the N=12 seed-1 run (3,275 batches in the measurement window) the ratio `measured / modeled` is 1.00, 2.01, 3.01, 4.01, 5.01, 6.01 for B = 1..6 [measured]. A batch of 7 requests took ~407 ms of wall time but occupies the virtual server for ~58 ms. The theory's `T_v(B) = theta_0 + max(theta_m, theta_f * Gamma(B))` assumes the whole batch occupies the verifier for `T_v(B)`.
2. **Measured time vs batch size is affine with a large intercept:** `wall_ms ~ 129.6 + 44.4 * B` (mean per B: 172.8, 218.3, 264.9, 309.3, 352.2, 389.9, 407.1 ms for B = 1..7) [measured]. The intercept is large for a 7B model, which points to per-call host overhead rather than GPU time (not separated, [inferred]).
3. **The logical token cost is not the work the GPU does.** The target runs a stateless `generate` over `context + draft tokens` with prefix caching disabled and `prompt_logprobs` for every prompt position [code]. Mean logical cost per batch is 8.0 (max 31) while mean actual tokens per batch is 384 (max 1,277) [measured]. Correlation with measured time: logical cost 0.77, actual tokens 0.85, batch size 0.85 (these cannot be separated with this data) [measured].
4. **The knee is not exercised.** In the analysed run (N=12, seed 1) batches reach B <= 7 and logical cost <= 31, versus configured limits 32 requests / 100 tokens, so neither limit binds there [measured]; other runs were not analysed. The compute-bound regime of the roofline is not sampled by the frontier runs.
5. An older offline profile (`results/target_profile_summary.json`) covers only batch sizes {1, 4}, gamma {0, 2}, context {128, 512}; it cannot anchor a fit.

Fixed-depth theory validation: the per-round decomposition needed for `x = phi(gamma, alpha) / (gamma*tau_d + kappa*gamma/r + W_q + T_v + delta)` exists in the CSV (`draft_latency_ms`, `upload_ms`, `edge_service_ms`, `download_ms`, queue wait via `server_arrival_ms`/`batch_start_ms`, `accepted_length`, `gamma`, `rtt_ms`). Two things are missing for a clean test: a fixed-gamma controller run recorded under the canonical contract (a `fixed_gamma` option exists in the config; I found no fixed-depth run under this contract), and a decision on the `T_v` semantic of item 1 (otherwise the predicted `T_v` and the simulator's `T_v` differ by the factor B).

## 4. Proposed canonical experiment contract (frozen once approved)

### 4.1 Frozen configuration (values read from the certification run)

| Item | Value |
|---|---|
| Hardware | Setonix `gpu` partition, one node, 8 GPUs (`--gres=gpu:8`): target on `cuda:0`, drafts on `cuda:1-7`; GPU memory 63.98 GiB per device; node and GPU model must be recorded per run (currently not) |
| Software | module `pytorch/2.7.1-rocm6.3.3`; torch 2.7.1a0, transformers 4.51.3, vLLM 0.9.1+rocm633 (V0 engine fallback), python 3.12.3 |
| Target / draft | `Qwen/Qwen2.5-7B-Instruct` / `Qwen/Qwen2.5-1.5B-Instruct`, dtype auto, `gpu_memory_utilization` 0.70, `max_model_len` 4096, prefix caching off, eager off |
| Decoding | greedy (temperature 0), `delta_proposal` payload, lossless greedy verification, `max_new_tokens` 96 |
| Speculation depths | `gamma_max` 4, `gamma_choices` [0, 1, 2, 3, 4], controller `fixed_gamma` 4 for fixed-depth runs |
| Controller | `capacity_dpp`, V = 50, `min_interactivity_tps` set to the requirement x per run, Bernstein-censored learning, delta 0.05 |
| Verifier | scheduler `capacity_knapsack`, `budget_mode: fixed`, `verify_token_budget: 100`, `max_batch_size: 32`, `batch_wait_ms: 150`, `target_replication_factor: 1`, `utility_lambda: 0.25` |
| Verifier semantics | `V1`: first-max tie-break, `max_logprobs=1`, `prompt_logprobs` on all prompt positions, simulated service time = measured wall time / B |
| Wireless | iid block fading; RTT 15-100 ms; uplink 5-50 Mbps, downlink 10-100 Mbps; SNR ranges as in the config; shared 40 MHz uplink (square-root allocator) and 80 MHz downlink (queue-weighted allocator) |
| Draft execution | `shared_gpu_emulation`; draft token rate 20-60 tok/s, fixed latency 5-20 ms |
| Workload | GSM8K test, 1,200 questions (`dataset seed 42`, shuffle buffer 1000), `questions_per_client` 40, `population_n_max` 30 |
| Nested population | client i always receives `samples[i*Q:(i+1)*Q]`; fingerprint `7d873c2e94ab447b` |
| Measurement window | warmup 30 s, measurement 480 s, rate denominator = the exact configured window |
| Seeds | `simulation.seed` = seed id, contiguous 1..12; initial 5, wave escalation (e.g. 6-8, then 9-12), never beyond 12 |
| Classifier | exact SLO x (no tolerance factor); per-client Bonferroni-bootstrap bounds (2,000 resamples, rng seed 0), delta 0.05; feasible iff min LCB >= x, infeasible iff min UCB < x, else uncertain; shortest decisive prefix; K < 2 never certified; queue slopes never veto |
| Validity | any client whose active span < 0.85 x window invalidates the run |

### 4.2 Reproducibility checklist (per campaign)

- [ ] Clean worktree at a tagged commit; refuse to run if `git status` shows tracked source changes; record `git_dirty` and a hash of any diff.
- [ ] Config hash and population fingerprint written at init and re-verified by every job.
- [ ] Verifier semantic generation (`V1`/`V2`) stored in `run_config.json` and in the run tag; never pool across generations.
- [ ] Hardware record produced inside the job: node, GPU model, ROCm/HIP version, driver, Slurm job id, module list.
- [ ] Quota preflight passes (file headroom >= planned files + margin; `/home` writable).
- [ ] Compact outputs and checksum manifest committed; raw bundle copied to persistent storage.
- [ ] Measurement window length, seed list and wave rule fixed before launch; no 240 s and 480 s records mixed in one verdict.
- [ ] Results at or after `4b24977` only (same simulator path); earlier results are not pooled.

## 5. Blockers and decisions needed

| ID | Item | Needed from user |
|---|---|---|
| D1 | Fate of the uncommitted verifier edit and the two local HF commits (branch them, leave them, or discard) | decision |
| D2 | Is "simulated verifier time = measured wall time / B" intended? It defines the canonical system and decides whether theory validation compares like with like | decision |
| D3 | Which file-count relief is acceptable (library trees, `~/.cache/huggingface`), and whether to use Acacia for raw retention | approval |
| D4 | Confirm the current `/scratch` retention window with Pawsey before relying on it | check |
| B1 | File-count headroom (~1,250) is smaller than any plausible campaign | blocks GPU work |
| B2 | Provenance patch (dirty flag, GPU info, job-side `environment.json`) not yet implemented | blocks canonical runs |

## 6. Minimum proposed patch (not implemented)

Behavior-neutral instrumentation, to be proven by a short parity run (same seed, N=1: identical committed token streams; timing and queue fields are expected to differ run to run on GPU):

1. **Provenance:** in `run_config.json` add `git_dirty`, `git_diff_sha256`, Slurm job id, node name, GPU model, ROCm/HIP version, effective verifier parameters (`max_logprobs`, `greedy_only`, tie-break id, `verifier_semantics`, service-time model). Write `environment.json` from the job process. Add a launcher guard on dirty tracked source files.
2. **Per-batch table (compact):** one row per batch with `batch_id, B, logical_tokens, ctx_tokens_sum, ctx_tokens_max, gamma_sum, actual_tokens, wall_ms, generate_ms, post_ms, modeled_ms, forced_service`; `generate_ms` and `post_ms` need two `perf_counter` stamps around `llm.generate` and the post-processing loop in `verify_batch_sync`.
3. **Candidate summary:** add gamma histogram, batch-size histogram, per-gamma acceptance counts and queue quantiles to the compact cache record (new schema version, old records rejected as today).
4. **Storage:** artifacts of each run packed into one file; Slurm logs to `/scratch`; quota preflight script.
5. **Optional GPU micro-check, separate from campaigns:** count positions where the top-2 logprobs are equal on a fixed prompt set to quantify the tie rate behind D1.

## 7. Minimal next steps before any GPU execution

1. Answer D1 and D2; record the semantic generation name.
2. Free file-count headroom (D3) and re-check `pawseyAccountBalance -p pawsey1257 -storage`.
3. Create the clean canonical worktree at `680845e` (or the instrumentation commit once reviewed) and tag it.
4. Review and implement the patch in section 6 in a separate commit; run the parity check.
5. Freeze the contract in section 4 as `configs/canonical_v1.yaml` and launch only through the guarded launcher.

## Sources

- Pawsey, Acacia object storage workshop page (mentions a 30-day retention on Setonix `/scratch`): https://pawsey.org.au/event/acacia-object-storage-workshop-conduct-1/
- Pawsey technical newsletter (purge notices, 2022-2023): https://pawsey.org.au/technical-newsletter/
- These are dated and were not verified against current Setonix documentation (decision item D4).
