# AUCspec RQ1 — saturation exploration (C1.2, exploratory, one seed)

Exploratory and descriptive. One run-seed, hybrid emulator, lockstep batches. No significance, no certified capacity, no N\*(x), no ICA, no policy comparison.
The original pilot (`RQ1_PILOT_4a7e866`) and its pre-registered rules are untouched; the extension is a separate run tag with a separate config.
**Headline: N = 24 could not be run — the frozen verifier configuration ran out of GPU memory; N = 32 was never reached.** The saturation plateau therefore is **not located**.

## 1. Population extension: design and compatibility
* The old catalog (GSM8K *test*, 1,319 prompts) cannot supply 32 × 64 = 2,048 slots. Reusing prompts would have kept the first 16 clients comparable but would have created 729 duplicate
  slots (correlated clients). Chosen instead: **a fresh, single-dataset, fixed catalog from the GSM8K *train* split** (same loader, same shuffle seed 42, first 2,048 samples), 0 duplicates.
* Because the prompts differ, **the seed-1 pilot (N = 8…16) is a separate diagnostic and is NOT directly comparable** with the extension; the anchors N = 12, 8, 16 were re-run inside the extension.
* Static client profiles are unchanged (the first 16 clients' hardware fingerprint equals the pilot's; tested). Only the prompt catalog and `population_n_max` (16 → 32) differ (tested: exactly 4 config keys change).

| | original pilot | extension |
|---|---|---|
| split / slots | GSM8K test / 1,024 (16 × 64) | GSM8K train / 2,048 (32 × 64) |
| catalog hash | `628857be…eac7b6a` | `02e79304…7992b1` |
| distinct ids / prompts / duplicate slots | 1,024 / 1,024 / 0 | 2,048 / 2,048 / 0 |
| prompt overlap between the two | – | 0 |
| frozen hash enforced by the simulator at start | no (recorded only) | yes (`expected_catalog_hash`); also verified on CPU and in the run (`catalog_record.json` hash = frozen hash) |

Guards (CPU tests): exact nested prefixes (first 16 clients' prompt sequences identical for n_max 16 and 32, block hashes and prefix hashes), 64 actual questions for every client, unique ids, **refusal (`CatalogError`) instead of silent truncation** when the catalog is shorter than n_max × qpc, workload-exhaustion lifecycle unchanged. Pilot population fingerprint (config level) and per-run population fingerprints are in the run directories; the extension's are in `RQ1_EXT_28907a6`.

## 2. Provenance and cost
Code `28907a673fc01fb05ccf984ce7a8262673f91038` (clean, pushed); Setonix CPU tests at that SHA: 55 modules, all pass except the pre-existing `test_payload_interface` float-equality failure. Run directory `RQ1_EXT_28907a6`
(config hash `19202143a02262d7`, `--planned-runs 5`). Settings identical to the pilot (runtime H, γ = 2, FCFS, `batch_wall`, price zero, `causal_virtual_time`, `isolated_profile` 22.2 ms/token, 30 s + 120 s, early stop +10 s, 64 questions).
Job **51186048**, 75 min requested, ran **26 min 29 s = 0.441 node-h** (failed at N = 24). Budget ledger (`results/rq1_extension/budget_ledger.json`): job 51179457 0.523 + job 51186048 0.441 = **0.964 node-h of 3.2**; 2 jobs authorized and used, no follow-up submitted.
Order run: **12, 8, 16** completed (exit 0, all gates), **24 failed**, 32 not started (the job stops on the first failure, as designed).

## 3. Completed points (seed 1, extension population)
| N | client rate min / mean / max | x_min | Y(N) | N·x_min | x_min/mean | verifier util. | batch wall mean / p90 / max (ms) | batch-size histogram | physical draft GPU ms/token (vs 22.2) | min in-window rounds | wall/virtual s = ratio | ledger events / violations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 (1st run) | 3.76 / 3.89 / 4.02 | 3.758 | 46.69 | 45.1 | 0.966 | 0.65 | 463 / 552 / 1273 | {12: 169} | 138 (+521 %) | 169 | 313/160 = 1.95 | 6018 / 0 |
| 8 (2nd run) | 5.05 / 5.17 / 5.31 | 5.050 | 41.38 | 40.4 | 0.976 | 0.53 | 288 / 338 / 705 | {8: 223} | 136 (+513 %) | 223 | 265/160 = 1.65 | 5426 / 0 |
| 16 (3rd run) | 3.47 / 3.55 / 3.65 | 3.467 | 56.73 | 55.5 | 0.978 | 0.68 | 532 / 670 / 1146 | {16: 154} | 142 (+538 %) | 154 | 345/160 = 2.16 | 6884 / 0 |

Validation: virtual draft latency 22.2 ms/token exactly (isolated profile); 64 questions assigned per client, no client exhausted or unverifiable, every sample reached the 96-token cap (0 EOS), clients stopped by the early stop at ≈ 160 s; 0 causality violations; no runtime fallback (guard did not raise); real/virtual ≤ 2.16 ≤ 3. Every batch again contained all N clients (lockstep).
Not audited separately: duplicate/dropped-request counting beyond the ledger check and the token-accounting identity (same limitation as the pilot report).

Pre-registered ratios on these exploratory numbers (informative only, not a test): Y(16)/Y(8) = **1.37** (< 1.5, but not ≤ 1.10), x_min(16)/x_min(8) = **0.69** (not ≤ 0.65). Y(12)/Y(8) = 1.13, Y(16)/Y(12) = 1.22 — goodput is **still rising at N = 16** and verifier utilisation is 0.68; no plateau is visible. The seed-1 pilot gave 1.67 and 0.83 for the same ratios with a cold, anomalous N = 8; the two sets differ in prompts, order and run, so they must not be pooled.

## 4. N = 24 failure (preserved evidence)
* Cause: `torch.OutOfMemoryError: HIP out of memory. Tried to allocate 4.08 GiB. GPU 0 … 3.82 GiB is free … 52.64 GiB allocated by PyTorch`, raised inside the **vLLM sampler** (`sampler.py` `get_logprobs → _get_ranks`) during `verify_batch_sync` for a batch of 24 requests (batch id 82; the console shows 83 batches × 24 requests completed before it). The engine runs with `target_gpu_memory_utilization 0.7`, chunked prefill disabled, prefix caching off (engine config line in the log).
* Interpretation (limited): the frozen verifier configuration cannot take a 24-sequence batch whose prompt-logprob ranking needs an extra ≈ 4 GiB on the target GCD. I did **not** measure the number of tokens in that step or whether vLLM split the prefill across steps (no per-step logging; the rounds CSV is written only at run end, so the N = 24 data are lost except the console log). **Nothing is known about N = 32 and no claim about 32 simultaneous sequences is made.**
* Per the instructions no guard was relaxed and no retry was made. Fixing this means changing a runtime/engine parameter (lower `target_gpu_memory_utilization`, a token cap per step, or a smaller `max_batch_size` that splits N ≥ 24 into several batches) — each changes what is being measured and needs your decision. Evidence: `results/rq1_extension/n24_oom_evidence.txt`.

## 5. Batch-wall tails and the N = 8 order question (descriptive; `slow_batch_diagnostics_seed1.json`, `ext_slow_batch_diagnostics.json`)
Slow batch = wall > max(600 ms, 2 × median).
| job / N | position in job | batches | median ms | slow batches | in warm-up / window | consecutive-run lengths / lag-1 autocorr |
|---|---|---|---|---|---|---|
| pilot N = 8 | 1st | 243 | 286 | **45** | 8 / 36 | bursts up to 15 / 0.62 |
| pilot N = 12 | 5th | 242 | 416 | 0 | – | – |
| pilot N = 16 | 6th | 199 | 542 | 3 | 0 / 3 | one burst of 3 |
| pilot N = 1 / 4 | 2nd / 4th | 459 / 379 | 74 / 160 | 13 / 4 | 1 / 0 | bursts up to 6 / 3 |
| ext N = 12 | **1st** | 231 | 402 | **11** | 0 / 11 | autocorr 0.81 |
| ext N = 8 | 2nd (hot) | 301 | 279 | 3 | 0 / 3 | – |
| ext N = 16 | 3rd | 202 | 539 | 1 | 0 / 1 | – |

Observed: slow batches arrive in **bursts** of consecutive batches (autocorrelation 0.45–0.81), occur at several N, are spread through the run (their share in the warm-up is about the share of time spent there), and slow batches had somewhat larger context at N = 8 (1,474 vs 1,263 processed tokens) that does not account for a 2–3× slowdown. The **first run of each job** showed the most slow batches (45 and 11) and the hot-start N = 8 had 3, with hot-start N = 8 x_min 5.05 vs 3.98 in the cold pilot run — **an order/warm-up effect is observed, but with one job per order, different prompts and N confounded with position, its cause (cold caches/JIT, clocks, interference) is not established.** Not asserted.

## 6. Questions
* **Plateau by N = 24 or 32?** Unknown: N = 24 failed (memory) and N = 32 was not reached. Up to N = 16 there is no plateau (Y 41.4 → 46.7 → 56.7).
* **Recommended range for later multi-seed measurements (within frozen H, no engine change):** N ∈ {4, 8, 12, 16} (add 1–2 as anchors if wanted), each seed with a **discarded burn-in run first** (or randomised order) so that no scientific point is the first run of a job. This characterises the pre-saturation regime and the per-client floor decline within ONE population (extension: x_min 5.05 / 3.76 / 3.47 at N = 8 / 12 / 16; pilot population: 7.76 at N = 1) — descriptive, populations not pooled; it cannot answer where goodput saturates.
* **If saturation must be located** (needs your decision): either change the target engine memory/step settings or cap `max_batch_size` (e.g. 16, so N = 24/32 become several sequential batches — a different, but physically natural, verifier batching limit), validate on one short run, then re-run N = 24, 32 with the same gates.
* **Cost of repeating the informative points:** per seed ≈ burn-in (~6 min) + N = 4, 8, 12, 16 (≈ 5.7, 7, 8, 8.6 min incl. startup) ≈ 0.55 node-h; 3 seeds ≈ 1.6 node-h; remaining budget 3.2 − 0.964 = 2.236 node-h, so 3 seeds fit with ≈ 0.6 node-h to spare. Extending to N = 24/32 would add ≈ 0.25 node-h per seed after the memory question is resolved.

## 7. Limits
Exploratory, one run-seed per job; population differs from the pilot; hybrid real/virtual emulator with an isolated draft profile (physical draft contention is +513…538 %, removed from virtual time by construction); homogeneous hardware with lockstep full-N batches; fixed γ = 2; 0 EOS; no N ≥ 24 data; the N = 8 order effect is observed but not explained. Nothing here changes the status of the unsupported claims in the C0/C1 audits. Seeds 2/3 were not started.

Artifacts: `results/rq1_extension/{prompt_catalogs.json, ext_seed1_compact.json, ext_slow_batch_diagnostics.json, slow_batch_diagnostics_seed1.json, n24_oom_evidence.txt, budget_ledger.json}`; raw CSV/ledgers on `/scratch/pawsey1257/quanla/canonical_raw/RQ1_EXT_28907a6/` (not in git).
