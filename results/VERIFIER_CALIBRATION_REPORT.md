# Verifier calibration report (Task B2)

One bounded GPU job, one node, no pricing enabled, no controller change, nothing merged. The cost model below is a **proposal for review**, not an accepted controller model.

## 1. Provenance

| | |
|---|---|
| Code | `24cb018761a9010ff603495280775279c0ee7310` on `research/verifier-batchwall`, worktree clean (`git_dirty=false` recorded job-side). The fitter was extended afterwards (acceptance-effect and design-check logic, same commit as this report); the harness that produced the data is unchanged. |
| Config | `configs/canonical_batchwall_smoke60.yaml` (file sha256 `c0894997…f4c8`; resolved-config sha256 `a00a57d0ff4b1e49…`); `service_time_model=batch_wall`, `verifier_price_mode=zero` (guard-enforced) |
| Verifier | V1 greedy, `max_logprobs=1`, tie-break `first_max_in_vllm_logprob_dict_order`; target `Qwen/Qwen2.5-7B-Instruct`, dtype auto, prefix caching off, eager off, `gpu_memory_utilization 0.7`, `max_model_len 4096`, seed from the simulator config; built through the same constructor arguments as the simulator |
| Software | vLLM 0.9.1+rocm633, torch 2.7.1a0, transformers 4.51.3, module `pytorch/2.7.1-rocm6.3.3` |
| Hardware | Slurm job 51101926, node `nid002146`, AMD Instinct MI250X (8 GCDs allocated, target on GCD 0), ROCm 6.3.3-74, HIP 6.3.42134, amdgpu 6.12.12 |
| Seed / schedule | seed 20261019; context pool sha256 `ec9d13e7…8f0d` |
| Data | raw per-call JSONL (2,027 lines) sha256 `16334ebc385db17ffccf696741dd0a48b8ea4b19b5bc1d214b947d47ba949308`; meta, fit hashes in `results/calibration_20261019/SHA256SUMS_raw_and_fits.txt`; original raw file kept on `/scratch/pawsey1257/quanla/calibration_raw/verifier_calib_20261019/`, a gzip copy (143 KB) is in the repository |

## 2. Context lengths

Final quantiles **p10/p50/p90 = 113 / 158 / 205 tokens**, from all 14,276 rounds of the canonical N=10 batch-wall smoke trace (400 GSM8K questions, 10 clients × 40, chat-template prompt plus the generated history up to 96 tokens). The V1 control trace gives 113 / 157 / 204; the N=1 trace (40 questions) gives 112 / 163 / 215 and its measurement-window-only values drift (124 / 182 / 235), so the earlier short-smoke quantiles (112/163/215) were close but I did not use them. Extremes: p99 = 246, max = 286 in the N=10 trace; the held-out set used lengths drawn uniformly from 100–230. Contexts were built from real prompts plus target-greedy history of exactly the target length (`context_length_quantiles_canonical_traces.json`). The calibration therefore does not cover prompts longer than about 290 tokens.

## 3. Calls, grid coverage and runtime

| | Count |
|---|---|
| Warm-up (3 per cell × 75) | 225 |
| Timed primary (20 per cell × 75) | 1,500 |
| Timed held-out heterogeneous (30 cells × 10) | 300 |
| Untimed engine warm-up | 10 |
| Verifier calls attempted / successful | 2,035 / 2,035 (no failure) |
| Preparation `generate` calls (two batched calls building 80 contexts and their proposals) | 2 |
| **Total target-model calls** | **2,037** (cap 2,500) |

Coverage: 5 batch sizes × 3 contexts × 5 depths = 75 primary cells, each with 3 warm-up and 20 timed repetitions, randomised per round (each timed round is a fresh permutation of the 75 cells); processed tokens per call from 113 to 2,560. Held-out: 30 cells, batch sizes 2/4/8/16, requests with distinct lengths (100–230) and mixed per-request depths 0–4.
Runtime: job elapsed **11 min 35 s** (06:56:25–07:08:00, requested 45 min); measurement span 580 s; preparation 6.2 s; the rest is model start-up. Raw data were fsync'd per call. No call was stopped by the 38-minute deadline guard.

## 4. Fitted cost model (full-batch wall time, never wall/B)

`T = θ0 + a·B + c·ΣL + b·Σγ` (ms), OLS on the 1,500 timed primary calls; 95% intervals from a block bootstrap over the 20 repetition rounds (1,000 resamples).

| Component | Estimate | 95% CI |
|---|---|---|
| Fixed batch overhead θ0 | 36.4 ms | [35.4, 37.8] |
| Per-request constant a | 5.81 ms | [4.99, 6.43] |
| Per context token c | 0.2058 ms | [0.2021, 0.2107] |
| **Marginal cost per draft token b** | **0.425 ms** | **[0.337, 0.508]** |
| Per-request total at context 113 / 158 / 205 | 29.1 / 38.3 / 48.0 ms | ±0.3 |

Generation time and host time were measured separately with a wrapper around `llm.generate`: host-side time (decode, rejection loop) is below 0.35 ms per call even at B=16, so the fit of `generate_ms` is identical to the fit of wall time; stamped batch wall and the outer timer agree within 0.02 ms.

## 5. Draft-token (speculative) cost

- Pooled within-round paired comparison, `(T(γ) − T(0))/(B·γ)` over 1,200 pairs: **0.390 ms per draft token per request, CI [0.346, 0.449]**; affine coefficient 0.425 [0.337, 0.508]. It is not indistinguishable from zero, but it is small: an extra draft token costs about 1% of a request's context cost at L = 158 and about twice a single context token.
- The cost is **not smooth**. Medians show a staircase in sequence length: at L = 158 (B=1) wall is 72.4 / 72.5 / 72.5 ms for γ = 0/1/2 and then 77.2 / 77.3 ms for γ = 3/4, i.e. a jump of about +5 ms (+6.6%, scaling with B: +40 ms at B=16) when the sequence length crosses 160; within a block the per-token cost is about 0.02–0.15 ms. At L = 113 and 205 the jumps are much smaller (largest +27 ms at B=16, γ=4). The affine coefficient is an average over this staircase. The per-γ paired means (0.54, 0.21, 0.54, 0.35 ms) are non-monotone for this reason.
- Acceptance outcome has no detectable effect on cost (coefficient on rejected draft tokens −0.21 ms [−0.98, 0.79]).

## 6. Train versus held-out error

| Evaluation | n | RMSE (ms) | MAE (ms) | MAPE | Median rel. err. | Mean residual |
|---|---|---|---|---|---|---|
| Train (all primary) | 1,500 | 31.9 | 7.6 | 3.1% | 2.4% | 0 |
| Temporal hold-out (fit rounds 0–13, test 14–19) | 450 | 33.9 | 7.4 | 3.0% | 2.4% | −0.2 |
| Leave B=16 out (extrapolation to B=16) | 300 | 11.9 | 10.0 | 1.6% | 1.4% | −2.2 |
| Independent heterogeneous set (mixed contexts and depths) | 300 | 50.2 | 7.6 | 2.3% | 1.6% | +1.1 |

RMSE is dominated by five stalls (single calls 2–3× the cell median: 604–1,131 ms against 222–414 ms, 0.3% of timed calls, at B = 4/8, scattered in time); median relative error of about 1.6–2.4% is the typical accuracy. Within-cell coefficient of variation: median 0.26%, maximum 41% (a stalled cell). Residuals by batch size have mean within ±2 ms (B=1: RMSE 2.9 ms; B=16: RMSE 10.4 ms). The model holds on contexts and per-request depths it never saw, so additivity across requests is supported.

## 7. Evidence on a roofline knee

**No evidence for a knee on this grid.** A broken-stick model on processed tokens W = ΣL + Σγ is worse than the affine model (ΔBIC = +11.0), its slope change is 0.019 ms/token with CI [−0.023, 0.024] (includes zero), the breakpoint CI [320, 2,560] spans the whole grid, and it does not improve the held-out RMSE. Within W ≤ 2,560 tokens (B ≤ 16 at these contexts) wall time is linear in batch work; I did not label any point memory- or compute-bound. This is only a statement about the covered range; it neither confirms nor excludes a knee at larger batch×context products. The block-boundary staircase of section 5 is a separate, discrete effect.

## 8. Comparison with independent smoke-test batch times

| Smoke run | Batches | Mean wall − predicted | Wall / predicted (median) |
|---|---|---|---|
| Batch-wall N=1 (B=1) | 1,466 | +1.9 ms | 1.02 |
| Batch-wall N=10 | 7,460 | **+86.5 ms** | **1.81** |

For N=10 the residual is almost constant across batch size (+84, +88, +90, +88, +84, +76 ms for B = 1…6): an additive per-batch overhead of roughly 85 ms that the isolated calibration does not contain; the earlier smoke-fit intercept (118 ms) minus the calibrated θ0 (36 ms) gives the same 82 ms, while the smoke per-request slope (42 ms) matches the calibrated per-request cost (about 39 ms at L = 158 with γ ≈ 2.5). N=1 matches the isolated calibration, so the extra time appears with the 10-client load, where seven draft workers occupy the other GCDs and the host runs ten clients' Python concurrently. Which mechanism (GPU/memory contention, host CPU or GIL contention with the asyncio event loop, thread hand-off) is **not identified**; no experiment separated them. Consequence: the calibrated model describes an isolated verifier; the simulator's measured batch wall under load is larger by a B-independent amount, so Q_s must keep using the measured wall (as designed) and any predicted cost used for pricing should be treated as a lower bound of the batch time seen in the simulator.

## 9. Status of the 7,460-batch coincidence

**Unresolved.** Evidence collected (`batch_count_coincidence_analysis.json`): both runs use the same 400 questions and end because every client finished its 40 questions (workload exhausted, not truncated); they differ in verifier requests (14,276 vs 14,931), committed tokens (38,400 vs 38,335), batch-size histograms (3,063/2,553/1,378/375/74/16/1 vs 3,202/2,176/1,234/613/188/46/1), batch-size sequences, virtual end times (1,558 s vs 1,202 s) and measurement-window rows (590 vs 800; batches touching the window 283 vs 384), so window filtering and workload size do not explain it. Batch ids are contiguous 0…7,459 in both and all timing invariants hold. I found no mechanism; a chance coincidence is possible but unquantified. A second seed would show whether batch counts vary between runs.

## 10. Other findings and issues

1. **Acceptance anomaly (important, unexplained).** The proposals were the target's own greedy continuation, yet only 48.7% of proposed tokens were accepted by `verify_batch_sync` (per-request full acceptance for γ = 1/2/3/4: 57% / 45% / 44% / 34%). The decode-path greedy token and the prefill-path argmax (from `prompt_logprobs` with `max_logprobs=1`) disagree far more often than ordinary bf16 near-ties would suggest. It does not affect cost (section 5), but it may affect acceptance statistics in simulator runs. I did not investigate further (no more GPU use is permitted in this task); candidates are numerical differences between the batched decode and prefill kernels, or how the best token is read from the top-1 plus prompt-token dictionary. Needs a dedicated check before acceptance rates are interpreted.
2. **Deviation from the brief on "malformed" cells.** Requiring every proposed token to be accepted would reject 1,223 of 2,025 calls and leave 40 of 75 cells without a single such call (all B=16 cells with γ ≥ 1). I therefore defined a malformed cell as one whose payload deviates from the design (batch size, context lengths, tokens presented); there were 0 violations. Acceptance is recorded per call and analysed as a covariate (no effect). I looked at acceptance rates before fitting; the choice rests on the mechanism (cost is computed on the presented tokens) and was then tested (coefficient above). A strict-acceptance fit is saved (`fit_strict.json`: 680 calls, no B=16, draft coefficient 0.90) and differs from the main estimate; with that reduced and unbalanced design (and the staircase) I do not trust it, but it shows the draft-token estimate is design-sensitive. Treat 0.4 ms as an average with a design-dependent uncertainty larger than the bootstrap CI.
3. **Causality logging** (decision-time and stamped server queue, theta_f, virtual decision time, max virtual finish folded into state) is implemented and unit-tested at the controller level, but no GPU simulator run has exercised the new CSV columns yet, so the end-to-end columns are unverified.
4. Rare stalls (about 0.3% of calls at 2–3× median; the smoke trace had a 1.8 s batch) are not modeled; treat them as noise in a priced controller or model them as a tail.

## 11. Recommendation

- **GO** to implement the calibrated-pricing code path as an opt-in mode behind `verifier_price_mode`, default `zero`, with unit tests and a separate baseline latency proxy, and to bring it to review. The measurements are internally consistent (held-out errors about 2%, a stable per-request and fixed-overhead decomposition, linear over the covered range).
- **NO-GO** for enabling it in any run or for rewriting past results until reviewed. Points that the review must settle: (a) the draft-token price is tiny relative to the per-request cost, so depth pricing will barely change γ and the effect will come through per-request and per-batch terms; (b) the isolated model under-predicts loaded simulator batches by about 85 ms per batch; (c) the staircase means a smooth marginal price is only an average; (d) the acceptance anomaly in item 10.1 and the 7,460-batch coincidence should be resolved or consciously accepted; (e) the κ_s/V grid in the smoke report should be re-checked against these magnitudes (per-request cost 29–48 ms, θ0 36 ms) before use.
