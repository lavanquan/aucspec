# Greedy verifier correctness audit (Task B3)

**Headline.** The ~49% acceptance of target-generated proposals is **not** a logprob-indexing error, **not** a `max_logprobs=1` artefact, **not** EOS handling and **not** (mostly) tie-breaking. In this vLLM 0.9.1+rocm633 build, **multi-sequence batched execution gives wrong target logits for a large share of sequences**: against an independent fp32 reference, verification at B=1 agrees on 99–100% of positions, at B=4 on 77–84%, at B=16 on 48–53%, and batched decoding is wrong as well. The root cause inside vLLM/ROCm is **not identified**; the evidence says the damage is in the batched forward pass, not in how `verify_batch_sync` reads its output. Nothing in canonical V1 was changed. **Consequence: V1 greedy verification with B > 1 does not preserve target-only output semantics in this environment, and every prior simulator result with B > 1 is affected to an unmeasured degree.** I recommend no further campaigns until this is resolved (section 9).

## 1. What was done

- Static audit of `verify_batch_sync`, `_best_token_id`, `_logprob_dict`, `_generate_one_greedy_token`, the calibration proposal code and the acceptance accounting; 10 CPU tests with synthetic logprob data (`legacy/tests/test_greedy_verifier_audit.py`); full suite 269 passed before the GPU job.
- One GPU diagnostic job (51109736, node `nid002228`, 1 node, requested 20 min, elapsed 2 min 40 s), code `b25de96` (clean tree), config `configs/canonical_batchwall_smoke60.yaml`. **189 target-model calls** (cap 200): 1 batched decode that reproduces the calibration pool, 4 B=1 decodes, 126 top-2 prefill calls (B = 16/4/1 × γ = 1/2/4 over 32 contexts), 54 calls of the unmodified `verify_batch_sync` with k=1 sampling (exact V1 request settings), 2 k=1 dictionary calls, 2 independent fp32 HuggingFace forward passes (GCD 1). Diagnostic engine: vLLM `max_logprobs=2` (canonical V1 is 1); V1 behaviour was reproduced with `prompt_logprobs=1`/`logprobs=1` through the unmodified method. Contexts: the 16 L=158 and 16 L=205 calibration pool contexts, regenerated with the same RNG; the regenerated pool reproduced the calibration job's per-slot first-token outcomes in 31/31 comparable slots.
- Raw diagnostic data (gzip) and four compact summaries are in `results/audit_20261019/`; raw original is on `/scratch/pawsey1257/quanla/greedy_diag/greedy_diag_20261019/`.

## 2. Static and CPU findings (Stage 1)

1. **Indexing is correct.** `prompt_logprobs[len(context)+offset]` is the distribution for the candidate token at that position (entry `i` describes prompt token `i` given tokens `< i`; index 0 is `None`). Measured: with the correct index the B=1 verifier agrees with fp32 on 99.2–100% of positions, whereas reading index −1 or +1 matches the proposal in only 3.9% / 2.1% of cases. CPU tests show that data placed one index earlier is not read.
2. **Dictionary handling.** vLLM returns top-k plus the observed prompt token: dict sizes were 1–2 for k=1 and 2–3 for k=2. `max` over the dict is the true argmax. Measured: the exact-V1 acceptance computed from k=1 equals the acceptance computed from the k=2 dictionaries of the same run in 100% of requests for every (B, γ) cell (9 cells), so `max_logprobs=1` is not hiding the argmax.
3. **EOS/truncation.** CPU tests: an accepted EOS ends verification and sets `reached_eos`; `remaining_tokens` truncates the commit but not `accepted_length`. The pool excluded any EOS in the first h+4 generated tokens, so EOS cannot explain the rejections.
4. **Tie rule (documented, but not the cause).** V1 uses the first maximal entry in dict insertion order; a CPU test shows an exact tie is resolved by dict order, not by token id. Measured at B=16, γ=4: of 72 mismatching positions, **0 were exact ties** and only 10 had a top-1/top-2 gap below 0.2 nats; 62 were clear disagreements. The bonus token comes from a different argmax implementation (vLLM's sampler), so V1 mixes two argmax implementations; this is a semantic wart but not the 49%.
5. **Sampling equivalence.** Proposal generation (`temperature 0, top_p 1`) and the verifier sampling are equivalent; the decode path's reported top-1 logprob equals the sampled token in 100% of cases, i.e. each path is self-consistent.
6. **Calibration acceptance accounting is correct** (CPU test: full acceptance gives `accepted_length = γ`, `check_realized` true; EOS or a reject gives false).

## 3. GPU diagnostic results (Stage 2)

**Reproduction rate: unmodified `verify_batch_sync`, k=1, fraction of requests**

| B (requests) | γ=1 full-accept | γ=2 full-accept | γ=4 full-accept | γ=4 first-token |
|---|---|---|---|---|
| 16 (32) | 46.9% | 31.3% | 18.8% | 50.0% |
| 4 (32) | 56.3% | 46.9% | 34.4% | 59.4% |
| 1 (8, hand-picked) | 62.5% | 62.5% | 37.5% | 62.5% |
| Truth under fp32 for these proposals | 65.6% | 59.4% | 40.6% | 65.6% |

For γ = 4 the verifier's accepted length equals the fp32 truth for 100% of requests at B=1, 84.4% at B=4 and 65.6% at B=16.

**Verifier (prefill) versus the independent fp32 oracle, per position**

| B | argmax equals fp32 argmax (γ=1 / 2 / 4) | median \|logprob error\| of the proposed token | p90 | max |
|---|---|---|---|---|
| 1 | 100% / 100% / 99.2% | 0.004–0.008 nats | 0.06–0.10 | 0.35 |
| 4 | 84.4% / 82.8% / 77.3% | 0.008–0.020 | 3.6–5.5 | 14.8 |
| 16 | 53.1% / 50.0% / 47.7% | 1.0–1.1 | 8.9–9.5 | 14.2 |

- **Whole-sequence failures.** Within a batch a sequence is either right at all positions (0 of 4 wrong) or wrong at all (4 of 4, sometimes 3). At B=4, the third sequence of every group (8 of 8 groups) is wrong in all four positions, and one other sequence is wrong at one position; at B=16 slots 2, 4, 5, 7–9 and 11–14 are wrong in both groups (slot 1 additionally in the first group). Sequences are all of equal length, so this is not a padding or length effect.
- **Not row misattribution:** none of the 82 wrong B=16 positions equals another sequence's correct (B=1) dictionary.
- **B changes the result:** B=16 vs B=1 same-argmax rate 48–53%, maximum logprob difference 11–15 nats; B=4 vs B=1: 25.8% of positions differ. Changing the candidate length at fixed B also changes results (γ=1 vs γ=4 at B=16: logprob differences up to 3.2 nats, argmax equal in 84–86%), consistent with batch layout dependence.
- **The corruption is in the forward pass, not the logprob extraction.** The sampler's own next token after the candidates (the bonus token) equals the fp32 argmax in 97–100% (B=1), 75% (B=4) and 41–53% (B=16).
- **Decode is affected too.** Of the proposals made by the batched (60-sequence) decode, only 71.9% are the fp32 argmax (fp32 rank ≤ 5 for 91.4%; the median error of the reported sampled logprob is 0.14 nats, maximum 18.7). Re-decoding four of the contexts at B=1 produced a different history in all four, i.e. the decode path is not batch-invariant either.
- **Ties do not explain it:** exact ties 0/72; near ties (gap < 0.2) 10/72; the fp32 top-2 gap at the 72 mismatches has median 1.51 nats (range 0.06–18.3); the verifier's argmax is the fp32 argmax in only 9 of the 72 mismatches, the proposal in 40.

## 4. Root cause and open hypotheses

**Established:** the discrepancy is produced by batched forward passes in this vLLM/ROCm build. B=1 is accurate; errors grow with batch size, are per-sequence rather than per-position, depend on slot and batch layout, and affect decode, prefill logprobs and the sampled next token alike. The proposal-generation job (a 60-sequence decode) therefore produced proposals that are themselves wrong for 28% of positions, and the B>1 verifier then judged them with its own errors; both effects add up to the ~49% acceptance.
**Not established (unresolved hypotheses):** (a) a defect in the ROCm multi-sequence attention path used here (all job scripts set `VLLM_USE_TRITON_FLASH_ATTN=0`, which selects a different attention implementation than the default); (b) a metadata/KV-block issue in the V0 engine's batched scheduling; (c) a data-type or kernel problem specific to bf16 on MI250X. Cross-sequence contamination fits "whole sequence right or wrong" but was not tested. None of this was probed because the audit allowed one bounded job.

## 5. Does V1 greedy verification preserve target-only output semantics?

**No, not for B > 1 in this environment.** At B=1 the verifier agrees with an independent fp32 oracle (99–100%); at B=4/16 it disagrees at 23–52% of positions, with logprob errors up to 14 nats, so acceptance decisions, correction tokens and bonus tokens are wrong for a large fraction of requests. A speculative-decoding output assembled from these verdicts is not the target's greedy output. The repository's `validate_correctness.py` cannot detect this because its "target-only" reference is generated through the same verifier. Past simulator runs (batch size about 2 on average at N=10, up to 7) were affected; the direction and size of the bias on goodput, α estimates and N*(x) is **not measured**.

## 6. Can the calibration dataset be used for timing? (Stage 3)

1. Cost of processing the submitted tokens is independent of the accept/reject outcome: the model receives the same payload and computes the same shapes (acceptance coefficient on rejected draft tokens −0.21 ms, CI [−0.98, 0.79]; host time < 0.35 ms).
2. Therefore the timing observations remain valid and I did not discard any: they describe the wall time of this exact (numerically defective) batched configuration. The 0.425 ms/token coefficient (paired 0.39 ms) stays meaningful as a **descriptive timing coefficient of the current configuration**.
3. Why the strict full-acceptance subset gave 0.90 ms: fully accepted calls are those whose sequences all happened to sit in "good" slots, which is a function of batch size and slot; the subset has no B=16 calls and 40 of 75 cells without any call, so the design is unbalanced and the draft term is confounded with batch size; the staircase in sequence length (section 5 of the calibration report) makes the estimate design-sensitive. It is a selection artefact, not a different physical cost.
4. **Caveat that matters:** if the correctness problem is fixed by a different attention backend, dtype, eager mode or engine, batched wall times will probably change. The calibrated model must **not** be called controller-ready until the underlying correctness issue is understood and the calibration repeated on the fixed configuration.

## 7. The 7,460-batch coincidence (existing logs and source)

Status: **no mechanism found; consistent with chance.** The source has no batch cap (no constant 7,460; batches are created per `_process` call), both runs end because all 40 questions of every client were completed, and the two runs differ in requests (14,276 vs 14,931), committed tokens, batch-size histograms, virtual end time and window rows. Across the 12 certified V1 N=10 runs (seeds 1–12) the batch count ranges 7,325–7,489 (mean 7,390, std 50), so 7,460 lies inside the normal range and two independent runs coincide at a given value with probability about 1/(2√π·50) ≈ 0.6%; no two of the 12 certified runs share a count. A chance coincidence at that probability is not excluded, but it is not proven either.

## 8. The +86.5 ms per-batch discrepancy (unresolved)

Existing logs only (`smoke_residual_audit.py`): relative to the isolated calibration model the loaded N=10 batches are +86.5 ms (batch-wall run) and +92.0 ms (V1 control run) longer on average, 5–95% range about 25–140 ms, almost independent of batch size (correlation 0.05–0.09), decreasing over virtual time (thirds: 90 / 93 / 76 ms, correlation −0.22), while N=1 matches the calibration (+1.9 ms). So it appears with the 10-client load and is not specific to `batch_wall`. I did not absorb it into the fitted constant. Candidate mechanisms (GPU/memory contention with seven draft workers, host CPU/GIL/event-loop effects, thread scheduling, differences in the batched path under load) are untested; no cause is claimed. A caution: since batched execution is itself unreliable here, load-dependent behaviour of the same path may be part of it.

## 9. Required follow-up (needs your approval; nothing run)

1. **Stop** further capacity, baseline or calibration campaigns on this configuration; results with B > 1 are not trustworthy as speculative-decoding outputs.
2. A small bounded GPU matrix to locate the defect, using the existing diagnostic script on B=16/γ=4 and B=1: attention backend (`VLLM_USE_TRITON_FLASH_ATTN` 0/1, `VLLM_ATTENTION_BACKEND` alternatives), `enforce_eager`, dtype (bf16/fp16), V0 vs V1 engine, `max_num_seqs`, chunked prefill; each cell checked against the fp32 oracle (about 30–50 calls per cell).
3. Until fixed, any verification used for experiments should be validated against the fp32 oracle, and a fix should be judged by agreement ≥ 99% at B = 16, not by acceptance rate.
4. Re-assess the certified V1 results as "batched-verifier-defective" once the extent is known; keep them preserved, do not rewrite.
5. After a correct configuration exists: repeat the verifier calibration, then reconsider pricing.

No implementation change to the canonical verifier, controller or pricing is proposed in this task; the code added here is diagnostic only.
