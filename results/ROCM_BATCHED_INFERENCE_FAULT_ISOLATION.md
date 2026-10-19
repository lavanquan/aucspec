# ROCm batched inference fault isolation (Task B4)

One bounded GPU job, no capacity/baseline/pricing/calibration work, nothing merged, canonical verifier untouched.

## 1. Summary

- **Root cause (demonstrated):** in this environment vLLM 0.9.1's ROCm backend runs prefill through its **naive SDPA fallback** (`flash_attn` is not installed and the Triton FA path is disabled), and that fallback receives the **cumulative offsets** `query_seq_start_loc = [0, l0, l0+l1, …]` where it expects **per-sequence lengths**. For a prefill step with three or more sequences, later sequences are therefore computed together with earlier ones (cross-sequence attention contamination) and some slots are wrong. Evidence: (a) introspection of the running engine (`use_naive_attn=True` in all 28 layers); (b) a CPU-only minimal reproduction that imports a verbatim copy of vLLM's function (AST-identical to the installed source) and reproduces exactly the wrong slots seen on the GPU; (c) the GPU slot pattern is the same in two different batch orders.
- **No tested configuration passes the correctness gates** (A, C, D, E identical failure; B and F could not run). **NO-GO** for every tested runtime.
- **Why Triton FA / V1 could not be tested:** a `triton 3.8.0` in the project's `PYTHONUSERBASE` shadows the `triton 3.3.1` that torch 2.7.1 ships; the Triton flash-attention kernel does not compile with it (B) and the V1 engine's inductor import fails (`triton_key`) (F).
- **Smallest next step (needs approval):** a single short GPU run of candidate **G** = current build + a runtime wrapper that turns the offsets into lengths (`scripts/rocm_sdpa_runtime_patch.py`, prepared, not applied) and candidate **H** = Triton 3.3.1 on the path. See section 9.

## 2. Provenance and what was run

| | |
|---|---|
| Job | Slurm 51128363, node `nid002240`, 1 node, 7 min 56 s of the 45 min limit |
| Code | `79eced35dff7223594f17221cfd5af707c5db871` on `research/verifier-batchwall`, clean tree (recorded job-side); config `configs/canonical_batchwall_smoke60.yaml` (resolved sha256 `a00a57d0ff4b1e49…`) |
| Hardware / software | AMD Instinct MI250X, ROCm 6.3.3-74, HIP 6.3.42134-a9a80e791, amdgpu 6.12.12; vLLM 0.9.1 (editable, `+rocm633`), torch 2.7.1a0+gite2d141d, transformers 4.51.3; target `Qwen/Qwen2.5-7B-Instruct`, prefix caching off, `gpu_memory_utilization` 0.7, `max_model_len` 4096; diagnostic engines use `max_logprobs=2` |
| Engine selected | V0 `vllm.engine.llm_engine.LLMEngine` in every successful run (log line: "rocm is experimental on VLLM_USE_V1=1. Falling back to V0 Engine"); backend `ROCmFlashAttentionBackend`/`ROCmFlashAttentionImpl`; chunked prefill off, `max_num_batched_tokens` 4096, `max_num_seqs` 256 |
| Frozen inputs | 32 dev contexts (length 158) and 32 held-out contexts (length 205), each = chat-template prompt + the first tokens of the GSM8K **gold** solution (no model generation); token-id sha256 per context in `frozen_contexts.json.gz`; identical ids, positions and sampling arguments in every comparison |
| References | (1) vLLM B=1; (2) independent **HF fp32** (eager attention, equal lengths, no padding) giving the greedy 4-token continuation and top-2 logprobs at the 4 candidate positions |
| Target-model calls | **194 of 400** (10 HF forward passes + 46 vLLM calls × 4 configurations; B and F failed before any call) |

Configurations (fresh process each; order and bases chosen by sequential fault isolation, no Cartesian product):

| # | Config | Outcome |
|---|---|---|
| A | `VLLM_USE_TRITON_FLASH_ATTN=0`, bf16, graphs | runs; reproduces the failure; **naive SDPA attention active** |
| B | `VLLM_USE_TRITON_FLASH_ATTN=1` | **fails to initialise**: `CompilationError` in `attn_fwd` (`AttributeError: '_aggregate_type' object has no attribute 'element_ty'`) with Triton 3.8.0 |
| C | A + `enforce_eager` (B had failed, so A was the base) | runs; numerically identical to A |
| D | A + `VLLM_ROCM_CUSTOM_PAGED_ATTN=0` | runs; identical to A |
| E | A with fp16 | runs; same pattern |
| F | A + `VLLM_USE_V1=1` (+ single process) | **fails to initialise**: `BackendCompilerFailed … ImportError: cannot import name 'triton_key' from triton.compiler.compiler` (user-site Triton 3.8.0 vs torch 2.7.1's inductor) |

Not tested, deliberately: other `VLLM_ATTENTION_BACKEND` values (TORCH_SDPA, XFORMERS, FLASH_ATTN). In this build `rocm.py` maps every non-`ROCM_FLASH` request to `ROCmFlashAttentionBackend` and only logs that the choice "is not supported in AMD GPUs", i.e. it would silently fall back to the same kernel; AITER is disabled (`VLLM_ROCM_USE_AITER=False`). The only distinct ROCm alternatives are the Triton FA kernel (B) and CK `flash_attn` (package not installed: `find_spec("flash_attn")` is `None`).

## 3. Correctness agreement by batch size (dev set, config A; C and D identical)

Per candidate position; "gap ≥ 0.5" keeps positions where the fp32 top-2 margin exceeds 0.5 nats, so bf16 near ties cannot be blamed (123 of 128 positions qualify).

| B | prefill argmax = fp32 (gap ≥ 0.5) | prefill argmax = vLLM B=1 (48 positions) | median / p90 / max \|logprob error\| vs fp32 |
|---|---|---|---|
| 1 | **100%** (46 positions in the 12 B=1 contexts) | n/a | ~1e-6 / 0.017 / 0.043 |
| 4 (identity order) | 80.5% | 75.0% | 0.0005 / 6.1 / 21.8 |
| 4 (shuffled) | 80.5% | 89.6% | 0.0004 / 5.7 / 14.1 |
| 16 (identity) | 43.9% | 43.8% | 1.85 / 10.6 / 22.2 |
| 16 (shuffled) | 46.3% | 41.7% | 2.22 / 9.2 / 14.1 |

fp16 (E): B=4 79.7–80.5%, B=16 44.7–48.8%: the same structure. **Gates (≥ 99% at B=4 and B=16): failed in A, C, D, E.** Held-out contexts were therefore never opened for any configuration (the in-process rule only evaluates them after the dev gate passes).

## 4. Decode versus prefill

| | greedy 4 tokens = fp32 continuation | first decode token = prefill argmax on the same context |
|---|---|---|
| B=1 (4 contexts) | 100% | n/a |
| B=4 (16 contexts) | 75.0% | 81.3% |
| B=16 identity (32) | 37.5% | 71.9% |
| B=16 shuffled (32) | 34.4% | 62.5% |

Decode and prefill disagree on the same context once the batch is larger than one, and decode is affected too. The decode shares are exactly the shares of slots that the prefill model leaves uncontaminated (3 of 4 = 75% at B=4, 6 of 16 = 37.5% at B=16), which is what you would get if the prompt KV of contaminated sequences were wrong and decode merely inherited it. That inheritance is a **hypothesis** consistent with the numbers; I did not run the isolating test (B=1 prefill followed by batched decode).

## 5. Batch-slot permutation evidence

At B=16 the wrong rate by position in the batch is **exactly 0% or 100%**, and identical in the identity order and in a random permutation:

- always wrong: slots 2, 4, 5, 7, 8, 9, 11, 12, 13, 14; always right: slots 0, 1, 3, 6, 10, 15 (both orders);
- only 2 of 32 contexts kept the same slot after shuffling, yet the wrong set moved with the slot, so **wrongness is a property of the batch slot, not of the input** (the "input-specific" fraction of 0.43 that the analysis prints is just the chance overlap of two sets that each cover 10 of 16 slots);
- at B=4 the third sequence is the wrong one in both orders.

The code-derived model (`rocm_fi_lib.naive_sdpa_status`) predicts exactly these sets, and also reproduced them for the earlier audit data (31 of 32 contexts; the one exception is a 3-of-4 case on a near tie).

## 6. Minimal reproduction (CPU only, no GPU, no vLLM engine)

`scripts/repro_rocm_naive_sdpa_bug.py` copies vLLM's `_sdpa_attention` verbatim (checked: function body AST-identical to `vllm/attention/backends/rocm_flash_attn.py` in the installed checkout), passes it what the backend passes (`query_seq_start_loc`) and compares with a per-sequence causal reference. Output of the CPU job 51131425 (23 s):

```
B=1 wrong slots []        B=2 wrong slots []
B=3 wrong slots [2]       B=4 wrong slots [2]
B=16 wrong slots [2, 4, 5, 7, 8, 9, 11, 12, 13, 14]    (max error up to 3.5)
with per-sequence lengths as the argument: all slots match the reference
```

For an upstream vLLM/ROCm report: the call in `ROCmFlashAttentionImpl.forward` passes `query_seq_start_loc` to `self.sdpa_attn_func`, whose loop does `end = start + seq_len`; conditions: ROCm, `VLLM_USE_TRITON_FLASH_ATTN=0`, `flash_attn` not installed, prefill step with ≥ 3 sequences. (Whether newer upstream releases already fix it was not checked.)

## 7. Causes: demonstrated versus hypotheses

| Factor | Status |
|---|---|
| Attention backend/implementation (naive SDPA fallback with offsets-as-lengths) | **Demonstrated** as the source of the wrong prefill logits (introspection + exact slot prediction + CPU reproduction) |
| Eager vs graph execution | No effect (C identical to A) |
| Custom paged attention | No effect (D identical to A) |
| dtype bf16 vs fp16 | No effect on the defect (E same pattern) |
| V0 vs V1 | **Untested**: V1 cannot initialise here (Triton mismatch) |
| Triton FA kernel | **Untested**: does not compile with the shadowing Triton 3.8.0; whether it works with Triton 3.3.1 is a hypothesis |
| Batching metadata / KV-cache handling | Decode errors that follow the prefill pattern suggest contaminated KV; **hypothesis**, not isolated |
| Other unresolved issues | None observed at B=1 (100% agreement with fp32) |

## 8. Latency and throughput (preliminary, naive path, one run each)

Mean wall per `generate` call (ms): prefill (candidate-length prompts) B=1: 78, B=4: 194, B=16: 703; 4-token decode B=1: 115, B=4: 219, B=16: 705; engine init 54 s (graphs) / 35 s (eager); eager was somewhat faster (prefill 62 / 173 / 669 ms). These are timings of the **defective** path; the old verifier timing calibration must not be reused for any configuration with a different attention implementation.

## 9. GO / NO-GO and next experiment

- **NO-GO** for A, C, D, E (fail gates) and B, F (cannot run). No runtime is currently acceptable for B > 2.
- **Why B ≤ 2 is currently correct:** with one or two sequences the offsets-as-lengths loop still partitions the batch correctly; this is why B=1 agrees with fp32 and why the isolated calibration looked consistent.
- **Recommended candidate (not yet validated):** G, the current build with `scripts/rocm_sdpa_runtime_patch.py` applied before engine creation (converts the cumulative vector to lengths; in-process, no reinstall; CPU tests only). Alternative H: put the system Triton 3.3.1 first on the path and retest the Triton FA kernel (B) and the V1 engine (F). Fallbacks if neither passes: a different supported vLLM/ROCm build, installing ROCm `flash_attn` (rebuild, needs approval), or a correctness-first reference verifier (HF) for validation.
- **Another GPU experiment is necessary**: one job, about 10 minutes and roughly 100–150 calls (G and H with the same dev/held-out/permutation gates, plus the B=1-prefill → batched-decode test to confirm KV inheritance). I have not submitted it.

## 10. Impact on earlier work (reporting only; nothing recomputed or altered)

Every earlier job exported `VLLM_USE_TRITON_FLASH_ATTN=0` and `flash_attn` is absent, so all earlier simulator, certification and calibration runs used the naive path. Verification requests that shared a prefill step with at least two earlier sequences were computed with contaminated attention. Under the equal-length model, about 13.7% of verification requests in the N=10 smoke batch-size mix would be affected (B=1–2 batches are unaffected); unequal lengths change the exact slots but not the nature of the defect. Historical N*, acceptance and timing numbers are preserved as they are and should be labelled "batched-verifier-defective" until a validated runtime exists.
