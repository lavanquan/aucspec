# ROCm batched-attention fix validation (Task B5)

One GPU job (Slurm 51134365, one node `nid002230`, 3 min 53 s of the 20 min limit, **194 of 200** target-model calls, no resubmission). Canonical verifier, controller, pricing, calibration and capacity work untouched; nothing merged.

## 1. Result in one paragraph

Both candidates pass every correctness gate on the development contexts and on the previously unopened held-out contexts. **G** (current build + runtime correction of the offsets-as-lengths defect) and **H** (Triton FlashAttention with the torch-compatible Triton 3.3.1, no patch) give 100% argmax agreement with the independent fp32 reference on all positions whose fp32 top-2 margin is ≥ 0.5 nats, for B = 4 and B = 16, for γ = 1/2/4, in original and shuffled batch order, and equal B=1 inference exactly. All residual mismatches are exact bf16 ties (vLLM top-2 gap 0.0) with fp32 margins of 0.07–0.16 nats. The earlier slot corruption is gone. **Recommendation: GO for H as the corrected runtime candidate (G as fallback)**, subject to the caveats in section 8; finite sample, so no claim of mathematical exactness.

## 2. Provenance

| | |
|---|---|
| Code | `7e226c2cdc3db46f86f70fcdb7b596df12dd92f3` on `research/verifier-batchwall`, tree clean (recorded job-side) |
| Config | `configs/canonical_batchwall_smoke60.yaml`, resolved sha256 `a00a57d0ff4b1e49…`; target `Qwen/Qwen2.5-7B-Instruct`, bf16, prefix caching off, graphs on, `gpu_memory_utilization` 0.7, `max_model_len` 4096; diagnostic engine `max_logprobs=2`; greedy |
| Hardware / software | AMD Instinct MI250X, ROCm 6.3.3-74, HIP 6.3.42134; vLLM 0.9.1 V0 engine (`UniProcExecutor`), torch 2.7.1a0 (`/usr/local/lib/python3.12/dist-packages/torch`) |
| Inputs | the frozen B4 token-id contexts (file sha256 starts `9559637ef8da7e76…`, identical to B4) and the existing HF fp32 oracle; dev: 32 contexts, 128 positions (123 with fp32 gap ≥ 0.5); held-out: 32 contexts, 128 positions (118 with gap ≥ 0.5). No HF calls were needed |
| Calls | G 124 (78 dev + 46 held-out), H 70 (48 dev + 22 held-out) |

## 3. Which attention implementation executed, and proof

| | G | H |
|---|---|---|
| Backend class | `ROCmFlashAttentionBackend` | `ROCmFlashAttentionBackend` |
| Implementation in all 28 layers | naive SDPA (`use_naive_attn=True` 28/28) **with the patched function bound in 28/28 layers** | Triton FA (`use_triton_flash_attn=True` 28/28, `use_naive_attn=False` 28/28 false, no `sdpa_attn_func` attribute in any layer) |
| Activation in the executing process | wrapper called **3,500** times, **3,500 converted, 0 violations, 0 pass-through**, by PID 2939173 = the PID of the engine process (single-process V0 executor, so there is no separate worker) | Triton kernel wrapper called **1,960** times by PID 2939666 = engine PID |
| Batch sizes seen by the patch | 1 (924 calls), 4 (1,792), 16 (756), and 256 (28 calls: the engine's memory-profiling dummy run, valid cumulative offsets) | n/a |
| Libraries | triton 3.8.0 from the project user-site (unchanged), flags `VLLM_USE_TRITON_FLASH_ATTN=0` | **triton 3.3.1** imported from a symlink shim `…/triton_shim/triton -> /usr/local/lib/python3.12/dist-packages/triton` placed first on `PYTHONPATH` (the symlink lives in the run directory on `/scratch`; nothing was installed, removed or edited); `VLLM_USE_TRITON_FLASH_ATTN=1`; the runner aborts if the imported Triton is not 3.3.x |

G's checks: patch applied before engine construction; strict validation after construction (first offset 0, monotone, last offset = number of prefill tokens, all lengths > 0; a violation raises, never silently falls back); the wrapper is only reached from the prefill branch, so decode-only calls do not enter it (call counts equal 28 layers × prefill steps); per-sequence lengths are not reinterpreted because anything not shaped like a cumulative vector raises in strict mode. The installed vLLM files are not edited. The V1 engine was not attempted (lower priority; no extra job).

## 4. Correctness by B, γ and batch order (argmax vs fp32, positions with fp32 gap ≥ 0.5; "vs B=1" over the same 12 contexts)

Denominators are positions. Dev set (n_ctx = 32; B=1 reference = 12 contexts).

| B | γ | order | G vs fp32 | G vs B=1 | H vs fp32 | H vs B=1 |
|---|---|---|---|---|---|---|
| 1 | 4 | – | 46/46 | – | 46/46 | – |
| 4 | 1 | identity | 31/31 | 12/12 | not run | – |
| 4 | 2 | identity / shuffled | 60/60 / 60/60 | 24/24 / 24/24 | not run | – |
| 4 | 4 | identity / shuffled | 123/123 / 123/123 | 48/48 / 48/48 | 123/123 / 123/123 | 48/48 / 48/48 |
| 16 | 1 | identity / shuffled | 31/31 / 31/31 | 12/12 / 12/12 | 31/31 / not run | 12/12 |
| 16 | 2 | identity / shuffled | 60/60 / 60/60 | 24/24 / 24/24 | 60/60 / not run | 24/24 |
| 16 | 4 | identity / shuffled | 123/123 / 123/123 | 48/48 / 48/48 | 123/123 / 123/123 | 48/48 / 48/48 |

Held-out set (previously unopened; no tuning). B=1 reference = 12 contexts for G, 8 for H.

| B | γ | order | G vs fp32 | G vs B=1 | H vs fp32 | H vs B=1 |
|---|---|---|---|---|---|---|
| 1 | 4 | – | 43/43 | – | 30/30 | – |
| 4 | 4 | identity / shuffled | 118/118 / 118/118 | 48/48 / 48/48 | 118/118 / not run | 32/32 |
| 16 | 1 | identity | 28/28 | 12/12 | not run | – |
| 16 | 2 | identity | 58/58 | 24/24 | not run | – |
| 16 | 4 | identity / shuffled | 118/118 / 118/118 | 48/48 / 48/48 | 118/118 / 118/118 | 32/32 / 32/32 |

Wrong-slot profile at B=16, γ=4: no wrong slot at gap ≥ 0.5 in either order for either candidate (the defective runtime had 10 of 16 slots wrong, identically in both orders). The gate (≥ 99% for every individual run, not pooled) passes for every run listed. Denominator caveat: at gap ≥ 0.5 a single miss in 123 positions is 99.2%; B=1 comparisons have 48 positions, so any miss would give 97.9%; none occurred.

## 5. Independent fp32 comparison and every residual mismatch

All positions (not only gap ≥ 0.5): G dev 127/128 at B=4 shuffled, otherwise 128/128 per run; G held-out 127/128 in B=4 identity and B=16 shuffled, otherwise 128/128; H dev 63/64 at B=16 γ=2, held-out 127/128 at B=16 identity. The mismatches are:

| Where | Context / position | fp32 argmax / gap | vLLM top-2 gap | Reading |
|---|---|---|---|---|
| G dev B=4 shuffled; H dev B=16 γ=2 | ctx 23, pos 1 | 4024 / 0.157 nats | **0.0** | exact bf16 tie, tie-break picked 5916 |
| G held-out B=4 identity and B=16 shuffled; H held-out B=16 identity | ctx 12, pos 1 | 1537 / 0.075 | **0.0** | exact bf16 tie, tie-break picked 2732 |

The same two contexts are correct in other runs: at a bf16 tie, which of the two tied tokens is reported depends on batch composition, so tie outcomes are order/composition dependent. This is the earlier documented V1 "first maximal dict entry" behaviour, not a kernel defect. No mismatch has an fp32 margin ≥ 0.5.

## 6. Decode versus prefill (same frozen contexts)

| Set / B | G four-token continuation = fp32 | H four-token = fp32 | failures with fp32 gap ≥ 0.5 |
|---|---|---|---|
| dev B=1 | 4/4 | 4/4 | 0 |
| dev B=4 | 15/16 | 15/16 | 0 (ctx 6 pos 0: fp32 gap 0.101) |
| dev B=16 identity | 31/32 | 31/32 | 0 (ctx 24 pos 1: gap 0.112) |
| dev B=16 shuffled | 31/32 | not run | 0 (same ctx 24) |
| held-out B=16 / B=4 / B=1 | 32/32, 16/16, 4/4 | 32/32, not run, not run | 0 |

First decode token = fp32 first token in 32/32 (G dev B=16), 15/16 (B=4). Decode chain = prefill argmax chain on the same context: G dev B=16 31/32, G held-out 32/32, H held-out 31/32 (the one disagreement is the ctx-12 tie above). All decode disagreements are near ties (fp32 gap 0.10–0.11); the runtime is not systematically wrong. Previously defective: 37.5% four-token agreement at B=16.

## 7. Preliminary latency (mean ms per `generate` call, same contexts; one run each; H includes Triton JIT compilation of new shapes)

| Call | Defective (B4 config A) | G | H |
|---|---|---|---|
| prefill B=1 | 78 | 72 | 134 (JIT) |
| prefill B=4 | 194 | 179 | 163 |
| prefill B=16 | 703 | 644 | 682 |
| decode(4 tokens) B=1 | 115 | 111 | 103 |
| decode B=4 | 219 | 209 | 177 |
| decode B=16 | 705 | 585 | 686 |
| engine init | 54 s | 62 s | 68 s |

No throughput claim. The old verifier timing calibration belongs to the defective runtime and must not be reused for G or H.

## 8. Recommendation

- **GO** for a corrected verifier runtime. Preferred: **H** (supported Triton FlashAttention, no patch of vLLM behaviour, only an import-path fix for the Triton version). **G** is a valid fallback that corrects the defect itself.
- Conditions/caveats: (a) finite sample: 64 contexts, two lengths, γ ≤ 4, one fixed shuffle seed, single engine process; (b) H dev was a lighter plan than G (no B=4 γ = 1/2, only one shuffled order for γ = 4), and its B=1 held-out reference covers 8 contexts; (c) both leave bf16 exact-tie flips (decided by batch composition) that V1's first-max tie rule does not resolve deterministically; whether to adopt a deterministic tie rule is a separate semantics decision (the V2 tie-break was not touched); (d) `PYTHONPATH` shim for H must be part of every launcher and recorded in provenance; the clean long-term fix is to stop the user-site Triton 3.8.0 from shadowing torch's 3.3.1 (needs approval, not done); (e) the V1 engine was not attempted.
- **Not yet done, needs separate approval:** the same gates on a larger/different context sample before any production use; a re-run of the smoke/calibration/capacity work on the corrected runtime (old timings and all earlier V1 results remain "batched-verifier-defective" and are untouched).

## 9. Upstream patch versus long-term monkey patch

The defect is a genuine vLLM bug (the ROCm backend passes `query_seq_start_loc` where `_sdpa_attention` expects lengths) and should be reported upstream with the CPU reproduction from the isolation report; a one-line upstream fix (or using `attn_metadata.seq_lens`) is preferable to a long-term runtime monkey patch, which depends on import order and a private function. As an operational stopgap G is acceptable only with the strict validation and activation logging used here; for production-like runs prefer H (or a vLLM/ROCm build in which either the naive path or the Triton/CK path is correct by default), and keep the correctness gate as a launch preflight (B=16 shuffled vs B=1 agreement ≥ 99%).
