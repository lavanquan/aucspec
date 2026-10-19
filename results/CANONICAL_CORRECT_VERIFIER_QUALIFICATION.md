# Canonical corrected verifier qualification (Task B6): final report

Runtime identity **V1-batchwall-tritonfa33**: vLLM 0.9.1 V0 engine, Triton FlashAttention with Triton 3.3.1 in all 28 layers, `max_logprobs=1`, original V1 greedy tie-break, prefix caching off, `service_time_model=batch_wall`, `verifier_price_mode=zero`.

## 1. Verdict

| Part | Result |
|---|---|
| Correctness qualification (dev, 64 contexts) | **PASS**: 283 mandatory checks, each ≥ 99% on margin-clear requests with non-zero denominators (all 100%) |
| Held-out qualification (32 contexts, opened only after dev passed) | **PASS**: 96 mandatory checks, all 100% on margin-clear requests |
| Live runtime evidence | **PASS**: Triton FA in 28/28 layers, 0 naive-SDPA layers, 11,508 Triton-FA invocations inside the executing process |
| Canonical-launcher wiring smoke | ran end to end through `canonical_launch.py`; all integration invariants pass when checked afterwards, **but the pipeline's own verdict is FAILED** (Slurm job 51144179 exit 12) because the launcher's workload-exhaustion guard declared the 20 s window invalid |
| Whole job | **FAILED** (qualification AND smoke must both succeed); classification below |

Failure class of job 51144179: **integration measurement-window sizing**, not a guard/environment failure and not a numerical/semantic failure. Five of the eight smoke clients finished all six questions after 15.8–16.5 virtual seconds, below the 0.85 × 20 s = 17 s active span the exhaustion guard requires (the corrected runtime finishes the short 48-token workload faster than I assumed). The smallest necessary fix is a smoke-config change (for example `questions_per_client` 6 → 8, or a 15 s window); I have not made it and have not submitted another job, as instructed.

**Recommendation: GO for freezing H as the corrected verifier runtime for correctness** (section 9), with the formal exit-0 smoke record still outstanding.

## 2. Why attempts 1 and 2 stopped (both my guard, 0 verifier calls)

1. Job 51136681: the guard compared the first `PYTHONPATH` entry with the shim after importing `target_client`, which prepends `legacy/src`.
2. Job 51142135: an ordering heuristic on `sys.path` after engine start-up. The smoke evidence now shows why it fired: vLLM imports `ray`, which inserts `…/site-packages/ray/thirdparty_files` at `sys.path[0]` (a user-site path) ahead of the shim. The module that matters, `triton` 3.3.1 from `/usr/local/lib/python3.12/dist-packages/triton`, was already imported correctly. The check is now record-only for `sys.path` and keeps the real version/file checks; launch-time ordering stays strict in the preflight.

## 3. Runtime and launcher changes, CPU tests

Corrected launcher (`VLLM_USE_TRITON_FLASH_ATTN=1`, recorded Triton shim `…/aucspec_runtime/tritonfa33/shim` → torch's Triton, checksummed `runtime/tritonfa33/MANIFEST.json`, preflight, no `pip install`, real exit code); legacy launchers labelled DEFECTIVE and refusing to run; configs without `runtime:` refused; in-process runtime guard and identity record; fail-closed orchestrator (exit 3/10/11/12/65); semantic gate (accepted length, committed stream, correction, bonus, `reached_eos`, verifier greedy tokens, corrupted-proposal rejection, B=1 identity, permutation consistency, decode/prefill); frozen-input integrity checker. **329 CPU tests pass** (26 for the failure modes of the gate and orchestrator).

## 3b. Provenance

| | |
|---|---|
| Code | `b7a1f66fca5facb4af8dcc6f6100ec1096ab924d`, tree clean (recorded in the job) |
| Job | Slurm **51144179**, node `nid003000`, 1 node, 7 min 17 s of 35 min; preflight 6 s, qualification 199 s (engine init 65 s), smoke 224 s |
| Config | qualification `configs/canonical_batchwall_tritonfa33_v1.yaml` (resolved sha256 `98411c459958f33c…`); smoke `…_smoke_wiring.yaml` (campaign hash `8d4854bafea314ae`, population fingerprint `c2f648e92c714b70`); run directory `tfa33_wiring_smoke_r4_20261019` initialised with the same SHA |
| Frozen inputs | sha256 `b59718762095c13c20fc196acc52e3b86038680a0785ae6f53390c680ae6cffc` (verified in the job), 64 dev (lengths 108/138/168/205 × 16) + 32 held-out (125/190 × 16), model-generated histories, independent HF fp32 references (36 HF calls in job 51136681; **0 new HF calls**); no earlier vLLM engine inspected them (job 51136681 had 0 calls) |
| Software | torch 2.7.1a0+gite2d141d (`/usr/local/lib/python3.12/dist-packages/torch`), **triton 3.3.1** (`/usr/local/lib/python3.12/dist-packages/triton`, imported through the shim), vLLM 0.9.1+rocm633 (`/software/…/aucspec/vllm_rocm_build/vllm/vllm`, editable, unmodified), HIP 6.3.42134, ROCm 6.3.3, MI250X, manifest sha256 `564571978a95c23c…` |
| Engine | `vllm.engine.llm_engine.LLMEngine` (V0), `UniProcExecutor`, bf16, graphs on, prefix caching off, chunked prefill off, `max_num_batched_tokens` 4096, `max_num_seqs` 256, `effective_max_logprobs` 1, tie-break `first_max_in_vllm_logprob_dict_order` |
| Calls | **398 vLLM calls** (2 warm-up + 322 dev + 74 held-out; cap 400), all successful; 0 HF calls; the short smoke is bounded separately |

## 4. Live attention-backend and kernel evidence

Inside the qualification process: 28/28 layers `use_triton_flash_attn=True`, 0 `use_naive_attn`, no `sdpa_attn_func` attribute; 11,508 Triton-FA calls counted by a wrapper, same PID as the engine. Inside the simulator process of the smoke: identity `V1-batchwall-tritonfa33`, 28/28 Triton FA, 0 naive, 13,160 Triton-FA calls, PID equal to the process that ran inference, semantics `batch_wall`/`zero`, git clean, Slurm job id recorded.

## 5. Correctness by B, γ, context group and permutation (dev; margin-filtered = fp32 top-2 gap ≥ 0.5; denominators are positions)

Verifier target tokens vs independent fp32 argmax (positions 0…γ−1):

| B | γ | orders | margin-filtered | raw |
|---|---|---|---|---|
| 1 | 4 | identity (32 contexts) | 116/116 | 128/128 |
| 2 | 4 | identity, 2 permutations | 116/116 each | 128/128 each |
| 4 | 4 | identity, 2 permutations | 236/236 each | 256/256 each |
| 8 | 4 | identity, 2 permutations | 236/236 each | 256/256 each |
| 16 | 4 | identity, 2 permutations | 236/236 each | 256/256 each |
| 32 | 4 | identity, 2 permutations | 236/236 each | 256/256 each |
| 16 | 1 / 2 / 3 | identity and 1 permutation | 61/61, 118/118, 176/176 | 64/64, 128/128, 192/192 |
| 32 | 1 / 2 / 3 | identity | 61/61, 118/118, 176/176 | all matching |
| 4 | 2 | identity | 118/118 | 128/128 |

By length group (B=16 and 32, γ=4): raw 64/64 in each of the four groups, margin-filtered 57/57, 58/58, 61/61, 60/60 for groups 108/138/168/205. No systematic wrong batch slot in any run (empty list). Identical to B=1 input results: 23/23 margin-clear contexts for every B ≥ 2 and every order (the B=1 subset has 23 margin-clear contexts: a small denominator). Permutation identity (every B, vs the identity order, margin-clear contexts): 23/23 (B=2), 46/46 (B = 4, 8, 16, 32).

Semantics of the real `verify_batch_sync` output (margin-clear requests, all runs): `accepted_length`, committed stream, correction, bonus, `reached_eos`, verifier greedy list: 100% in every run (283 checks). Bonus token raw (including near ties): 62–64 of 64 in each run of 64 contexts. Corrupted proposals (rejection planted at a known position): accepted length, committed stream and correction correct for 55/55 margin-clear requests in each of γ=4 at B = 4/16/32 (identity and permutation) and 59/59 for γ=2; raw 62–64 of 64. Batched decode vs the fp32 four-token chain (margin-clear contexts): B=1 11/11, B=4 48/48, B=8 48/48, B=16 48/48 (both orders), B=32 48/48; raw 14/16, 63/64, 62/64, 62/64, 62/64, 64/64. Decode chain vs verifier greedy output on the same context (B=16, γ=4): 46/46.

## 6. Held-out (opened only after dev passed)

Margin-filtered vs fp32: B=1 60/60 (raw 64/64), B = 4, 8, 16, 32 at γ=4: 123/123 (raw 128/128) in every order (B=4: 2 orders, B=16: 3 orders), B=16 γ=2: 61/61. Identical to B=1 input: 13/13 for every B and order; permutation identity 27/27; accepted length, committed stream, correction, bonus, EOS flag and greedy list 100% on margin-clear requests; corrupted proposals 30/30 (B=4 and B=16); decode chain 6/6, 28/28 (B=4, 16, 32); decode vs prefill 27/27. Raw decode: B=1 7/8, B=4 30/32, B=16 31/32, B=32 31/32.

## 7. Remaining near-tie discrepancies (not hidden)

Dev: 25 raw mismatches (11 bonus-token positions, 9 decode chains, 5 corrupted-proposal checks); held-out: 5 (decode chains). **All 30 have fp32 top-2 gaps below 0.5 nats** (dev 0.051–0.294, held-out 0.017–0.074); none has a margin ≥ 0.5. Across batch orders only near-tie contexts change their verifier result: relative to the identity order, 1 context at B=4 (one permutation), 1 context at B=8 (in each of two permutations), and none at B=2, 16 and 32; every margin-clear context is identical (see the permutation checks above). B=32 scheduling: the engine does **not** always prefill 32 sequences at once: with 4096 batched tokens per step, calls often split (for example 32 → 22 + 10, or 24 + 8, or 27 + 5); some calls fit in one step (4064 tokens). B=16 prefills in one step (1,792–2,868 tokens). The B=32 agreement therefore characterises the engine's behaviour for a 32-request submission, not necessarily 32-way simultaneous prefill.

## 8. Wiring smoke (canonical launcher, N=8, seed 1, 6 questions/client, 48 new tokens; **not** a performance result)

Checked after the job on the CPU with the unchanged checker (the orchestrator skipped it because the smoke process exited nonzero): verifier batches 470 with **97 batches of B > 2** (counts 1: 252, 2: 121, 3: 65, 4: 26, 5: 3, 6: 3); timing/serialization invariants pass (0 overlapping batches, occupancy equals modeled service to 3e-11 ms, modeled/wall = 1.000, round-latency decomposition residual 5.5e-11 ms, verifier busy fraction 0.90); acceptance accounting consistent (826 rounds, accepted length within [0, γ], committed ≤ accepted + 1, useful = committed; mean accepted/γ = 0.78); workload accounting consistent (8 clients × 6 samples, no sample over 48 tokens, token sums equal); provenance correct. The run was declared INVALID by the exhaustion guard (min rate 0.95 tok/s is not meaningful).

## 9. Is the runtime safe to freeze, and what follows

- **Freeze: yes for correctness.** Under unchanged thresholds, dev and previously unopened held-out contexts agree with the independent fp32 reference and with B=1 for B up to 32, γ up to 4, three batch orders, correct and corrupted proposals, and batched decode; the only discrepancies are bf16 near ties. Finite sample (96 contexts, six lengths, one engine process): no claim of universal exactness. Near ties are decided by composition-dependent bf16 rounding and V1's first-max dict rule; a deterministic tie rule remains a separate, un-made semantics decision.
- **Outstanding**: one exit-0 smoke record (config sizing fix above); not needed to start the next phase unless you want the formal pipeline success.
- **Calibration must be repeated** on this runtime (expected): the earlier verifier timing model belongs to the defective naive-SDPA runtime. Latencies seen here (verify B=1/2/4/8/16/32: 145/90/146/256/646/932 ms; mixed context lengths 108–205) are not a calibration.
- **Legacy results**: every earlier simulator, certification, smoke and calibration result used the defective runtime (naive SDPA, wrong attention for three or more sequences per prefill step) and is incompatible with V1-batchwall-tritonfa33; they are preserved unchanged and must be labelled "batched-verifier-defective", not mixed with corrected-runtime results.

## 10. GO / NO-GO

**GO** to freeze **V1-batchwall-tritonfa33 (H)** as the corrected verifier runtime. Not started, as instructed: calibration, pricing, capacity experiments. After this, the project should return to the AUCspec capacity-frontier roadmap on the corrected runtime.
