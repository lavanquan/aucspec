# Canonical corrected verifier qualification (Task B6) — STATUS: NOT QUALIFIED YET (two attempts stopped by my own guard)

Neither GPU job produced a single verifier measurement: **0 vLLM correctness calls so far**. Both stops were defects in the runtime guard that I wrote, not findings about the corrected runtime. Per the instructions, no further job has been submitted; a third attempt needs your approval. No GO/NO-GO can be given.

## 1. The two failed attempts

| | Job 51136681 (first) | Job 51142135 (retry) |
|---|---|---|
| Node / time / exit | `nid002230`, 2 min 2 s, exit 1 | `nid002174`, 1 min 51 s, **Slurm `FAILED`, ExitCode 10** (fail-closed code propagated) |
| Preflight (versions, paths, checksums, deps) | passed | passed (6 s) |
| Frozen inputs | built here: 96 contexts, 36 HF fp32 calls | **reused**: sha256 `b59718762095c13c20fc196acc52e3b86038680a0785ae6f53390c680ae6cffc`, 64 dev + 32 held-out, integrity check passed in the job |
| Engine | not reached | **constructed** (init 35 s, CUDA graphs captured) |
| Stop reason | `PYTHONPATH` first entry was `legacy/src` (importing `target_client` prepends it) | `RuntimeGuardError('sys.path lists the user-site before the Triton shim')`, raised by the in-process guard right after the engine was built |
| vLLM verification calls | 0 | **0** |
| Orchestration | smoke skipped | **fail-closed worked as designed**: qualification rc ≠ 0 → job exit 10, smoke skipped, no output mistaken for success |

Analysis of the second stop. The failing condition was an ordering heuristic (`sys.path.index(shim) < index of any "pythonuserbase" entry`) evaluated *after* vLLM engine start-up. It cannot be a statement about which Triton was used: by then `triton` is already in `sys.modules`, and the same call also verified the imported version and real file against the manifest (only the ordering check is named in the error, so the version/path checks passed). The ordering at process start is correct: a CPU probe (job 51143094) shows the shim at `sys.path` index 1 and the user-site at 5 at start and after importing torch, triton 3.3.1, transformers, the repository and the guard modules; the probe could not go on to `import vllm` on a CPU node (it hangs there), so **which vLLM step reorders `sys.path` is not identified**. Because the check was a proxy, not an invariant, it was wrong to enforce it after engine start. The live-engine check (28/28 Triton-FA layers, zero naive layers) had not run yet, so the layer evidence is still missing.

Fix (commit following `e3836c0`, 329 CPU tests pass):
- Launch-time ordering stays strict (`preflight_runtime.py`, before any repository import).
- In process: the environment variable must have the shim first once the single known `legacy/src` prefix is ignored; the real imported `triton` (version 3.3.1 and file under torch's package directory, never the user-site 3.8.0) is checked against the manifest; `sys.path` order is only **recorded** (head of `sys.path` stored in the evidence), not enforced.
- `enforce_runtime` now evaluates all checks (environment, imports, live engine, config semantics) and reports every violation together with the `sys.path` head instead of stopping at the first, so the next run cannot hide a second problem behind the first.

## 2. CPU fixes and test evidence (all before the retry unless noted)

- **Gate A (fail-closed orchestration)**: nonzero exit for qualification process failure/missing output (10), dev or held-out gate failure or incompleteness (11), a mandatory comparison with zero margin-clear observations (11), missing runtime evidence or call overrun (11), preflight (65) or frozen-input integrity failure (3), smoke or smoke-check failure (12); smoke only after every gate; job success only if qualification and smoke both pass. 12 regression tests drive a fake runner through each failure mode.
- **Gate B (semantics, not only argmax)**: `check_true`/`check_corrupt`/`evaluate_set` validate `accepted_length`, committed stream, correction, bonus, `reached_eos` and verifier greedy tokens against the independent fp32 expectation (`expected_true`/`expected_corrupt` logic), corrupted proposals must be rejected at the planted position with fp32's token, results must equal B=1 and be permutation-consistent, decode must equal the fp32 chain and prefill. Gate = 99% per run on margin-clear requests with non-zero denominators; raw counts (near ties included) are reported alongside. Synthetic tests show that argmax-correct but semantically wrong outputs (wrong committed stream, correction, bonus, accepted length, EOS flag, rejection position) fail the gate and that near-tie flips stay visible in raw counts without failing the clear gate.
- **Gate C**: `verify_frozen_inputs.py` checks sha256, 64/32 contexts, length groups, token hashes, fp32 top-2 consistency, proposal = fp32 greedy chain, no EOS, disjoint questions; the earlier job's result shows 0 vLLM calls and no engine evidence (so no engine inspected these inputs). The runtime guard also checks `service_time_model=batch_wall` and `verifier_price_mode=zero` (config semantics), V0 engine, `max_logprobs=1`, original V1 tie-break, prefix caching off. An accounting probe of `integration_checks.py` logic on existing smoke CSVs gave zero false positives.
- Immutable run directories: `tfa33_wiring_smoke_r2_20261019` and `…_r3_20261019` are preserved unmodified; outputs of job 51136681 and 51142135 are untouched (`rocm_qual_20261019/`, `rocm_qual_r2_20261019/`).

## 3. Delivered and still valid

Launcher `scripts/run_canonical_job.sh` (Triton FA, recorded shim, preflight, no `pip install`, real exit code); legacy launchers labelled DEFECTIVE and refusing to run; configs without `runtime:` refused; runtime identity `V1-batchwall-tritonfa33` recorded by the simulator; qualification harness (B = 1…32, γ = 0…4, three batch orders, corrupted proposals, decode, B=32 scheduling instrumentation); integration-smoke checker (Triton FA live in the simulator process with call counts, timing/serialization invariants, B > 2 exposure reported as a limitation if absent, acceptance and workload accounting, provenance). V1 tie-break, controller and pricing are untouched; V2 changes are not imported.

## 4. Not established

Everything about correctness at B = 1…32 on the fresh contexts, held-out agreement, semantic checks on real `verify_batch_sync` output, the simulator wiring smoke, and the live 28-layer evidence. The retry used 0 of 400 calls; 0 HF calls (reused). Calibration must be repeated for the corrected runtime once it qualifies (expected), and all earlier V1 results remain "batched-verifier-defective" and unchanged.

## 5. Recommendation

**No GO/NO-GO yet.** A third and final attempt needs your approval; with the guard now reporting all violations and recording rather than enforcing `sys.path` order, the remaining risk is another unforeseen guard assumption. If you prefer, I can first run a minimal one-node check that only builds the engine, runs the guard and two verifier calls (about 3 minutes) before spending the full job; I have not done this because it is a separate GPU job.
