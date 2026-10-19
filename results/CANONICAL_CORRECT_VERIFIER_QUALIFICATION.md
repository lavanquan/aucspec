# Canonical corrected verifier qualification (Task B6) — INTERIM STATUS: not yet qualified

**The single approved GPU job (Slurm 51136681) did not reach the vLLM engine.** The runtime guard I added stopped it before any vLLM call (0 verifier calls), because of a defect in my own guard, not in the runtime. Qualification, held-out testing and the wiring smoke have therefore **not** been run. A corrected job needs your approval because the budget was one job and no automatic resubmission.

## 1. What happened in job 51136681 (node `nid002230`, 2 min 2 s, exit 1)

| Step | Result |
|---|---|
| Preflight (versions, import paths, checksums, frozen dependencies) | **passed**: torch 2.7.1a0, **triton 3.3.1 imported through the shim** (`…/aucspec_runtime/tritonfa33/shim/triton`), real path under `/usr/local/lib/python3.12/dist-packages/triton`, checksums equal the manifest |
| HF fp32 reference + fresh context construction | **passed** (95 s, 36 HF calls): 96 fresh contexts built, 64 dev (4 length groups: 108/138/168/205) and 32 held-out (2 groups: 125/190), model-generated histories, equal-length groups, questions disjoint from the B4/B5 sets; fp32 greedy 4-token continuation and top-2 at 5 positions per context; no context contains EOS |
| Qualification (vLLM) | **aborted by my guard before engine construction**: `GuardError: runtime environment refused: PYTHONPATH must start with the Triton shim … (starts with '…/legacy/src')` |
| Integration smoke | skipped (qualification gate not passed) |

Root cause: `edge_specsim/target_client.py` prepends `legacy/src` to `os.environ["PYTHONPATH"]` at import time (so vLLM subprocesses can import the package). My guard compared the *first entry of the environment variable* with the shim, so it fired after that import. The actual `sys.path` of the process was still correct (the shim was found first when `triton` was imported, and preflight proved Triton 3.3.1), so the guard was too strict in the wrong place. The vLLM engine, the simulator and the earlier B5 results are unaffected. The guard did what it was designed to do (abort on an unexpected environment instead of continuing).

## 2. Fix (committed, CPU-tested, not yet run on GPU)

- The strict "shim is the first `PYTHONPATH` entry" rule is kept at launch time (`preflight_runtime.py`, run before any repository import).
- The in-process check (`runtime_guard.enforce_runtime`, `canonical_launch`, qualification runner) now requires the shim to be present on `PYTHONPATH`, to be on `sys.path` **before the user-site**, and the real imported `triton` file and version to match the manifest. New unit test pins both behaviours (301 tests pass).
- The orchestrator can reuse the already-frozen qualification contexts from this job (`--reuse-frozen`): no vLLM call touched them, so the dev and held-out sets remain uninspected by the runtime and the 36 HF calls need not be repeated.
- Estimated retry cost: 0 HF calls + 398 vLLM calls (cap 400; 500 allowed for the qualification portion), about 10–12 minutes of the 35 requested (engine start about 70 s, Triton JIT on first shapes, then the matrix; the wiring smoke about 6 minutes).

## 3. Delivered so far (branch `research/verifier-batchwall`)

1. **Launcher**: `scripts/run_canonical_job.sh` sets `VLLM_USE_TRITON_FLASH_ATTN=1`, puts the recorded Triton shim first on `PYTHONPATH`, runs `preflight_runtime.py`, contains no `pip install`, and exits with the simulator's real exit code (`rc=$?; … exit $rc`). The shim is one symlink created by `runtime/tritonfa33/setup_shim.sh` (idempotent, refuses to overwrite) in a persistent directory under `/software`; paths and sha256 of the key files are in `runtime/tritonfa33/MANIFEST.json` (generated inside the job environment). Nothing was installed or removed.
2. **Legacy launchers**: 12 repository launchers that select `VLLM_USE_TRITON_FLASH_ATTN=0` carry a "DEFECTIVE LEGACY RUNTIME GUARD" and exit 64 unless `ALLOW_DEFECTIVE_LEGACY_RUNTIME=1`; the previous canonical launcher is kept as `scripts/run_canonical_job_legacy_naive_sdpa.sh`. Configs of the old runtime are labelled, and `canonical_guard` refuses any config without `runtime.identity: V1-batchwall-tritonfa33` and `require_triton_fa: true`. The launcher also refuses `ALLOW_DEFECTIVE_LEGACY_RUNTIME`, a backend override or a V1-engine request.
3. **Runtime identity** `V1-batchwall-tritonfa33` (configs `canonical_batchwall_tritonfa33_v1.yaml` and `…_smoke_wiring.yaml`): V0 engine, Triton FA, `max_logprobs=1`, original V1 first-max tie-break, prefix caching off, `service_time_model=batch_wall`, `verifier_price_mode=zero`. The simulator, right after building the target, verifies from inside its own process that all attention layers use Triton FA with zero naive layers, counts Triton-FA calls and writes the identity (versions, import paths, checksums, engine class, layer counts, commit, config hash, population fingerprint) into `run_config.json`. The V2 tie-break edits are not imported and no controller/pricing code changed.
4. **Qualification harness** (`scripts/rocm_qual_*.py`): canonical engine (`max_logprobs=1`), all verification through the unmodified `verify_batch_sync`; B = 1, 2, 4, 8, 16, 32; γ = 0…4; three distinct batch orders at γ = 4 for every B ≥ 2; correct and deliberately corrupted proposals (to test rejection and correction tokens against the fp32 reference); batched decode; scheduler instrumentation to report prefill splitting at B = 32 (32 × about 170 tokens exceeds the engine's 4096-token step, so it will not be reported as a 32-way simultaneous prefill); raw and margin-filtered agreement with denominators; held-out opened only if the dev gates pass.
5. **Tests**: 301 CPU tests pass (guards, environment selection, launcher content, legacy refusal executed through `bash`, plans and metrics).

## 4. Not yet established

Correctness at B = 1…32 on fresh variable-length contexts; held-out agreement with the independent fp32 reference; actual-`verify_batch_sync` acceptance and corrected-token agreement; B > 2 coverage in the simulator; end-to-end invariants and provenance of a real simulator run; whether the runtime is safe to freeze. **Recommendation at this point: no GO/NO-GO** until the corrected job has run. Calibration will have to be repeated for the new runtime (expected), and all earlier V1 results stay "batched-verifier-defective" and untouched.

## 5. Ready-to-run retry (needs your approval)

```
sbatch --job-name=tfa33_qual --output=/scratch/pawsey1257/quanla/rocm_qualification/rocm_qual_r2_20261019/slurm_%j.log \
  scripts/run_rocm_qualification_job.sh <worktree> configs/canonical_batchwall_tritonfa33_v1.yaml \
  /scratch/pawsey1257/quanla/rocm_fi/rocm_fi_20261019/frozen_contexts.json rocm_qual_r2_20261019 tfa33_wiring_smoke_r2_20261019 \
  /scratch/pawsey1257/quanla/rocm_qualification/rocm_qual_20261019/frozen_qualification_contexts.json
```
