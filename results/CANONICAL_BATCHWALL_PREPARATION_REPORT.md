# Canonical V1-batchwall preparation report (Task A, no GPU)

Branch `research/verifier-batchwall`, code commit `7fe982e` on top of `680845e` (main-line certified revision). No GPU job, sweep or baseline was run. Nothing was merged into main.

## 1. Branch / worktree status and preservation

| Piece | Where preserved | State |
|---|---|---|
| Local commits `cb34518`, `cf5f434` (HF target runtime; `hf_target_client.py`, `specplan_worker_hf.py`, tests; not imported by the simulator) | branch `research/hf-target-runtime` at `cf5f434`, pushed to origin | untouched, not merged |
| Uncommitted edits to `legacy/src/edge_specsim/target_client.py` and `scripts/collect_trace.py` (greedy tie-break = smallest token id, `max_logprobs` 1->2, trace-collection attention changes) | (a) patch `/software/projects/pawsey1257/quanla/aucspec_preserved/uncommitted_verifier_and_trace_20261018.patch` (+ `.sha256`); (b) branch `research/verifier-v2-tiebreak-wip` commit `941453e` (worktree `/scratch/pawsey1257/quanla/aucspec_worktrees/wt_verifier_v2_wip`), pushed to origin | md5 of both files verified equal to the dirty main dir |
| Original working dir `/software/projects/pawsey1257/quanla/aucspec` | detached `cf5f434`, still dirty (the two files above) + untracked files | **not modified** |
| Implementation | branch `research/verifier-batchwall` commit `7fe982e`, worktree `/scratch/pawsey1257/quanla/aucspec_worktrees/wt_batchwall`, pushed to origin | clean |

The V2 tie-break / `max_logprobs` change is NOT in `research/verifier-batchwall`; it remains a separate decision (see section 8).

## 2. Verifier-clock semantics, before and after

Before (V1, `service_time_model: amortized_per_request`, still the default):

- `target_client.verify_batch_sync`: `target_latency_ms = batch_wall / B` for every request.
- `verification_batcher._process`: `modeled_batch_service_ms = max(target_latency_ms) = batch_wall / B`; `verify_finish_ms = batch_start_ms + modeled`; `server_available_at_ms = verify_finish_ms`.
- Consequence: a batch of B occupies the single virtual verifier for 1/B of its real wall time (measured/modeled ratio = B in the analysed data; wall is roughly 129.6 + 44.4 B ms).

After (`service_time_model: batch_wall`, "V1-batchwall"):

- `modeled_batch_service_ms = batch_wall_ms` (full measured wall of the batched call, identical for every request in the batch).
- `verify_finish_ms = batch_start_ms + batch_wall_ms`, shared by all requests of the batch; `batch_start_ms = max(batch_ready_ms, server_available_at_ms)` is unchanged, so the next batch cannot start before this one finishes.
- `client_receive_ms = verify_finish_ms + rtt/2 + download` is unchanged; verifier time enters the round latency exactly once (`target_latency_ms` is only logged, never added to the timeline; checked by test and by reading `_advance_virtual_timeline`).
- Amortized `batch_wall / B` is kept as a diagnostic (`verification_amortized_request_ms`, `target_latency_ms`).

Distinct metrics now logged per batch (round CSV, one repeated row per request, dedupe on `verification_batch_id`):
measured event-loop service (`verification_batch_measured_service_ms`, unchanged meaning), measured batch wall (`verification_batch_wall_ms`), simulated service (`verification_batch_service_ms`), amortized cost (`verification_amortized_request_ms`), actual batch size, logical verifier tokens (`verification_batch_token_cost`, gamma+1 per request), processed context tokens (`..._context_tokens_sum`, `..._max`) and processed tokens (`..._processed_tokens` = context + draft tokens), `verification_service_time_model`.

## 3. Files modified and tests

Modified: `models.py` (+`target_batch_wall_ms`, `stamp_batch_timing`), `target_client.py` (uses `stamp_batch_timing`; descriptive `VERIFIER_SEMANTICS="V1"`, `TIE_BREAK_RULE`, `effective_max_logprobs`; no behaviour change), `verification_queue.py` (metadata fields), `verification_batcher.py` (switch, diagnostics; `VLLMCandidateVerifier` import moved under `TYPE_CHECKING` so the batcher imports without torch), `simulator.py` (switch plumbed, CSV columns, `provenance` + `verifier_semantics` in `run_config.json`), `scripts/run_capacity_frontier.py` (passes campaign hash and population fingerprint to the run config).
New: `provenance.py`, `scripts/canonical_guard.py`, `scripts/canonical_launch.py`, `scripts/run_canonical_job.sh`, `configs/canonical_batchwall_v1.yaml`, three test files.

Tests (CPU only, `.venv-sim`, `PYTHONPATH=legacy/src`): **240 passed** = 213 pre-existing (same 213 pass on the unpatched tree) + 27 new:

- `test_verifier_service_time.py` (13): B=1 identical in both models; B>1 occupancy = full wall under `batch_wall`; V1 control reproduces wall/B (isolated compatibility test); one shared finish time per batch; second batch starts at the first one's finish (single verifier); idle gaps not charged; token/batch-size accounting and constraints (`max_batch_size`) unchanged; fallback if a verifier does not stamp wall time; invalid model rejected.
- `test_provenance_and_guard.py` (13): timeline counts verifier time once (simulator imported with torch modules stubbed); git clean/dirty/untracked-source/non-repo; Slurm env capture; config hash; semantics record; guards (dirty tree, HEAD mismatch, config semantics, quota parse/headroom).
- `test_canonical_launch.py` (1): compact per-batch summary on a synthetic CSV.

Not runnable here and unrelated to the change: `test_network_allocator.py`, `test_payload_interface.py`, `test_project_layout.py` fail at collection (`No module named torch`) on the login node, identically before and after.

No test exposed a deeper inconsistency in the batcher. One observation: `controller.observe_server_batch` accumulates the server virtual queue Q_s from `theta_f_ms_per_token * token_cost`, which equals the modeled batch service, so under `batch_wall` Q_s grows B times faster than under V1. In fixed-budget mode this is inert for decisions because the theta_f signal passed to `choose_gamma` and to the knapsack is 0.0 (`estimate_verify_budget_signal` no-roofline return), so the price terms `Q_s*theta_f*(gamma+1)` vanish. This is derived from code, not measured; it was not changed.

## 4. Unchanged behaviours

Greedy decoding and tie-break rule, `max_logprobs`, acceptance logic, models, batching policy (`capacity_knapsack`, budget 100, max batch 32, wait 150 ms), wireless/channel model, DPP controller, exact-SLO classifier (`classify_feasibility_final`), population construction, exhaustion guard. With `service_time_model` unset the code path is V1 and yields the same modeled service as before (isolated test).

## 5. Proposed naming

- `V1` = legacy amortized clock (all certified results so far; labelled in new logs as `service_time_model=amortized_per_request`).
- `V1-batchwall` = `service_time_model=batch_wall`, the canonical simulator going forward.
- Certified numbers (e.g. N*(3.0) in [10, 11], N*(x) table) are V1 results and must not be quoted as V1-batchwall; they need re-measurement. The tie-break variant, if adopted, would be a separate axis (`V2` tie-break), not mixed into this name.

## 6. Provenance / instrumentation (behaviour-neutral)

`run_config.json` now records: git SHA, dirty flag, dirty tracked files, untracked source files (under `legacy/src`, `legacy/scripts`, `scripts`, `configs`), diff hash; hostname, SLURM job id / node / nodelist / partition, loaded modules, ROCm version file, amdgpu driver version, GPU names and HIP version collected inside the job process; resolved-config SHA-256, campaign config hash, population fingerprint; verifier semantics, tie-break rule, effective `max_logprobs`, service-time model, budget, max batch. Collection is best-effort and never aborts a run (errors are recorded). Batch wall vs modeled service and processed token counts are in the CSV (section 2). Workload exhaustion and measurement duration continue to come from the existing `measure_candidate` path.
Guard: `canonical_launch.py init/run` refuse a dirty source tree, a HEAD different from the one recorded at init, a config whose `service_time_model` is not explicitly `batch_wall`, and insufficient file-count headroom.

## 7. Storage / quota / Acacia

Quota now (`pawseyAccountBalance`, 2026-10-18):

| FS | files (user) | share of limit | space |
|---|---|---|---|
| /software | 248,845 | 99.5% (limit about 250,000; headroom about 1,155) | 76.1% of project quota |
| /scratch | 213,460 | 21.3% | about 125 GiB used |
| /home | 3,124 | 31.2% | 1024 MiB, 100% full |

`/software` is effectively full, so the preflight in the launcher will refuse to start (the guard was run against the real output and parses it correctly). The user's `/software/projects/pawsey1257/quanla` holds about 248.9k files, of which `aucspec` is 56.4k (`.pythonuserbase-rocm-vllm` 39.5k, `.venv-sim` 7.6k, `vllm_rocm_build` 3.3k, `results` 2.7k, `.venv-rocm` 1.6k). Other large directories there: `specdiff_cqb` 24.9k, `pyuserbase` 24.8k, `pyuser` 16.0k, `pluto_pylibs_py310` 15.4k, `pluto_pylibs` 12.5k, `v2dev_dirty_backup_2026-09-28` 12.1k, `specdiff_cqb_v3dev` 9.9k, `specdiff_pylibs` 9.3k, `specplan-v4-09b` 7.4k, `tempospec_runtime_results` 6.8k.

**Headroom plan (>= 10,000 files; nothing done, needs approval).** Candidates that are mine or clearly redundant: (1) `aucspec/vllm_rocm_build` (3.3k build tree; archive to one tarball on /scratch then remove), (2) worktree `aucspec_wt_921dedd` (about 0.6k, certified code is in git), (3) `aucspec/.venv-rocm` (1.6k, if superseded), (4) tar each of `v2dev_dirty_backup_2026-09-28` (12.1k) into one file on /scratch and then remove the directory. Items 1+2+4 free about 16k. The other-project environments (`pyuserbase`, `pluto_*`, `specdiff_*`, `pyuser`) are not touched in this plan. Each tarball would be checksum-verified (`tar --compare`) before any removal. Recommended order: archive first, ask, then delete.

**/scratch purge policy.** A search result excerpt of Pawsey's filesystem-policy documentation states a 21-day inactivity window (changed 10 June 2024; the 30-day figure in older sources is outdated), and warns that `rclone` copies can set old mtimes (use `--local-no-set-modtime`). The documentation page itself did not render through the fetch tool, so this is from a search excerpt and should be confirmed on the live page or with Pawsey support before relying on it. Implication: the 12 raw tars in `/scratch/pawsey1257/quanla/aucspec_raw_archive/` (19.88 GiB, `SHA256SUMS.txt`) are at risk if untouched for 21 days, and `/scratch` is not permanent storage.

**Acacia.** Not configured: `rclone/1.68.1` exists as a module, but there is no `~/.config/rclone`, no `RCLONE_*`/`AWS_*` environment variables, and `/home` is 100% full so `rclone config` cannot write there. **Transfer status: nothing uploaded.** Blocker: an Acacia keypair must be created by the user in the Pawsey portal (S3 endpoint `https://projects.pawsey.org.au`), and a bucket in the project's Acacia allocation; the rclone config should then be placed under `/software` or `/scratch` via `RCLONE_CONFIG`. I did not look for or create any credentials. Once configured: `rclone copy --local-no-set-modtime`-style upload of the 12 tars plus `SHA256SUMS.txt`, then `rclone check`/`rclone cryptcheck` and a checksum manifest committed to git. Checksum verification of the existing archives on /scratch itself was done earlier (`tar --compare`).

Storage-safe launcher: `canonical_launch.py init` creates the immutable run dir under `results/capacity_frontier/<tag>/` with `raw` and `logs` as symlinks into `/scratch/pawsey1257/quanla/canonical_raw/<tag>/`, compact JSON summaries in `compact/`, plus a quota preflight (estimated files per run: 4 on /software, 6 on /scratch, margin 2000).

## 8. Remaining blockers before GPU smoke testing

1. `/software` file count (99.5%): the launcher preflight will refuse until headroom exists; needs approval for section 7 plan.
2. Acacia not set up (user must create keys); raw archives remain only on purgeable `/scratch`.
3. Decision needed: adopt or exclude the V2 tie-break / `max_logprobs=2` change (currently excluded; it changes verifier outputs and would need its own parity run).
4. `theta_f` signal is 0 in fixed-budget mode, so verifier-price terms are inactive in all canonical runs; decide whether this is intended before claiming the DPP uses a verifier price.
5. Parity expectation to confirm on GPU: B=1 should reproduce V1 timing; B>1 runs will shift the capacity frontier (verifier busy time grows by about B), so all N*(x) values need re-measurement.
6. `configs/canonical_batchwall_v1.yaml` is a copy of the x=3.0 certification config plus `service_time_model: batch_wall`; the canonical x-grid, seeds and duration still need to be chosen. The `service_time_model` key does not affect the config hash of old runs (they never contain it).
7. The HF target-runtime branch is not part of the simulator and is out of scope.

## 9. Recommended commands for the controlled GPU parity / sanity test (not run)

Prerequisite: headroom resolved, then from the clean worktree:

```bash
cd /scratch/pawsey1257/quanla/aucspec_worktrees/wt_batchwall
git status --short   # must be empty
PY=/software/projects/pawsey1257/quanla/aucspec/.venv-sim/bin/python
PYTHONPATH=legacy/src $PY -m pytest -q legacy/tests --ignore=legacy/tests/test_network_allocator.py --ignore=legacy/tests/test_payload_interface.py --ignore=legacy/tests/test_project_layout.py
TAG=bw_smoke_$(date +%Y%m%d)
$PY scripts/canonical_launch.py init --run-tag $TAG --config configs/canonical_batchwall_v1.yaml --planned-runs 4
for N in 1 10; do for S in 1; do
  sbatch --job-name=can_N${N}_s${S} --output=/scratch/pawsey1257/quanla/canonical_raw/$TAG/logs/n${N}_s${S}_%j.log \
    scripts/run_canonical_job.sh $PWD $TAG $N $S
done; done
```

Checks afterwards: for N=1 the `batch_wall_ms` equals `verification_batch_service_ms` and wall/modeled = 1 (also true in V1); for N=10, `wall_over_modeled_mean` should be about 1 (it was about B in V1); `run_config.json` shows `git_dirty=false`, the SLURM job id, GPU names and `service_time_model=batch_wall`; the exhaustion guard passes. A V1 control at N=10 from the original certified revision `921dedd` gives the like-for-like comparison. Add 2-3 more seeds at N=10 only after the first pair passes.
