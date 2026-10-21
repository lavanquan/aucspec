# AUCspec C3.1 — RQ3 pilot contract (committed BEFORE any GPU work; nothing requested)

**Status of this document: a pre-registration of the only pilot that C3.1 considers scientifically defensible, together with the verdict that it is not yet justified.**
**GO/NO-GO: NO-GO. Requested walltime: 0. Requested budget: 0 node-h. No job is, or will be, submitted on the basis of this document.**
Why NO-GO (details: `AUCSPEC_C3_1_PHYSICAL_FEASIBILITY_REVIEW.md`): in populations grounded in the measured GSM8K/Qwen traces the treatment changes batch members often but raises the minimum rate by only
+3.4 % (γ=4) / +5.2 % (γ=2) at the single constrained cell (N=24, B_max=8), by 0 at N=32, B_max=8 and at both B_max=16 negative controls; double-digit effects need an acceptance spread 3–9× the measured one.
A four-run pilot cannot resolve a ≈ 3–5 % effect against unknown hardware run-to-run noise, so it would be spending ≈ 50–70 % of the remaining budget to learn almost nothing.

The contract below defines what a pilot would have to look like **if** the preconditions in §9 are first satisfied and the user separately authorizes budget in writing.

## 1. Question and non-goals
Question: with everything else identical, does deficit-first ranking of the eligible requests raise the minimum per-client useful-token rate of a real-verifier run, in the constrained-serving regime where the batch cap binds?
Non-goals: certified capacity, a frontier, AUC, adaptive depth, verifier pricing, wireless control, any claim about the manuscript's Theorems, any claim about other populations.

## 2. Exact baseline and treatment (identical runtime, only the ranking differs)

| label | role | batching | ranking |
|---|---|---|---|
| R (not run) | legacy head+W FCFS, historical reference only | legacy | arrival |
| **B** | fair baseline | C3 work-conserving: batch starts at max(F_prev, a_head+W), eligible set = everything arrived by the start; same caps/constraints as T | arrival order (ties: client id) — `scheduler: c3_fcfs` |
| **T** | treatment | identical to B | deficit-first, **arrival tie-break** — `scheduler: c3_deficit`, `priority_mode: deficit_fcfs` |

* Treatment variant: the **α-free** member of the family (rank by effective deficit Z_eff = Z_i(τ_i) + x_req·(s − τ_i)/1000, ties by arrival, `max_skips = 2`). In every binding batch of the CPU study it selected exactly the same members as the density-tie-break variant (0.00 % differences, 20 populations × 5 cells), so it needs no acceptance estimate and no cost model: it uses only information the server already has (Z_i, arrival time).
* Fixed in both arms: runtime H (V1-batchwall-tritonfa33, vLLM 0.9.1 V0, Triton FA, prefix caching off, chunked prefill off, `max_num_batched_tokens` 4096), V1 greedy verification, `batch_wall` accounting, causal coordinator with ledger, constant γ = 2 (the depth validated on GPU; memory and gates were measured at γ=2), price zero, `isolated_profile` draft 22.2 ms/token, x_req = 3 tok/s, V = 50, `max_batch_size = 8`, `verify_token_budget`/processed-token budget 4096, W = 150 ms, `max_skips = 2`, 30 s warm-up + 120 s window + 10 s margin, early stop, workload lifecycle checks.
* Cell: **N = 24, B_max = 8** (the only measured-grounded cell with a nonzero CPU effect; N > 2·B_max so the cap binds in ≈ 99 % of batches).

## 3. Matched population and run-seed
Population = the frozen GSM8K-train catalog block assignment (catalog hash `02e79304…`, `population_seed 20261020`), identical for B and T: same prompts per client, same rtt/draft/uplink configuration (homogeneous, as in C1), same start offsets (`clients.start_offset_ms_max = 800`, deterministic hash), **no synthetic jitter** (jitter is not an observed property of the C1 runs; CPU shows offsets alone do not persist — documented, not hidden). Pairing: (B, T) share population and run-seed. Planned runs: B and T at run-seeds s1 and s2 (4 runs). Order: B(s1) first as the memory/validity check of B_max = 8; abort the remaining runs if it is invalid.

## 4. Authentic acceptance
Acceptance is whatever Qwen2.5-7B-Instruct (target) / Qwen2.5-1.5B (draft) produce under greedy verification. No Bernoulli draw, no oracle α, no CPU-fitted constant enters the run (T is α-free). Per-client acceptance, per-prompt acceptance and useful tokens per verification are recorded and reported as observed.

## 5. Metrics
Primary: **x_min** = minimum over clients of useful tokens per second in the measurement window (as in RQ1), paired difference T − B per run-seed.
Secondary (all reported, none used to rescue a null): aggregate goodput Y, Jain index, SLO deficit Σ max(0, 3 − x_i), fraction of clients below 3 tok/s, maximum queue wait, skipped-service events (requests served after ≥ 1 skip), batch-size histogram, **membership difference** (on the logged eligible sets of the B run, share of binding batches in which T's choice differs from B's, and the mean number of swapped members).
Diagnostics: batch wall times vs processed tokens (re-checks the affine model; **not** a calibration claim), peak GPU memory (`simulation.memory_summary`), real/virtual ratio.

## 6. Validity and fail-closed conditions (pre-declared; any failure ⇒ the run is excluded and the comparison for that seed is not reported)
1. Causal ledger clean (D1/D2); verifier serialisation ok; workload lifecycle valid (no exhausted workload, early stop as protocol).
2. No `ConstraintInfeasible`; no batch above B_max = 8 or above 4096 processed tokens; no empty batch; skip bound `max_skips + ⌈(N−1)/B_max⌉` respected.
3. No OOM and no engine error. Peak reserved memory recorded; the pilot is **stopped, not tuned**, if B(s1) reaches the previously observed ceiling (≥ 63.0 GiB reserved).
4. Original real/virtual gate (≤ 3.0): N = 24 failed it in C1.3 (3.07). Runs carry the label **exploratory_v2**, the original verdict is quoted verbatim, and nothing is claimed under the original protocol.
5. Identical information check (automatic): the resolved configurations of B and T are diffed; any difference other than `scheduler/priority_mode` invalidates the pair.
6. GPU memory safety is not proven by any of the above; it is only observed.

## 7. Interpretation (fixed in advance; the CPU-derived effect sizes are NOT thresholds)
* Noise reference: the two B runs (s1, s2) give the only estimate of hardware run-to-run variation of x_min; the **detectable difference is not claimed to be smaller than twice their absolute difference**.
* **No detectable effect** (|Δ| within that reference, or of mixed sign across seeds): reported as "no evidence of a floor effect at this cell"; deficit-first is removed from the headline contribution and AUCspec is a measurement/characterisation paper (RQ1 + causal methodology).
* **Δ < 0 in any valid seed**: falsification at this cell; same consequence.
* **Δ > 0 beyond the reference in both seeds**: a *pilot observation* for this single population and cell. It does not establish capacity, does not generalise to other populations, and by itself does not justify a certification campaign; it would justify only the next step (§9, more populations).
* A null or negative result is reported with the same prominence as a positive one, with all secondary metrics.

## 8. Precise limit of any statistical claim
One population, two run-seeds, one cell: **no inference to populations, no confidence interval for an effect, no p-value, no capacity certification, no frontier statement.** Under greedy verification the run-seeds mostly repeat one trajectory, so they estimate only wall-time noise, not population variation. A population-level claim would need ≥ 5 independent populations (the CPU study used 20) with population as the random effect, paired B/T per population, and an N-sweep for any capacity statement: ≈ 30 runs ≈ 6–7 node-h, a separate decision outside this budget.

## 9. Preconditions for a GO (none currently satisfied)
1. Written user authorization of scope and budget.
2. A realistic source of client heterogeneity under frozen H: the CPU benefit scales with acceptance spread (measured 0.017 → +3–5 %; ≥ 0.05 needed for ≥ 9 %). The only cheap way to create it authentically is a mixed-domain frozen catalog (e.g., math + summarisation + code); its per-domain α under frozen H is **UNKNOWN** (legacy 0.77–0.93 is from a pre-correction pipeline). Measuring it is a separate profiling step that needs its own authorization and is not a policy comparison.
3. CPU replication of §4 of the review on that population (cheap) showing a benefit with grounded parameters.
4. Frozen analysis script and this contract unchanged; fix or waive the inherited `test_payload_interface` float-precision failure.
5. B_max = 8 memory plan reviewed (untested on GPU).

## 10. Requested walltime and budget (summary)

| item | requested now | if later authorized (estimate, not a request) |
|---|---|---|
| GPU jobs | **none** | 4 runs (B,T × 2 run-seeds), one node, `--time=01:15:00` |
| node-h | **0** | ≈ 0.75 expected (0.18–0.25 per N=24 run), ceiling 1.0 node-h of the remaining 1.474 |
| profiling step of §9.2 | none | to be costed separately |
