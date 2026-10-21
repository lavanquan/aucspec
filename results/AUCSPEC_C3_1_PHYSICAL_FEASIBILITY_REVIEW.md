# AUCspec C3.1 — Physical feasibility bridge before any GPU policy comparison

Branch `research/c3-1-physical-feasibility` (base `research/c3-minimal-controller` @ `87852c9`). CPU only: no GPU job, no calibration, no frontier, no pricing, no main merge,
no controller expansion, no parameter tuning. Tags: **[M]** measured in the valid C1.3/C1.4 GPU traces · **[CPU]** model-derived (real scheduler code, CPU verifier stand-in) · **[A]** assumption · **UNKNOWN** not identifiable from existing traces.

## 0. Verdict

**Pilot GO/NO-GO: NO-GO.** Under conditions grounded in the measured GSM8K/Qwen traces, deficit-first scheduling changes the batch members often (≈70 % of binding batches at N=24, B_max=8) but moves the outcome very little:
**x_min +3.4 % (all 20 populations > 0), goodput −0.06 %, Jain +0.001, SLO deficit 0** at (N=24, B_max=8); **0** at (N=32, B_max=8) and at both B_max=16 negative controls; **+13 %** only at N=48, B_max=16 — outside the validated region (memory headroom 0.5 GiB, real/virtual gate already failed at N=24).
Benefits of +9 %…+51 % appear only when the acceptance spread across clients is 3×–9× the measured 0.017, or with hardware/jitter heterogeneity that the C1 configuration does not have. By the task's own criterion ("benefit requires an artificial acceptance range or unrealistic population ⇒ NO-GO") the answer is NO-GO. The contract (`AUCSPEC_C3_1_RQ3_PILOT_CONTRACT.md`) records what would have to change.

## 1. Task A — what the valid GPU traces actually contain (`results/c3_1/trace_conditions.json`, script `scripts/c3_1/trace_conditions.py`)

GSM8K-train, γ=2, FCFS, runs C1.3 cap-16 (N=8,16,24), C1.2 ext (N=12,16), C1.4 probe (N=28,32); frozen runtime H; 136 client estimates.

| quantity | measured [M] | C3 positive CPU scenario [A] | gap |
|---|---|---|---|
| per-client acceptance α̂ (censored-geometric MLE) | mean **0.930**, sd **0.017**, range **0.88–0.96**; between-client *true* sd (sampling variance removed) 0–0.015 | α_i ~ U[0.60, 0.95] (sd ≈ 0.10) | measured spread ≈ **6× smaller**; 0.60 never observed |
| per-prompt acceptance | mean 0.93, observed sd 0.035–0.057, between-prompt true sd 0.006–0.036 (mean 0.020), p5 0.82–0.87 | not modelled in C3; added in C3.1 with sd 0.02 | – |
| positional check (i.i.d. within 2 positions) | α₁ = 0.922, α₂\|1 = 0.936–0.939 | geometric i.i.d. | consistent for positions 1–2 only |
| useful tokens / verification | mean 2.78–2.79 (γ=2, max 3), median 3 | follows from α | – |
| per-client delivered rate | x_min/x_mean **0.95–0.98**, Jain **0.9994–0.9998** (all runs) | x_min/x_mean 0.55–0.70, Jain 0.96 | the CPU hetero scenario is far less uniform than any real run |
| hardware / network config | rtt 50 ms and draft profile **identical** for all clients; uplink 5–20 Mbps (per-round SNR draw) | rtt 20–100, draft 20–60 tok/s, uplink 5–50 | C1 configs are homogeneous |
| context per request | batch means **158** (p5 121, p95 196), ctx_max ≤ 263; 96 generated tokens/sample, ~34 rounds/sample | U[80, 300] (mean 190) | CPU range a bit wider; moment-matched family in C3.1 (prompt 85–170) |
| arrival structure | members of a batch arrive within **0.02–0.04 ms** (median); next arrival after finish **104.5 ms, sd 4.6 ms** (lock-step) | offsets 0–800 ms + jitter ≤ 120 ms | real traces are perfectly phase-locked |
| eligible work at batch start | exactly the previous cohort (8/16/12/14/16); **never > 16** (0 % of batches) | (24–48 clients, B_max 8–16) | cap never bound in any valid run |
| batch size / service | sizes 8,12,14,16; wall mean 403–511 ms (p5–p95 220–678); processed tokens max **3451** | – | – |
| GPU memory | cap-16 probe: peak allocated **55.98 / 55.72 GiB** of 63.98 (N=28/32), reserved 63.4 GiB ⇒ headroom ≈ **0.5 GiB**; cap-32 N=24 OOM (4.08 GiB sampler logprob allocation, ~7,200 positions implied) | B_max=8 | **B_max=8 memory: UNKNOWN** (never run); expected lower, not measured |
| legacy mixed-domain evidence | AUC_ACHIEVABLE_REGION_REPORT: α̂ gsm8k 0.89 / cnn_summarize **0.77** / math 0.93 (real Qwen2.5-7B/1.5B) | – | **produced by the pre-correction pipeline (before the batch-wall / greedy-verifier fixes); not valid under frozen H**; shows that cross-domain heterogeneity may exist, not its size under H |

**UNKNOWN (not invented):** acceptance at positions 3–4 (all traces have γ = 2; the γ=4 arms of every CPU study extrapolate a geometric model); whether acceptance is i.i.d. beyond position 2; the prompt-length distribution of the catalog (hashes only); any arrival jitter / device heterogeneity of a real deployment;
cross-domain α under frozen H; GPU memory at B_max = 8.

**Can the contention be produced by staggered starts + authentic GSM8K prompts? (CPU check, same code)** Start offsets alone do **not** de-synchronise: after a few rounds the cohorts phase-lock again
(next-arrival-after-finish p5–p95 = [148.9, 149.0] ms with offsets, identical to the lock-step population; real traces [104.5, 104.5]); only added device jitter keeps arrivals spread ([154, 262] ms). The cap *does* bind at B_max=8 because N=24 > 2·B_max
(a structural fact, independent of heterogeneity), so "contention" exists — but with GSM8K-only, homogeneous-hardware clients the deficit ordering is almost identical to arrival order, hence the small effect in §4. Jitter is not a measured property of the C1 runs; using it is a pilot design choice, not authentic data.

## 2. Task B — hard-limit boundary (fail-closed)

Audit: `_greedy_fill` previously admitted the first ranked request unconditionally, even if it alone exceeded the processed-token budget M or the predicted-time cap T_max (a silent bypass of a hard limit; no run in C3 triggered it — the largest contexts were far below 4096).
**Fix (smallest):** `preflight_request(req, params)` raises `ConstraintInfeasible` (a `ValueError`) if `ctx + γ > M` or the predicted singleton time `> T_max`; it is called (i) by `VerificationBatcher.submit` before the request is queued and (ii) for every eligible request at the start of selection.
A run containing an inadmissible request is therefore **explicitly invalid** (the exception aborts the run; no row is produced). Liveness is preserved for admissible workloads: if every eligible request is admissible alone, the top-ranked one always fits, so the selection is never empty (tested with requests that each use 3000 of 4096 tokens).
Boundary is inclusive (`ctx+γ == M` admitted). Legacy schedulers, Triton FA, the vLLM sampler and model semantics are untouched. Tests: `test_oversize_singleton_fails_closed…`, `test_singleton_whose_predicted_service_exceeds_the_time_cap_fails_closed`, `test_hard_limits_are_inclusive…`, `test_submit_preflight_rejects…`, `test_real_simulator_fails_closed…`.

Three different kinds of constraint, deliberately not conflated:

| constraint | nature | what it guarantees | what it does not |
|---|---|---|---|
| batch size cap B_max, processed-token budget M (=4096, `max_num_batched_tokens`, chunked prefill off) | **hard admission / token-budget** limit, exact arithmetic on request sizes | no batch above B_max or M (0 violations in 4,848 CPU runs); an inadmissible singleton is rejected | memory: M bounds tokens, not bytes |
| predicted service-time cap T_max (default ∞) | **model-based**: uses the affine CostModel, a prediction with ≈ 12 % MAPE (loaded runs; γ-coefficient unidentified) | batches whose *predicted* time exceeds T_max are not formed | the *actual* time; it is a control, not a safety guarantee |
| GPU memory safety | **physical**; the sampler's logprob workspace scales with processed tokens × vocabulary | – | **cannot be proven from CPU tests.** Evidence is empirical only: cap-16 probes peaked at 63.4/64 GiB reserved (headroom ≈ 0.5 GiB), cap-32 N=24 OOM'd; B_max=8 is untested |

## 3. Task C — the three arms, kept separate and shown matched

Labels: **R** = legacy head+W FCFS (`fix4+legacy`, historical reference only); **B** = C3 work-conserving FCFS (`fix4+PF`, fair baseline); **T** = C3 work-conserving deficit-first (`fix4+DSz`, treatment); **A** = diagnostic ablation, deficit-first with arrival tie-break (`fix4+DSa`).
* `test_baseline_and_treatment_have_identical_information…` (4 populations): the `Spec` dataclasses of B and T differ **only** in `scheduler` (`c3_fcfs` vs `c3_deficit`) and, inside the C3 parameters, in `priority_mode`; alphas, prompt lengths, start offsets, jitter, caps, token budget, cost model, V, x_req, depth, run-seed, truth costs are identical; both controllers are `fixed_gamma = 4`; prompt assignment and rtt are identical in the actual simulator.
* First batch (all clients arrive together, N=24, B_max=8) and overloaded batches (N=24/B_max=8, N=48/B_max=16) in the actual simulator: both arms fill the cap exactly, respect the 4096-token budget and the causal ledger; B is arrival-ordered (ties: client id).
* **Member differences, not deficit-value changes** (same eligible sets, binding batches only, 20 populations, measured family):

| cell | binding batches | T ≠ B | A ≠ B | T ≠ A | density-only ≠ B | mean members swapped T vs B |
|---|---|---|---|---|---|---|
| N=24, B_max=8 | 99 % | **69 %** | 69 % | **0 %** | 100 % | 1.15 of 8 |
| N=32, B_max=8 | 100 % | **0–3 %** | 0–3 % | 0 % | 1–3 % | ≈ 0 |
| N=48, B_max=16 | 99 % | **99 %** | 99 % | 0 % | 100 % | 7.3 of 16 |
| N=24 / 32, B_max=16 (negative control) | ≈ 0 % | not applicable (cap does not bind) | | | | |

  Two consequences: (i) T and A choose the **same** members in every binding batch ⇒ the density term (and therefore the α̂ estimate) plays no role — the working mechanism is the deficit ordering alone, and the pilot treatment can be the α-free variant; (ii) at N=32, B_max=8 deficit order equals arrival order (a fully rotating lock-step), so the "treatment" is the baseline there.

## 4. Task D — robustness over independent populations (`results/c3_1/robustness_summary.json`, 1,260 runs)

Algorithms fixed. Populations: 20 independent draws per family (population = sampling unit; one paired run per population). Families: **measured** (α_i ~ N(0.930, 0.017) clipped to the measured 0.88–0.96, per-prompt sd 0.02, prompts 85–170 tokens, homogeneous hardware, start offsets 0–800 ms),
**measured+jitter** (adds ≤ 120 ms access jitter, an [A]), **wide_async** (the C3 assumption-driven family, kept for contrast), and a dose-response over the client acceptance sd ∈ {0.05, 0.10, 0.15} on the measured family. Paired Δ = T − B; mean over populations, [10th, 90th percentile], fraction of populations with Δ > 0.

| family / cell | B levels (x_min, Y) | Δ x_min [p10, p90] | rel. | Δ goodput | Δ Jain | Δ SLO deficit Σmax(0,3−x_i) | Δ skipped-service share | populations with Δ>0 |
|---|---|---|---|---|---|---|---|---|
| measured, B_max=16, N=24 *(negative control)* | 4.24, 111.8 | +0.001 [−0.04, +0.03] | 0.0 % | −0.06 % | 0.000 | 0 | 0 | 35 % |
| measured, B_max=16, N=32 *(negative control)* | 3.23, 115.8 | −0.008 [−0.04, +0.03] | −0.3 % | −0.15 % | 0.000 | 0 | 0 | 15 % |
| **measured, B_max=8, N=24** | 4.00, 105.8 | **+0.135 [+0.04, +0.24]** | **+3.4 %** | −0.06 % | +0.001 | 0 (all clients ≥ 3 tok/s) | −0.13 | **100 %** |
| measured, B_max=8, N=32 | 2.98, 106.4 | +0.005 [−0.00, +0.04] | +0.2 % | +0.09 % | 0.000 | −0.004 | 0 | 20 % |
| measured, B_max=16, N=48 *(outside validated region)* | 2.08, 114.7 | +0.272 [+0.22, +0.34] | +13.2 % | +0.04 % | +0.003 | −0.05 | −0.28 | 100 % |
| measured+jitter, B_max=8, N=24 | 4.01, 105.8 | +0.123 [+0.03, +0.20] | +3.1 % | −0.01 % | +0.001 | 0 | −0.13 | 95 % |
| acceptance sd 0.05, B_max=8, N=24 | 3.52, 105.3 | +0.326 [+0.23, +0.43] | +9.4 % | −0.33 % | +0.005 | 0 | −0.14 | 100 % |
| acceptance sd 0.10, B_max=8, N=24 | 2.91, 102.8 | +0.533 [+0.35, +0.75] | +19.1 % | −1.17 % | +0.014 | −0.21 | −0.19 | 100 % |
| acceptance sd 0.15, B_max=8, N=24 | 2.40, 100.0 | +0.740 [+0.54, +0.90] | +32.4 % | −2.24 % | +0.025 | −0.96 | −0.24 | 100 % |
| wide_async (C3 assumption), B_max=8, N=24 | 1.81, 62.0 | +0.604 [+0.51, +0.76] | +33.6 % | −3.53 % | +0.036 | **+0.35 (worse)** | −0.43 | 100 % |
| wide_async, B_max=8/16, N=32 | – | +0.001 / +0.019 | ≈ 0 / +1.3 % | | | | | 25 % / 70 % |

**Replication at the GPU-validated depth γ = 2** (`results/c3_1/robustness_gamma2_summary.json`, 400 runs, B = `fix2+PF`, A = α-free `fix2+DSa`, 20 populations): measured family N=24, B_max=8: baseline x_min 2.76, Δ **+0.142 [+0.129, +0.155] (+5.2 %)**, goodput +0.03 %, 20/20 populations positive, binding 99 %, members differ 97 %; N=32, B_max=8: 0.000; B_max=16 negative controls: +0.007 / 0.000; N=48, B_max=16: +0.113 (+7.9 %); with jitter: +5.0 %, 0.0, −0.002, +0.1, +7.6 %. 0 ledger violations, 0 invalid.

Physical constraints in all 1,260 runs: 0 ledger violations, 0 invalid, 0 empty batches, 0 cap violations, 0 token-budget violations, 0 starvation-bound violations.
Reading: (a) the negative controls are null, as the structural argument predicts; (b) in the proposed constrained regime the measured-grounded gain is **+3.4 % at one cell and ≈ 0 at the other**; (c) the effect scales with acceptance spread (3.4 % → 9.4 % → 19 % → 32 % for sd 0.017 → 0.05 → 0.10 → 0.15), so double-digit gains need a population 3×–9× more dispersed than the one measured;
(d) with the measured population there is no SLO-deficit benefit to harvest (every client already exceeds 3 tok/s at B_max=8, N=24); in the wide family the deficit sum even rises (+0.35): the mechanism lifts the floor client by taking service from others.

## 5. Interpretation, limits, and what this does *not* show

* All of §4 is model-derived with an affine verifier; the real verifier's batch-composition dependence (padding, step splitting beyond 4096 tokens) is not modelled.
* Acceptance beyond position 2 is UNKNOWN; the γ=4 results assume a geometric extension of α ≈ 0.93.
* The measured populations are GSM8K-only. Cross-domain mixing could create the spread that the mechanism needs; its size under frozen H is UNKNOWN (legacy evidence, 0.77–0.93, is from a pipeline later found defective).
* Lock-step phase-locking is a property of the emulator with homogeneous clients; real deployments may have jitter, but that would be an assumption, not a measurement.

## 6. Tests added in C3.1 (all CPU)

`legacy/tests/test_c3_controller.py` (+4 fail-closed tests; 35 total), `legacy/tests/test_c3_1_baseline_treatment.py` (12). Full local suite: only the 4 known Windows failures + 2 torch collection errors; Setonix per-module result in §7.

## 7. Platform validation of the final SHA
Job 51268436, Setonix `work` (CPU), SHA `e6a8917`, per-module pytest incl. torch modules: 62 modules, **594 passed, 1 failed** — the inherited `test_payload_interface` float32/float64 assertion only (identical on `fa3fe1f`, `7e802a1`, `8670e5b`); `test_c3_controller.py` 35/35, `test_c3_1_baseline_treatment.py` 12/12. No new failure.
