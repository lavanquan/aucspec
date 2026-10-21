# AUCspec C4 — Minimum publishable evidence: three independent paper paths

Scope: CPU-only assessment from evidence already in the repositories (C0–C3.1, C4 CPU runs, `aucspec-latex` @ `2986a14`). No GPU performance or paper result is projected; "budget" lines are measured run costs from earlier reports (≈ 0.18–0.25 node-h per N ≥ 24 run; ≈ 0.65 node-h per seed of N ∈ {8,16,24,32}; ≈ 10–14 node-h to certify one x-value; 1.474 node-h remaining) and **effort estimates** (calendar time of one researcher), not predictions of outcomes. Novelty is **not assumed**: every novelty statement below is a question for literature verification, not a finding.

## 1. The three paths

### Path 1 — Capacity characterisation: a structural result for finite batch caps, full-context verifier costs and interactivity requirements

* **Nontrivial claim (candidate).** For a verifier with batch cap B, per-batch overhead a and per-request cost d = c + b·ctx + b_γ·γ (stateless full-context verification), the saturated closed-system capacity is x(N) = φ / (⌈N/B⌉·a + N·d̄) with cohort quantisation (staircase in N, e.g. 16+8 at N=24), an unsaturated branch R = max{⌈N/B⌉a + N d̄, device + RTT + W + a + B d̄}, a regime boundary N₁(x), and the structural fact that any batch-*selection* policy is vacuous for N ≤ 2·B (eligible ≤ N − |previous batch|). The claim is a theorem with explicit conditions (when cohorts persist, when de-synchronised arrivals break it) plus a validation, not the formula alone.
* **Already proven / measured.**
  * The form of Prop. 1 (affine in N within a branch) is proved; the structural vacuity argument (N ≤ 2B) is argued and confirmed on 3,500+ CPU runs and on the 7 GPU runs (eligible work never > 16, 0 %).
  * Out-of-N check: constants (a, b) fitted on N ≤ 16 runs predict goodput at the held-out N = 24 / 28 / 32 with −1.7 / −3.5 / −6.0 % error (`results/c4/finite_cap_law_check.json`; one seed, one catalog; measured ctx and φ inputs; the fitted intercept is unstable, 11 vs 52 ms, so only b is well identified). N ≤ 16 is *not* in the saturated branch (the law over-predicts by +43 % at N=16) — the unsaturated branch is not yet characterised.
  * Descriptive RQ1: goodput +6 % vs floor −21 % over N = 24→32.
* **Missing.** (a) The theorem with conditions and proof for the unsaturated and saturated branches, including cohort quantisation and the monotonicity region of x\*(N); (b) an exact CPU discrete-event check of the theorem (ground truth available); (c) independent validation: other seeds, another catalog (the test-split pilot already exists for N ≤ 16), and a de-synchronised-arrival condition; (d) identification of b_γ (varied-γ measurements at fixed membership); (e) a statement of which parts are not simply bulk-service / closed batch-service queue results.
* **Prior-work questions requiring verification.** Bulk-service queues (Bailey-type), closed queueing networks with batch service, cyclic/polling systems with cohort rotation; analytic models of continuous batching and of LLM serving capacity; "goodput at SLO attainment" as a capacity notion in disaggregated/chunked-prefill serving (DistServe-style definitions); analytic goodput models for speculative decoding. *Which of the cohort-staircase / N ≤ 2B / non-monotonicity results are already known?*
* **Minimum budget.** CPU/theory: ≈ 2–4 weeks (estimate). GPU: a varied-γ verifier micro-benchmark (tens of GPU-minutes, ≈ 0.1–0.3 node-h) + 2 further seeds of N ∈ {8,16,24,32} (≈ 1.3 node-h) ≈ the entire remaining budget; a de-synchronised-arrival run set would need extra budget not available.
* **Disconfirming outcome.** Held-out error > 10 % at N ≥ 24 on other seeds/catalog; the law fails under jittered arrivals (⇒ it is an emulator-cohort artefact); or the result is a re-statement of known bulk-service results.
* **Can it support a full paper alone?** Unlikely alone (it is a modelling result about one regime); it can be the "mechanism" half of a paper paired with Path 2's measurement method.

### Path 2 — Statistically rigorous, sample-efficient measurement of controller-achieved capacity boundaries under uncertain, possibly nonmonotone feasibility

* **Nontrivial claim (candidate).** An adaptive procedure that returns, for a given policy and population, a certified bracket of N\*(x) (or x\*(N)) with proven simultaneous error control per client and across the search, valid when feasibility is *not* monotone in N, and uses substantially fewer runs than fixed-K bisection — e.g. by sequential (anytime-valid) classification per point, alpha-spending across the search path, reuse of runs across x via the per-client rate samples, and a structural prior (Path 1) to place probes.
* **Already proven / measured.**
  * C0 quantified that the percentile-bootstrap classifier is anti-conservative (false-FEASIBLE rate 12.3 % at K=3, 4.7 % at K=5, 1.8 % at K=7 when the weakest client's true mean equals x) and C1 replaced it by a Student-t Bonferroni classifier with INVALID/UNCERTAIN states; bound logic bugs M5–M9 fixed; lifecycle validity, population/run-seed separation, causal ledger, determinism: tested.
  * One real instance of non-monotone feasibility (pilot x_min(8) < x_min(12), unexplained).
  * Cost datum: certifying one x-value with the current scheme ≈ 10–14 node-h.
* **Missing.** (a) An algorithm with a *proved* error guarantee under non-monotonicity and a measured saving over the current fixed-K scheme — shown first on a CPU ground-truth simulator (the C3 discrete-event harness provides the true boundary) over many synthetic boundaries, including non-monotone ones; (b) a sample-complexity statement; (c) one GPU demonstration (a single certified boundary point) — which the current budget cannot pay (≈ 10–14 node-h) unless the adaptive scheme itself reduces the cost, which is the claim to be tested, not assumed; (d) a decision on whether Student-t on K ≈ 5–12 run-seeds is valid when run-seeds mostly repeat one trajectory (greedy verification ⇒ small run-to-run variance; population variance is the real random effect).
* **Prior-work questions.** Sequential/anytime-valid testing and alpha-spending; multiple-comparison control for simultaneous per-client intervals; noisy bisection / stochastic root finding / active level-set estimation under monotone and nonmonotone assumptions; SLO-capacity or load-testing methodology in performance engineering; A/B-testing-style sample-efficiency results. *Is "certified capacity under nonmonotone feasibility with per-client simultaneous SLOs" new, or an application of existing level-set estimation?*
* **Minimum budget.** CPU: ≈ 3–5 weeks (algorithm, proof, simulation study). GPU: the demonstration is the problem — one certified point ≈ 10–14 node-h with the current scheme (budget 1.474); a trace-resampling demonstration is impossible with one seed per N.
* **Disconfirming outcome.** No saving over fixed-K bisection at equal error control on CPU ground truth; or the guarantee needs monotonicity the data violate; or sequential testing is already standard and the contribution reduces to engineering.
* **Standalone paper?** Possibly as a measurement-methods paper, with the CPU study as the main evidence and a small GPU case study; not with the current GPU budget.

### Path 3 — SLO-aware verifier scheduling

* **Nontrivial claim (candidate).** A scheduler that uses the SLO-deficit state to raise the minimum per-client rate beyond standard fair scheduling under physical batch constraints.
* **Already proven / measured.** C4 falsification (this branch): age-only is identical to FCFS in the grounded families, stamped Z alone reproduces +5.0 % of the +5.2 % effect, so the effect is carried by the deficit state — but it is +3–5 % x_min at one of two constrained cells, 0 at the other and at both negative controls, +8–13 % only outside the validated region, **no gain in the number of clients meeting the SLO** (at x_req=3 it is worse: 3.8 vs 5.6 of 24 meet it), +396 ms worst-case wait, goodput ±0.1 %. Larger effects need acceptance dispersion 3–9× the measured one (C3.1).
* **Missing.** A real heterogeneous workload under frozen H (cross-domain α unknown), a baseline set that includes the standard fair schedulers, and a regime (N > 2·B_max) that real deployments actually operate in; a theorem relating the deficit rule to max-min fairness / LAS is *already standard*.
* **Prior-work questions.** Virtual-token-counter fairness in LLM serving (OSDI'24-type), least-attained-service, deficit round robin, max-min fair scheduling, SLO-aware speculative serving (SmartSpec/TurboSpec, AdaServe): *is the Lindley deficit weighting distinct from these?*
* **Minimum budget.** ≥ 0.75 node-h for the 4-run contract (`AUCSPEC_C3_1_RQ3_PILOT_CONTRACT.md`) **plus** an unbudgeted mixed-domain α profiling step; certification ≈ 6–7 node-h. Not affordable.
* **Disconfirming outcome (already partly observed).** No SLO-satisfaction gain over FCFS; effect at the noise level of the real verifier.
* **Standalone paper?** No. Do not pursue as a contribution.

## 2. Smallest next research task (CPU-only, no new controller)

**Task:** *Prove and validate the finite-cap closed-system capacity law.*
1. State the model: N closed streams, batch cap B, work-conserving formation, per-batch overhead a, per-request cost d_i = c + b·ctx_i + b_γγ, device+RTT delay D, hold W, acceptance φ. Derive x(N) for the saturated branch (⌈N/B⌉a + Σd_i), the unsaturated branch, the transition N₁, the cohort split rule (which cohorts form, e.g. 16+8, 14+14), and the monotonicity region of x\*(N) (where it can *increase* with N).
2. Verify the theorem **exactly** against the existing CPU discrete-event harness (ground truth; all assumptions satisfiable) across (B, a, b, N, ctx) grids; report where deterministic cohort rotation breaks (jitter, heterogeneous D).
3. Evaluate it **leave-N-out** on the seven existing GPU runs (done once for N ≥ 24 with 3 constants; extend to the unsaturated branch and report the prediction interval, not point errors) and on the test-split pilot (a second catalog).
4. Write the theorem's conditions and a one-page positioning against the literature questions of Path 1 (this is the step that decides novelty).
**Success:** closed-form law with proof, exact on the DES, held-out error ≤ 5–10 % on N ≥ 24 for both catalogs with stated conditions, and at least one structural statement not found in the bulk-service/closed-network literature.
**Disconfirming:** the law is known; or fails under any de-synchronisation that a real population exhibits; or the unsaturated branch cannot be predicted from fitted constants.
**Cost:** ≈ 1–2 weeks (estimate), zero GPU.

## 3. Budget summary (measured unit costs; no outcome is projected)

| item | GPU node-h | status |
|---|---|---|
| remaining | 1.474 | – |
| varied-γ verifier micro-benchmark (Path 1, b_γ) | ≈ 0.1–0.3 | not authorized |
| 2 more seeds of N ∈ {8,16,24,32} | ≈ 1.3 | not authorized |
| certify one x-value (Path 2 demo, current scheme) | ≈ 10–14 | unaffordable |
| scheduler pilot (Path 3, 4 runs) | ≈ 0.75 (+ profiling) | NO-GO |

## 4. Consolidated literature-verification questions (none answered here)

1. Is the Lindley deficit weighting distinct from virtual-token-counter fairness, least-attained-service and deficit round robin?
2. Are cohort staircases, N ≤ 2B selection vacuity and non-monotone x\*(N) known in bulk-service / closed batch-service networks?
3. How does "interactivity–concurrency capacity" differ from "goodput at SLO attainment" used by LLM-serving systems?
4. Do analytic speculative-decoding goodput models already include full-context verification cost and batch caps?
5. Is simultaneous per-client SLO certification with non-monotone feasibility new relative to active level-set estimation / sequential testing?

## 5. What not to do

Do not run a GPU scheduler pilot; do not add controller features; do not present Capacity-DPP, adaptive depth, pricing, knapsack or wireless as validated; do not cite any N\*(x) or ICA that was produced before C1 or from fewer than K = 5 run-seeds as certified.
