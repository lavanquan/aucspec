# AUCspec — Final research disposition (measurement-methodology novelty veto)

Branch `research/final-disposition` (base `research/c5a-novelty-gate` @ `1d44144`). Literature and claim assessment only: no GPU, no simulation campaign, no implementation change, no manuscript edit, no merge.

## Conclusion: **ARCHIVE FOR SIGMETRICS**

None of the candidate methodological claims survives the primary-literature veto as a *precise, falsifiable, not-already-established* statement that can be evaluated with the resources that exist (≈ 1.474 GPU node-hours; one-seed descriptive GPU data). What remains is a useful, correct artifact (a causally-consistent hybrid emulator, a measurement-validity audit, an exact-SLO classifier with known operating characteristics) and a body of negative findings. That is not a full-paper contribution for SIGMETRICS within current resources. §5 gives a clean stopping point; no new roadmap is proposed.

## 1. Reading limitations (stated, not assumed)

| source | how read | status |
|---|---|---|
| Wang et al., "Revisiting Service Level Objectives and System Level Metrics in LLM Serving", arXiv:2410.14257v2 (Oct 2025) | full text extracted | **read** |
| Zhong et al., "DistServe", OSDI 2024, arXiv:2401.09670v3 | full text extracted in C5-A (§1 goodput, §4 Alg. 1) | **read** |
| "Vidur: A Large-Scale Simulation Framework for LLM Inference", arXiv:2405.05465v2 (MLSys 2024) | full text extracted (§6 Vidur-Search, §7 fidelity) | **read** |
| Feng et al., "Frontier", arXiv:2605.21312v2 | full text extracted; skimmed for validity/causality/statistics | **read (skim)** |
| Howard, Ramdas, McAuliffe, Sekhon, "Time-uniform, nonparametric, nonasymptotic confidence sequences", Ann. Statist. 2021, arXiv:1810.08240v9 | full text extracted (Thms 1, 2, 4, Prop. 1) | **read** |
| Karp & Kleinberg, "Noisy binary search and its applications", SODA 2007 | full text (third-party PDF copy) | **read (abstract + statements)** |
| Dereniowski, Łukasiewicz, Uznański, "Noisy (Binary) Searching: Simple, Fast and Correct", arXiv:2107.05753v4 | full text extracted | **read (abstract + intro)** |
| Frazier, Henderson, Waeber, "Probabilistic bisection converges almost as quickly as stochastic approximation", arXiv:1612.03964v1 (published Math. Oper. Res. 2019) | full text extracted | **read** |
| Cheshire, Ménard, Carpentier, "The Influence of Shape Constraints on the Thresholding Bandit Problem", COLT 2020, arXiv:2006.10006v3 | full text extracted (Table 1) | **read** |
| Gotovos, Casati, Hitz, Krause, "Active Learning for Level Set Estimation", IJCAI 2013 | abstract and search summary only | **not read in full** |
| Fujimoto, courseware slides "Deadlock Avoidance Using Null Messages — Chandy–Misra–Bryant Algorithm" (SIGSIM) | extracted slide text (≈ 4 kB, assumptions and Algorithm A only) | **read (partial)** |
| Chandy & Misra (1979), Bryant (1977), Fujimoto (1990 CACM), Lan & DeMets (1983), Johari et al. (always-valid inference), Georges et al. (2007), Hurlbert (1984), Law & Kelton, DieCast/SliceTime-type time-dilation emulators | not accessible or not consulted | **not read — questions only** |

Anything in the "not read" row is never used below as evidence that a claim is or is not novel.

## 2. Separating engineering components from potential contributions

| component (AUCspec) | what it is | what the literature already establishes (read) | status |
|---|---|---|---|
| Conservative virtual-event causality and determinism: per-client lower bound on the next arrival (t + RTT/2), a batch with cutoff C formed only when every idle client's bound exceeds C, decisions wait for horizon > t | an application of conservative parallel discrete-event simulation with lookahead | Fujimoto slides: Chandy–Misra–Bryant assumes timestamp-ordered links and a **lower bound on the timestamp (LBTS)** of subsequent messages; null messages / lookahead avoid deadlock; goal: events processed in timestamp order. The coordinator is this scheme with lookahead = network delay. The determinism property is a consequence of the scheme | engineering component; the invariant is established, not new |
| Measurement-window validity and workload non-exhaustion | a validity guard replacing a span heuristic (C0 M1/M2) | Vidur §7 explicitly evaluates only at 85 % of capacity because near capacity errors "blow up" (its §7 text on dynamic workloads) — validity near the boundary is a recognised issue; steady-state/warm-up handling is textbook simulation output analysis (not read) | engineering/hygiene |
| Simultaneous per-client confidence bounds (Student-t, Bonferroni, intersection–union rule: feasible iff min_i L_i ≥ x, infeasible iff min_i U_i < x) | standard multiple-testing construction | Howard et al. give *time-uniform* alternatives (Thm 1 stitched boundaries, Thm 2 mixtures, Thm 4 empirical-Bernstein) whose property (P2) is "no final sample size has to be chosen" — i.e. sequential escalation without inflating error | known; the small-K failure of the percentile bootstrap (C0) is a bug-level finding about our own earlier code |
| Adaptive boundary search with uncertain decisions | bisection with repeated measurements and an UNCERTAIN state | **noisy binary search**: Karp–Kleinberg (monotone coin probabilities; Θ(log n/ε²) type bounds, abstract), Dereniowski et al. (query complexity); **probabilistic bisection with a power-one test at each queried point** (Frazier–Henderson–Waeber, arXiv:1612.03964: Assumption 1 monotone root, exponential-in-n convergence of the median in the number of completed power-one tests); **capacity search by binary search**: Vidur-Search §6 ("a simple binary search which searches for the maximum QPS which does not increase the scheduling delay beyond a threshold"), DistServe Alg. 1 ("finds the maximum rate that meets the SLO attainment target with binary search and simulation trials") | established in structure; our version adds an UNCERTAIN state and a per-client intersection rule |
| Nonmonotone policy-achieved feasibility | search must not assume monotonicity | **thresholding bandits**: Cheshire–Ménard–Carpentier Table 1 gives minimax simple-regret rates for the unstructured, monotone, unimodal and concave cases (unstructured ≈ √(K log K / T), monotone ≈ √(log K / T)): the price of *not* assuming monotonicity is characterised; LSE (Gotovos et al., abstract) addresses unstructured level sets with confidence bounds | the theory is known; our *empirical* evidence of nonmonotone achieved feasibility is weak (C5-A: the pilot reversal 3.975 vs 4.033 = 1.5 %, same order as cross-launch replicates 0.2–1.2 %, and a documented first-run anomaly; the DES counterexamples are constructed policies) |
| Population-level vs execution-seed uncertainty | estimand and replication unit | the distinction is pseudo-replication / variance components (standard; not read). Our own data: GPU cross-launch replicates 0.2–1.2 %; population variance on GPU is **unmeasured** (one population per catalog) | statistical hygiene; no evidence |
| SLO metrics for streaming users | per-client sustained rate ≥ x | **Wang et al. (arXiv:2410.14257)** already formalise user-experience SLOs for streaming: token deadlines defined relative to the request's commitment (not the previous token), reading speed (their Fig. 2 uses 4.5 tokens/s), and *smooth goodput*; they show two ways existing SLO/goodput metrics can be gamed (delaying tokens; abandoning requests) | our "per-client sustained rate" is a coarse, window-averaged cousin; closed-loop N-stream capacity differs from their open-loop request goodput, but the user-experience rationale is theirs |
| LLM-serving simulators | trace-driven/predictive | Vidur (<9 % request-level error; evaluated at 85 % of capacity), Frontier (disaggregated DES, scheduler-batch-engine loop incl. speculative decoding); they predict cost, they do not provide statistical certification of boundaries | adjacent; the certification layer is where AUCspec differs, see §3 |

**Not novel merely because combined.** Every row above is either an application of an established scheme or a standard statistical construction; the combination is a careful engineering pipeline.

## 3. Falsifiable novelty statements — each tested against the literature gate

For each candidate: **A** precise claim not already established; **B** assumptions, estimand, unit of replication; **C** fair baseline; **D** guarantee/independent result required; **E** can it be evaluated with accessible resources; **F** disconfirming result.

### Candidate 1 — Causal hybrid emulation changes capacity conclusions
* **A.** "In an emulator that advances virtual time with *measured* batch service times, forming batches in real arrival order biases the estimated capacity by Δ relative to a conservative-PDES (LBTS) coordinator, and the coordinator removes the bias." The coordinator and its determinism are CMB/Fujimoto; the only unestablished piece is Δ.
* **B.** Estimand: N\*(x) or x_min(N) for a fixed policy and fixed population; unit: population (run-seeds only repeat near-deterministic greedy trajectories).
* **C.** Real-order mode (already in the code, flagged) on the same population.
* **D.** Paired measurements in both modes on the corrected backend over several populations.
* **E.** The only evidence is from the pre-C1 backend (C0: 25–31 % of batches omitted already-arrived requests; 6.7 % of depth decisions read virtual-future state): tier D, a defect measurement of earlier code, not a result about emulators. A fresh Δ needs GPU or a simulation campaign — **not available/not authorised**.
* **F.** |Δ| within cross-launch noise (≈ 1 %) ⇒ no claim.
* **Gate result: fails** — the general statement is textbook PDES; the specific Δ is a property of this implementation and is unmeasured.

### Candidate 2 — Sequential, simultaneous-SLO certification saves runs
* **A.** "A confidence-sequence classifier (time-uniform per-client bounds) embedded in a power-one-test bisection over a possibly nonmonotone feasibility map certifies N\*(x) with a stated family-wise error and uses fewer runs than the fixed-K Student-t scheme." Every ingredient is prior art (Howard Thms 1/2/4; FHW power-one bisection; Cheshire et al. for the nonmonotone cost; Vidur/DistServe for capacity search). The guarantee for the composition would follow from a union (alpha-spending) argument over the search path — a routine corollary, not a new theorem. The *saving* is the only open item and it is an empirical engineering question without prior evidence.
* **B.** Estimand m(N, x) = min_i μ_i(N, x) with μ_i the **population-conditional** expected rate of client i; unit of replication: run-seed *conditional on the population* — so the certificate speaks about that fixed population only.
* **C.** (i) the existing fixed-K Student-t Bonferroni bisection; (ii) a Vidur/DistServe-style binary search without error control.
* **D.** A proof of family-wise error control on the search path (routine) **and** an independent result showing the saving at equal error — requires a ground-truth simulator study over many synthetic boundaries (including nonmonotone) plus one real certified point.
* **E.** The simulation study is a CPU campaign (excluded here); the real certified point costs ≈ 10–14 node-h per x-value versus 1.474 remaining — **an unfunded dependency**, and the saving itself would have to be demonstrated before it can be assumed to reduce that cost.
* **F.** No saving at equal family-wise error against fixed-K, or the guarantee requires monotonicity the data violate.
* **Gate result: fails** — no un-answered *scientific* question after the literature gate; what remains is an unevaluated engineering optimisation.

### Candidate 3 — Run-seed replication certifies the wrong estimand
* **A.** "With deterministic greedy verification, run-seeds are near-replicates, so K-seed intervals certify only the population-conditional rate; population-level capacity claims need P independent populations and a variance-component analysis." This is pseudo-replication / variance-component reasoning (standard; unread sources), applied.
* **B.** Estimand: E_population[x_min(N)]; unit of replication: population draw (prompt block + client configuration), P ≳ 5–20; run-seeds nested within population.
* **C.** The C1 design: K run-seeds, one population.
* **D.** Measured between-population vs within-population variance components on the real verifier.
* **E.** The GPU data contain one population per catalog; the only replicate information is 0.2–1.2 % cross-launch noise; population variance is unmeasured (a P-population campaign ≳ 6–7 node-h per cell by earlier estimates) — **unfunded**.
* **F.** Population variance ≈ run variance (≲ 1–2 %) ⇒ the distinction is empirically irrelevant here.
* **Gate result: fails** — standard statistics, with no evidence in hand.

### Not a candidate
*Nonmonotone feasibility.* Established theory (Cheshire et al., Table 1) and no robust empirical instance; the one reversal in the data is within noise and tied to a documented anomaly. *Finite-batch-cap capacity law:* stopped in C5-A.

## 4. Resource constraint

≈ 1.474 GPU node-hours remain; the C1 points are single-seed descriptive and stay so. Any claim needing a *certified* GPU point (≈ 10–14 node-h per x-value in the C1.4 estimate) or a multi-population campaign (≈ 6–7 node-h per cell) is reported as an **unfunded dependency**, not a plan. No saving from a sequential method is assumed: that saving would itself need the evidence listed under Candidate 2-E.

## 5. Clean stopping point

1. Treat the repository as an archived artifact. The reproducible, correct parts are: the causal virtual-time coordinator and its tests, the measurement-validity and classifier code with its Monte-Carlo operating characteristics, the descriptive RQ1 data (labelled single-seed, exploratory_v2 where applicable, original-gate failures preserved), the C0–C5-A reports.
2. What may truthfully be said about the work: (a) a single-seed descriptive curve shows goodput saturating while the per-user floor falls (N = 24 → 32: +6 % vs −21 %); (b) an audit of measurement-validity defects in an earlier pipeline and their fixes; (c) CPU-model evidence that the proposed joint-control components are inert or small under measured conditions; (d) a deterministic cohort-accounting identity that reproduces the GPU cycle times when measured batch times are used.
3. What must not be said: a certified capacity boundary or ICA; a validated controller; a structural capacity theorem as a contribution; novelty of the measurement method as a whole.
4. No further GPU spending is recommended on this line; the remaining 1.474 node-h are not enough to change the disposition.
5. If a short experience/lessons write-up is ever wanted, it is limited to the statements in item 2 and must carry the reading limitations above; this document does not propose it as a research direction.
