# AUCspec C4 — SIGMETRICS paper viability review

Reviewed: `lavanquan/aucspec` branch `research/c3-1-physical-feasibility` @ `e2e4eee` (C0–C3.1 evidence) and `lavanquan/aucspec-latex` @ `2986a14` (Introduction, System Model, Capacity Structure, Framework, Evaluation; the Implementation, Related Work, Discussion and Conclusion sections are empty headings, and the Evaluation section is a result-agnostic scaffold with `TODO` figures — it currently contains **no results**). The manuscript is not modified; every proposal below is separate.
CPU only. No GPU, no new controller feature, no pricing, no historical rewrite, no main merge. Companion files: `AUCSPEC_C4_SCHEDULER_FALSIFICATION.md`, `AUCSPEC_C4_MINIMUM_PUBLISHABLE_EVIDENCE.md`.

Status vocabulary (kept strict): **VERIFIED** (mathematically true or confirmed by tests/measurement as stated) · **CONDITIONAL** (true under assumptions that the evidence does not establish for the implemented system) · **UNTESTED** (proposal without evidence) · **CONTRADICTED / UNSUPPORTED** (evidence points against it, or the premise is false for the measured system).

## 0. Conclusion

**CONDITIONAL** — AUCspec as currently scoped (joint speculation-depth / queue-priced batching / wireless control with a drift-plus-penalty story, structural propositions on a roofline reference model, and a certified ICA) is **not ready for SIGMETRICS** because its controller headline is unsupported or contradicted and its evaluation has no results. A different, narrower paper — *measuring and explaining the interactivity–concurrency capacity of verifier-centric speculative serving under finite batch caps and stateless full-context verification, with a certified-measurement method* — has a defensible core, but only if one specific result is produced (§5): a theorem-level, independently validated finite-cap capacity characterization (Path 1), and/or a sample-efficient certified-boundary method with proven error control and measured savings (Path 2). The smallest next research task is CPU-only (§6).

## 1. What is actually established (evidence tiers)

| tier | content | citable as |
|---|---|---|
| A | runtime H is correct (B6 qualification) | the verifier is correct |
| B | wiring/serialisation/causality of the hybrid emulator (C1: causal virtual time, exact occupancy, population/run-seed separation, lifecycle validity, classifier code); ≈ 590 CPU tests (Setonix per-module run) | methodology / implementation correctness |
| C | **descriptive RQ1 data**: one run-seed, one catalog (GSM8K-train), γ=2, FCFS, homogeneous hardware, cap-16: N = 8, 16, 24, 28, 32 (+ test-split pilot N = 1…16, ext N = 12, 16) | a single-seed characterisation curve; **no confidence statement** |
| D | all pre-C1 N\*(x), ICA, calibration and certification numbers (defective backend) | defect evidence only |
| CPU | C2/C3/C3.1/C4 mechanism studies (real scheduler code, CPU verifier stand-in) | model-derived mechanism evidence, never GPU capacity |

**Descriptive vs certified.** The RQ1 points are descriptive: N = 24 failed the original real/virtual gate (3.07 > 3.0) and N = 28, 32 would have (3.21, 3.28); they carry the label `exploratory_v2`. They show x_min 3.43 (N=16) → 3.28 (24) → 2.89 (28) → 2.58 (32) while goodput goes 56.4 → 81.6 → 84.2 → 86.5 tok/s (+6 % from N=24 to 32 for −21 % in the floor) — consistent with the paper's motivating observation, from **one seed**. **No point in the repository is confidence-certified**: the Student-t simultaneous classifier needs K ≥ 5 (typically 8–12) independent run-seeds per N, ≈ 10–14 node-h per x-value (C1.4 estimate) against 1.474 node-h remaining.

## 2. Claim / evidence matrix

### 2.1 Introduction (contributions and motivating statements)

| # | claim (location) | status | evidence |
|---|---|---|---|
| I1 | Goodput saturates while the per-user floor keeps falling; a capacity view exposes it (§1) | **CONDITIONAL — descriptive support** | RQ1: +6 % goodput vs −21 % floor, N=24→32, single seed, exploratory_v2. Expected from any saturated closed system, so it is a motivating observation, not a result |
| I2 | Contribution 1: capacity region, downward closure, dual views, exact area decomposition (§1, §2.3, §2.5) | **VERIFIED (math, definitional)** for the existential region; **CONDITIONAL** for measured regions | duality/area identity are identities; Lemma 1 relies on monotone service times. Measured counter-instance: pilot x_min(8) = 3.975 < x_min(12) = 4.033 (45 unexplained slow batches in the first run). Modest novelty: see Path 1 |
| I3 | Contribution 2: structural laws for a symmetric roofline model; depth crossovers; finite regime partition | **CONDITIONAL / premise contradicted** | §2.2 |
| I4 | Contribution 3: queue-based online controller coordinating depth, batching, bandwidth; "idealized DPP achievability" separated from the asynchronous implementation | **UNSUPPORTED (controller) / CONDITIONAL (theorem)** | §2.3 |
| I5 | Contribution 4: implementation with real Qwen/vLLM; Capacity-DPP beats speculative-serving baselines; contrasts capacity with goodput | **UNTESTED** | no baseline (TurboSpec, GoodSpeed, FixedSLO) has been run on the corrected runtime; no Capacity-DPP-vs-baseline GPU comparison exists |
| I6 | Abstract: "validate the analysis and controller using real speculative decoding" | **UNSUPPORTED** | analysis not validated on the corrected runtime; controller not validated |
| I7 | "A scheduler that greedily favors high-acceptance requests can starve a lower-acceptance client and move outside the capacity region" (§1) | **CONDITIONAL (model-derived, small)** | C3/C3.1: density ranking lowers the floor (Δx_min −0.17 tok/s at N=48,B=16 in the wide population) but only when the cap binds; in grounded populations effects are ±3 % |

### 2.2 System Model and Capacity Structure

| # | statement | status | evidence / reason |
|---|---|---|---|
| S1 | Batch constraints \|B\| ≤ B_max, Γ = Σ(γ_i+1) ≤ G_max (§2.1) | **CONDITIONAL — mismatched with the physical constraint** | the engine limit that binds is *processed tokens* ctx+γ ≤ 4096 (stateless full-context verification, prefix caching off); Γ in positions never binds (logical budget 100 vs physical 4096); memory is driven by processed tokens × vocabulary (cap-16 headroom ≈ 0.5 GiB) |
| S2 | Lemma 1 (downward closure) | see I2 | – |
| S3 | Gated symmetric cycle with batch size = N (§3.1) | **CONDITIONAL** | with B_max < N the cycle contains ⌈N/B_max⌉ batches and a fixed overhead each; measured cohorts are 16+8 (N=24), 14+14 (28), 16+16 (32). Not in the reference model |
| S4 | Roofline verifier T^v = θ₀ + max{T_mem, T_comp(θ_f N(γ+1))}; fitted knee (§3.2, §6.2.1) | **UNSUPPORTED on the measured system** | C0: roofline knee never observed; C2: wall ≈ 52 ms + 0.19 ms × processed full-context tokens (R² 0.74; leave-one-run-out MAE 46 ms vs 63 ms for θ_f(γ+1)); the positions model is no better than batch-size alone; marginal request cost depends on context (12→92 ms for ctx 100→450) while θ_f(γ+1) does not |
| S5 | Prop. 1 (fixed-depth law on an affine branch) | **VERIFIED (math); CONDITIONAL (applicability)** | the measured system *is* affine in N within a cohort structure, so the form survives with different constants plus the ⌈N/B⌉ overhead term. Held-out check (C4): constants fitted on N ≤ 16 predict goodput at N=24/28/32 with −1.7 / −3.5 / −6.0 % error (one seed; the intercept is unstable: 11 ms vs 52 ms between fits) |
| S6 | Cor. 1 (Θ(1/x), goodput saturation) | **VERIFIED (math); descriptive support (I1)** | – |
| S7 | Prop. 2 (efficiency per verifier-token position strictly decreases with depth) and the reading "deeper speculation decreases capacity" | **VERIFIED (math); premise CONTRADICTED** | cost is proportional to ctx, not to γ+1; with the preliminary affine full-context cost verifier-ms per useful token is *minimised at γ=4* for α ≥ 0.3 until the per-candidate-token cost b_γ exceeds 0.44 / 1.0 / 3.6 / 10 / 20 ms for α = 0.3 / 0.4 / 0.6 / 0.8 / 0.92. b_γ is **not identified** (γ ≡ 2 in all traces; M3 bootstrap CI [−1.7, 5.5]); a context-aware cost makes deeper speculation *more* attractive (C3 economics) |
| S8 | Prop. 3 / Cor. 2 (pairwise crossover, finite regime partition) | **VERIFIED (math); UNTESTED (empirically)** | requires the unmeasured γ-slope; no crossover or regime transition has been observed |
| S9 | Nonmonotone γ*(N) allowed; "test, not assume" (§3.4) | **VERIFIED as methodology** | – |

### 2.3 Framework

| # | statement | status | evidence |
|---|---|---|---|
| F1 | Z_i deficit queue, Lemma 2 (rate stability ⇒ floor) | **VERIFIED** | standard; Lindley recursion implemented and tested; at the measured loads Z ≈ 0 until saturation (mean 0.05 at N=8, 0.7 at 24, 8–33 at 28–32) |
| F2 | Server queue Q_s, device queue Q_i, θ_f(γ+1) priced depth score (eq. gamma-score) | **CONTRADICTED by C2** | production θ_f = 0 ⇒ the Q_s term is identically zero ⇒ Capacity-DPP ≡ constant γ_max (bit-identical to fixed γ=4 in homogeneous populations); Q_s is "last batch wall − slot", R² 0.998 → 0 vs actual wall as load rises, not backlog; Q_i acts only through a 5–10-slot-stale read and lowers x_min; activating Q_s in ms units forces single-request batches (94–99.9 %) |
| F3 | Queue-priced batch knapsack (eq. batch-knapsack) | **CONTRADICTED by C2/C3** | member sets identical to FCFS in 100 % of multi-request batches; the cap binds in 0.3 % of batches at N ≤ 32, B_max=16 — structurally, because eligible work ≤ N − \|previous batch\| ≤ B_max iff N ≤ 2·B_max |
| F4 | Square-root bandwidth allocation | **UNSUPPORTED for greedy SD** | with 96-byte payloads any allocator changes uploads by ≤ 2.1 ms (0.25 % of a round); matters only for full-distribution uploads (seconds), a protocol not used |
| F5 | Online acceptance learning with UCB/Bernstein | **UNTESTED on the corrected runtime**; CPU shows unoptimistic learned depth control is absorbed at γ=0 | the optimism is necessary; it was not exercised in C1 |
| F6 | Prop. 4 (idealised synchronised DPP achievability, O(1/V) gap) | **CONDITIONAL — reference system only** | standard Neely-type argument; assumptions (ε-slack stationary policy, bounded increments, same-state joint decision) do not hold for the asynchronous per-round implementation; the manuscript states this correctly, but then cannot claim the implementation is near-optimal. **No matching derivation exists** |
| F7 | Exact-SLO finite-run classifier with simultaneous per-client intervals ("Bonferroni-adjusted bootstrap") | **VERIFIED (CPU) but the text is outdated** | C0 showed the percentile bootstrap anti-conservative (P[declare feasible \| truth = x] = 12.3 % at K=3, 4.7 % at K=5, 1.8 % at K=7 vs 0.17 % for Student-t); C1 implemented Student-t Bonferroni. No GPU boundary has been certified |
| F8 | Monotone boundary search with local re-certification, interval-valued uncertain points, area bounds | **VERIFIED (CPU tests; C0 bugs M5/M6 fixed)**; **CONDITIONAL** on monotonicity | one non-monotone instance observed (pilot N=8); the probe for N\*+2, N\*+3, 1.5N\* is specified but not run on GPU |
| F9 | ICA | **UNTESTED** | no valid boundary → no valid ICA; all earlier ICA numbers are tier D |
| F10 | "The Z term is essential: it prevents a throughput-efficient scheduler from starving a harder client" (§4.1.3) | **CONDITIONAL (small, model-derived)** | C3/C3.1/C4: deficit-state ranking raises x_min by 3–5 % at one constrained cell and 0 elsewhere; stamped Z alone reproduces it; it does not increase the number of clients that meet the SLO (it decreases it at x_req=3) |

### 2.4 Evaluation research questions

| RQ | status | reason |
|---|---|---|
| RQ1 capacity exposes what goodput hides | **descriptive only** | single seed (see I1); dual reconstruction and ICA not done |
| RQ2 structural model explains the boundary | **UNSUPPORTED as designed** | roofline/θ_f(γ+1)/Prop. 2 premises contradicted; a different structural model (finite cap, full context) has a partial held-out check (S5) |
| RQ3 Capacity-DPP enlarges the achieved region vs TurboSpec/GoodSpeed/FixedSLO | **UNTESTED; mechanism evidence negative** | Capacity-DPP at θ_f = 0 ≡ constant γ_max; no GPU comparison exists; C3.1 GPU pilot is NO-GO |
| RQ4 which components matter | **answered on CPU, negatively** | depth inert at the assumed cost, batch pricing and bandwidth inert, deficit ranking +3–5 % in one cell |
| RQ5 heterogeneity / robustness | **UNTESTED** | C1 populations are GSM8K-only with identical hardware (acceptance sd 0.017, Jain 0.9995); cross-domain α under the frozen runtime is UNKNOWN |

## 3. Reconstruction of the paper's contribution list from the evidence

| original contribution | survives as |
|---|---|
| 1 capacity region + dual views + area | **definitions and measurement object** (KEEP), with the existential-vs-achieved distinction and a warning that measured feasibility may be non-monotone |
| 2 roofline structural laws | **replaced** by a finite-cap, full-context law whose form is Prop. 1 with ⌈N/B⌉ overheads (CONDITIONAL, §4 Path 1) |
| 3 Capacity-DPP joint controller + DPP theorem | **negative / diagnostic results** about what controllers can and cannot change (see §4 outline §6), not a contribution |
| 4 real-system evaluation vs baselines | **measurement study of the baseline policy (FCFS)** + methodology; baselines comparison NOT SUPPORTED |

## 4. Verdicts for the manuscript's components

| component | verdict | note |
|---|---|---|
| Capacity-DPP (as the named contribution) | **DROP** | Z-state kept as a diagnostic; the named controller ≡ constant γ_max in the configuration used for all GPU evidence |
| Adaptive speculation depth | **DROP** from the headline; **CONDITIONAL** on a measured b_γ | inert at the assumed cost; mis-specification costs up to −42 % goodput (C3) |
| Server pricing (Q_s, θ_f) | **DROP** | not backlog; unit-inconsistent; collapses batching when active |
| Knapsack batching | **DROP** | member sets identical to FCFS; cap binds only for N > 2·B_max |
| Wireless allocation | **DROP** (limitation) | irrelevant for greedy payloads |
| ICA | **CONDITIONAL** | keep as a summary computed only from certified bounds; none exists |
| Structural propositions | Prop. 1 **CONDITIONAL** (restated with finite caps); Prop. 2 **DROP** as a result (premise false for the measured cost; keep as a remark on per-position cost); Prop. 3 / Cor. 2 **CONDITIONAL** on measured b_γ; Cor. 1 **KEEP** (motivation); Lemma 1 **KEEP** with the monotonicity caveat |
| Asynchronous controller guarantee | **DROP** — Prop. 4 stays a statement about the synchronised reference system and must not appear as support for the implementation |
| Exact-SLO classifier / search / lifecycle validity | **KEEP** (core of the methodology; update bootstrap → Student-t text) |
| Hybrid causal virtual-time emulator | **KEEP** (methodology; state that clients are virtual and phase-locked, see threat in §5) |

## 5. Threats that apply to every path

1. **Lock-step is an emulator property.** In the valid traces all members of a cohort arrive within 0.02–0.04 ms and re-arrive 104.5 ms ± 4.6 ms after the batch finishes; the emulator has identical virtual clients. Real client populations are not phase-locked, and the C3 CPU study shows start offsets alone do not de-synchronise, whereas jitter does. A capacity law derived from cohort rotation may be an artefact of the measurement environment unless validated with de-synchronised arrivals.
2. **One seed, one catalog, γ = 2, homogeneous hardware.** Nothing generalises; the acceptance dispersion is 0.017.
3. **b_γ unidentified**, so any γ-dependence of capacity is an assumption.
4. **Budget:** 1.474 node-h remaining; a single certified x-point is ≈ 10–14 node-h.
5. **Literature positioning is unverified** (see the questions in `AUCSPEC_C4_MINIMUM_PUBLISHABLE_EVIDENCE.md`): "goodput at an SLO" (DistServe-style), fairness schedulers (VTC), bulk-service and closed queueing networks may already contain the paper's structural claims.

## 6. Proposed alternative outline (separate from the manuscript; history untouched)

**Working title:** *Measuring and Explaining the Interactivity–Concurrency Capacity of Verifier-Centric Speculative Serving.*

1. **Introduction** — the object (capacity region, N\*(x), x\*(N)), why aggregate goodput hides it (descriptive RQ1 preview), and the three claims: (i) a measurement methodology with proven error control; (ii) a finite-cap, full-context structural characterisation; (iii) an honest account of what control can and cannot change.
2. **System and capacity model** — region, existential vs controller-achieved inner regions, downward closure *with its monotonicity condition and a measured counterexample*, dual views, area identity.
3. **Measurement methodology** — hybrid real-verifier/virtual-client emulator, conservative causal virtual time (D1/D2 invariants), exact occupancy, population/run-seed separation, workload-lifecycle validity, exact-SLO Student-t simultaneous classifier with INVALID/UNCERTAIN states, non-monotone-aware boundary search, interval-valued ICA. Validation: Monte-Carlo operating characteristics (CPU), causal-ledger tests, determinism under permuted scheduling.
4. **Structure of capacity under finite batch caps and stateless full-context verification** — cost model T = a + Σ(b·ctx + b_γγ); closed-form saturated law with ⌈N/B⌉ overheads; cohort quantisation (staircase in N); the structural fact that batch-selection policies are vacuous unless N > 2·B_max; the role of b_γ; conditions for the law to hold under de-synchronised arrivals. *(Needs the Path 1 result.)*
5. **Empirical characterisation** — RQ1 (descriptive and, when budget allows, certified), held-out validation of the law, goodput-vs-capacity contrast, sensitivity to catalog/seed.
6. **What serving control can change (negative and conditional results)** — baseline correction (work-conserving formation is worth 10–15 % under asynchrony and must be in every baseline), adaptive depth inert unless b_γ ≳ 1–5 ms/token, deficit-aware ranking +3–5 % only where the cap binds and not in SLO satisfaction, bandwidth irrelevant for greedy payloads.
7. **Related work; limitations** (virtual clients, lock-step, γ=2, one catalog, 1.474 node-h).
8. **Conclusion.**

**Figure order:** F1 goodput vs per-user floor (descriptive → certified); F2 measured vs predicted cycle/capacity (held-out N); F3 cohort staircase and non-monotone x\*(N) cases; F4 selection leverage map in (N, B_max); F5 certification operating characteristics (CPU Monte-Carlo, error rate vs K, savings of the adaptive scheme); F6 control-effect summary (paired CPU deltas, clearly labelled model-derived).

## 7. Smallest next research task (CPU-only)

State and prove the saturated and unsaturated capacity law for a closed system with batch cap B, affine per-request cost and cohort dynamics (including the staircase and its non-monotone points), verify it exactly against a CPU discrete-event simulation, evaluate it leave-N-out on the existing seven GPU runs, and identify the conditions under which de-synchronised arrivals break it. Details, success and disconfirming criteria: `AUCSPEC_C4_MINIMUM_PUBLISHABLE_EVIDENCE.md` §2.
