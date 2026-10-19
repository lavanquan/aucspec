# AUCspec scientific readiness audit (Task C0)

CPU-only. Code audited at `d330e88` (qualified execution `b7a1f66`), manuscript `aucspec-latex@2986a14`. No GPU job, no recalibration, no frontier, no pricing, no change to verifier semantics. Evidence below comes from reading the code and from read-only analyses of existing simulator logs (scripts in `scripts/audit_c0/`, outputs in `results/audit_c0_20261020/`).

## 0. Summary

Runtime H is correct (tier A). Everything the paper wants to *measure with it* is not yet sound: the measurement pipeline has several defects that can invalidate or mis-certify a boundary, the simulator's virtual-time batch formation is not causally complete, and three of the four control mechanisms the paper ablates (server price, queue-priced batching, bandwidth allocation) are inactive in the current configuration, while the fourth (depth) is driven by a unit-inconsistent device-queue price. None of this needs a new runtime investigation; most of it is CPU work.

**Top five scientific blockers**

| # | Blocker | Class | CPU-fixable? |
|---|---|---|---|
| 1 | No valid capacity evidence exists: every N*, ICA and calibration number was produced on the defective backend | evidence gap | the *fixes* are CPU; the evidence needs GPU (RQ1 pilot) |
| 2 | Measurement validity: workload-exhaustion guard mis-specified (also wrongly flagged the B6 smoke) and not enforced in search; classifier anti-conservative for K ≤ 7; seeds redraw the client population; search/ICA bounds not valid when points are uncertain | implementation bugs + estimand | yes |
| 3 | Virtual-time fidelity: 25–31% of verifier batches omit requests that had already arrived in virtual time; 6.7% of depth decisions see server-queue state from the virtual future | implementation bug | yes (design + tests); GPU re-validation inside the pilot |
| 4 | Joint-control mechanisms inert or ill-scaled: server price = 0, knapsack never binds, bandwidth payload ~0.03 ms, depth set by a device-queue price (ms² against tokens) producing a 4,0,4,0 limit cycle | modeling + implementation | partly (normalisation, instrumentation); calibration needs GPU |
| 5 | No physical grounding for the structural model on the corrected runtime: no calibration, roofline knee never observed, verifier cost dominated by per-request context work, isolated vs loaded gap ≈ +86 ms/batch unexplained | evidence gap | needs GPU (bounded) |

## 1. Task 1 — Frozen scientific runtime identity (CPU)

Frozen record: `results/runtime_identity/V1-batchwall-tritonfa33.json` (qualified SHA `b7a1f66fca5facb4af8dcc6f6100ec1096ab924d`, manifest sha256 `564571978a95c23c…`, Qwen2.5-7B target bf16 on one GCD, Qwen2.5-1.5B drafts on cuda:1–7, Triton 3.3.1, Triton FA in all 28 layers, V0 engine, `batch_wall`, `verifier_price_mode=zero`, `max_logprobs=1`, original V1 tie-break, prefix caching off). Evidence tiers, kept strictly separate:

| Tier | Contents | Citable as |
|---|---|---|
| A correctness-qualified | B6 qualification (dev 283 + held-out 96 gated checks, live kernel evidence) | the runtime is correct |
| B valid integration/timing diagnostics | wiring invariants of the smoke (470 batches, 97 with B>2, zero overlap, occupancy = modeled service, accounting) | wiring/serialisation correctness only |
| C invalid short-window throughput | rates from the 20 s window of `tfa33_wiring_smoke_r4_20261019` | nothing (about 11 rounds per client; ±1 round edge noise) |
| D legacy defective backend | all Gate B–D, N*(x) sweep, x=3.0 certification, smokes at `04d0bda`, calibration `b5a3def`, V1-control | defect evidence only; never as capacity or timing evidence |

**Erratum to my B6 report (see `results/CANONICAL_QUALIFICATION_ERRATUM.md`).** I wrote that 5 of 8 smoke clients "finished all six questions after 15.8–16.5 virtual seconds". That is wrong. The clients ran for 107–128 virtual seconds; the 15.8–16.5 s figure is the span between the first and last *in-window* rounds, which is smaller than the 17 s guard threshold merely because rounds last about 1.8 s (≈ 2 rounds of edge slack in a 20 s window). The INVALID verdict is a **false positive of the guard**, the recorded results are unchanged, and the "smallest fix" is the guard definition (M1), not a longer smoke.

## 2. Task 2 — Measurement pipeline audit (candidate (N, x) → ICA)

Traced path: `build_run_config` → `run_simulation.py` → per-round rows (`in_measurement_window` by virtual `client_receive_ms`) → `measure_candidate` (`compute_sustained_service_rates`, exact window denominator) → `classify_feasibility_final` → `feasibility_at_final` seed escalation → `monotone_scale_search_final` / `xstar_search` → `area_from_nstar` / `area_from_xstar` / `bootstrap_ica`.

Correct as implemented: exact fixed denominator `T_m` for every admitted client; clients with no tokens contribute 0; K=1 is "heuristic_only"; no `(1−ε)x` relaxation; cap-feasible is a lower bound; local re-check of N*−1, N*, N*+1; seeds run as separate jobs/processes with a cache keyed by (x, N, policy, seed, config hash).

| ID | Finding | Severity | Class | Minimal correction |
|---|---|---|---|---|
| M1 | **Exhaustion guard measures the wrong thing.** `span = max−min of in-window receive times < 0.85·T_m` is edge slack, not workload exhaustion: false positive when 2·(round time) > 15% of T_m (every window ≲ 60 s here, as in the B6 smoke), false negative when a client exhausts at ≥ 85% of the window (rate under-counted up to 15%). | HIGH | bug | Record in the simulator whether each client still had work at the end of the window (`workload_exhausted_before_end = no row with receive time > measurement_end`, or explicit completion time of the last assigned sample < end) and invalidate on that; keep the span only as a diagnostic. |
| M2 | **Exhaustion is not enforced in the search.** `measure_candidate` prints and stores `exhausted_clients`; `feasibility_at_final`/`_decision_from_measurements` ignore it, so an exhausted client's low rate can classify INFEASIBLE (the code comment itself names this risk). Only the certification harness invalidates. | HIGH | bug | Treat any exhausted measurement as INVALID: exclude from K, abort the search point with a "needs larger workload" status; never feed it to the classifier. |
| M3 | **Classifier is anti-conservative for the K actually used.** Percentile bootstrap of K seed values (K = 3 → 5 → 7, near boundary ≥ 5) at per-client level δ/N cannot exceed the sample range. Monte-Carlo (σ = 0.3, N = 10, 600 trials): per-client coverage vs nominal 0.995 is 0.76 (K=3), 0.91 (K=5), 0.96 (K=7), 0.97 (K=12); probability of declaring FEASIBLE when the weakest client's true mean equals x: 12.3% (K=3), 4.7% (K=5), 1.8% (K=7), 1.5% (K=12), against 0.17% for a Bonferroni Student-t interval (`bootstrap_classifier_operating_characteristics.json`). | HIGH (certification claim) | statistical bug | Replace by Bonferroni Student-t (df = K−1) per client, require K ≥ 5 before any "feasible/infeasible" is called certified (K = 3 only for screening); keep the bootstrap only as a sensitivity check. |
| M4 | **The seeds redraw the client population.** `random.seed(sim.seed)` then per-client RTT, uplink Mbps, SNR, draft speed/latency, SNR phase are sampled from wide ranges (RTT 15–100 ms, draft speed ×0.7–1.5, tps 20–60 …) in client order. Nesting in N holds *within* a seed, but "client i" at seed 1 and seed 2 are different devices, so the per-client confidence intervals average over different clients and the estimand is the ensemble over random hardware draws, not the fixed "nested master population" the manuscript describes. The population fingerprint mixes the seed. | HIGH (estimand) | design/bug | Separate `population_seed` (fixed per experiment: client hardware, prompts, SNR phases) from `run_seed` (timing/channel noise). State explicitly which estimand each table reports; add population resampling only for the robustness section. |
| M5 | **Invalid upper bound after an UNCERTAIN midpoint.** In `monotone_scale_search_final`, an uncertain `mid` sets `hi = mid`, and `upper_bound_n = first_infeasible if not None else hi`; an uncertain N is not an upper bound (the boundary may lie above it). `area_from_nstar` then uses that as `area_upper`, which can fall below the truth. | HIGH (ICA) | bug | Track the smallest *certified-infeasible* N; `upper_bound_n = that − 1` (or n_ref); treat uncertain points as gray: keep them for diagnostics and re-test with more seeds, never as bounds. (The manual x=3.0 reporting already used this rule.) |
| M6 | **Staircase area bounds use left-endpoint values for both bounds.** `n_at(x)` returns `ns[k]` on `[x_k, x_{k+1})`; since N*(x) is non-increasing this over-states N* inside the interval, so `area_lower` is not a lower bound between grid points and no discretisation gap is reported. | HIGH (ICA) | bug | Lower integral: N*_lower at the **right** endpoint of each interval; upper: N*_upper at the **left** endpoint; report the grid-discretisation gap separately. |
| M7 | `xstar_search`: an uncertain `x_lo` is returned as `lower_feasible_x`, an uncertain `x_hi` is treated as infeasible when bisecting; both feed `area_from_xstar`. | MEDIUM | bug | Same rule as M5 (certified endpoints only). |
| M8 | **No automated dual consistency check.** `analyze_final` never calls `area_from_xstar` nor compares N*(x) with x*(N); RQ1's "independent reconstruction" and the area-duality check have no code. | MEDIUM | gap | Add a CPU function comparing the staircase inverse of N*(x) with measured x*(N) intervals and the two area bounds, with unit tests on synthetic monotone data. |
| M9 | Monotonicity of feasibility in N is assumed for a *specific controller*. Larger N can improve batching efficiency (the manuscript says so), so feasibility need not be monotone; only neighbours of the found boundary are re-tested. | MEDIUM | design | Pre-registered non-monotonicity probe: test N*+2, N*+3 and one point at 1.5 N* for each headline x; report violations rather than silently repairing. |
| M10 | Window edges: rate counts whole rounds with receive time in the window. Unbiased for a stationary renewal process but noise ≈ ±1 round (±5–10% of a client's tokens at T_m = 20–60 s, ≈ 0.4% at 480 s). | LOW | approximation | Use T_m ≥ 120 s for any citable number; report per-client edge-round counts. |
| M11 | **Actual vs virtual time.** The virtual clock is assembled from measured components (draft GPU latency without queueing, physical batch wall time, modeled RTT/transfer). Real-time effects leak in: loaded batches are ≈ 86 ms longer than isolated calibration (cause unidentified), and real execution order decides which requests are present when a batch forms (see D1). | MEDIUM | approximation + bug D1 | Declare "hybrid real-time/virtual" in the paper; report loaded-vs-isolated residuals; fix D1. |
| M12 | **Run length = workload length, not window length.** Clients keep going until all assigned prompts are consumed (e.g. 107–128 virtual seconds for a 20 s window; 1,558 s for the 40-question workloads of the certification runs), so cost is dominated by unused post-window time. | MEDIUM (cost) | design | Optional stop `measurement_end + margin`, with M1's exhaustion flag recorded at the stop (behaviour inside the window unchanged). |
| M13 | "Independent job launches": candidate seeds do run as separate jobs, but no launch-level replication is done and all jobs share the same node type/software. | LOW | scope | Keep the manuscript's wording to "separate process launches"; add launch-level repeats only at headline boundary points. |

ICA bounds when boundary points stay uncertain: with M5 and M6 fixed the lower/upper step integrals are valid; until then ICA bounds must not be quoted. Dual consistency (M8) is untested code. Consequence for existing numbers: the x = 3.0 interval N* ∈ [10, 11] was derived by hand with the correct rule, but it is a defective-backend result (tier D).

## 3. Task 3 — Joint-control validity (what is active in code)

Evidence: read of `controller.py`, `verification_queue.py`, `verification_batcher.py`, `simulator.py` and read-only analysis of the B6 smoke log (N = 8, Capacity-DPP, 826 rounds; tier B/C use only for *activity* of mechanisms, not rates).

| Mechanism | What the code does | Observed activity | Decisions that genuinely change |
|---|---|---|---|
| Service deficit Z_i | benefit weight (V + Z_i) in the depth score and knapsack value; Lindley update per slot | Z_i positive in 99% of rounds (mean 68, max 113) | depth only through the *ratio* of benefit to the device-queue price; no direct effect on batching because values are all positive and constraints do not bind |
| Server queue Q_s | Lindley on measured batch service in 100 ms slots; priced as `Q_s·θ_f·(γ+1)` | Q_s positive in 93% of rounds (mean 494 ms) but `θ_f = 0.0` stamped on every request (0% non-zero) | **none** — it is logged and updated, never priced (fixed-budget zero price) |
| Verifier cost pricing | `θ_f` from roofline signal; fixed-budget mode returns 0 | inactive by construction | none; the physical verifier work (context recompute) is not represented in any priced quantity |
| Device queue Q_i | `Q_i·γ·(τ_d + κ/r)` in the depth score | positive in 40% of rounds (max 86) | **the only active depth driver**: selected γ ∈ {0, 4} only (328 vs 498 of 826); per sample, round r alternates 4, 0, 4, 0 (device queue 13.0 after a γ = 4 round, 0.0 after γ = 0); corr(γ, α̂) = −0.03. Price (ms) × cost (ms) is compared with (V+Z)·ΔΦ in tokens: units are inconsistent and the switching point is set by the arbitrary scales V = 50 and `slot_ms` = 100 |
| Network / bandwidth allocation | square-root uplink, queue-weighted downlink | mean upload 0.027 ms, download 0.009 ms against 299 ms draft and 62 ms RTT; greedy proposals are token ids (~100 B) | **none**: the allocator cannot affect any outcome in this regime (the RQ4 bandwidth ablation would be null) |
| Verification batch selection (`capacity_knapsack`) | exact DP on `capacity_dpp_value` under max batch 32 and token budget 100 | max token fill 0.30; no batch ever reached 32 requests; forced service 0% | **none**: with all values positive and constraints slack the DP selects every eligible request; batch composition is set by the fixed 150 ms wait window and arrivals |
| Acceptance learning (UCB/Bernstein) | optimistic α̂ feeds φ̂ | depth insensitive to α̂ (bimodal, device-queue driven) | negligible at present |
| Virtual-time causality | controller state updated in real processing order | **D1**: 25–31% of batches omit requests already arrived in virtual time (mean 162–185 ms early); **D2**: 6.7% of depth decisions see Q_s state that already includes a batch finishing 212 ms (mean) later in virtual time, and 47% of the values stamped on requests at submission do | D1 changes batch sizes and waiting times (active today); D2 is inert only because the server price is zero |

Classification.
- *Implementation correctness bugs (blockers for any claim about joint control)*: D1 batch formation not causally complete (`VerificationBatcher` forms a batch from whatever is present in real time); D2 controller state not time-consistent; the stamped server state is read after the draft `await`; exhaustion/classifier items of section 2.
- *Scientific modeling approximations (must be stated, not necessarily fixed)*: prices in mismatched units (Q_s, Q_i in ms against tokens); fixed 150 ms batch wait as a global constant; hybrid real/virtual clock and loaded timing; payload so small that the wireless coupling disappears; token budget counted in logical positions (γ+1) while the GPU does full-context recompute.
- *Optional improvements*: adaptive batch wait, processed-token budget, early stopping at the window end, acceptance-profile refinements.

Only the first category blocks the next experiments; the second blocks RQ3/RQ4 *interpretation* (a gain cannot be attributed to a price that does not exist).

## 4. Task 4 / 5 / 6

The RQ1 pilot design, its authorization conditions and cost are in `results/AUCSPEC_RQ1_MINIMAL_PILOT_PLAN.md` (section 5 there covers RQ3/RQ4 readiness). The claim–evidence matrix and figure plan follow.

## 5. Claim–evidence matrix (manuscript `2986a14`; sections 1–4 and the evaluation scaffold; there is no implementation section in `main.tex`, but the text cites `sec:implementation`)

| RQ / claim | Manuscript says | Evidence today | Status / flag |
|---|---|---|---|
| RQ1 capacity phenomenon: goodput hides concurrency loss; N*(x) vs Y at the boundary; dual x*(N) check | intro, §6.2, Fig. 1–2 TODO | none valid (tier D only); code for dual check missing (M8) | **unsupported**; do not state numbers; pilot proposed |
| RQ1 "statistically certified" boundaries | §4 Bonferroni classifier, exact boundaries need both neighbours certified | classifier anti-conservative for K ≤ 7 (M3); estimand mixes populations (M4); exhaustion unenforced (M2) | **unsupported** until M1–M6 fixed; the sentence "independent process/job launches" over-states the protocol (M13) |
| RQ2 affine fixed-depth law (Prop. 1) and 1/x scaling (Cor. 1) | §3, validated "against profiled Qwen/vLLM" | algebra is exact for the model; no measured fit on the corrected runtime | **model-only**; needs fixed-depth sweeps (pilot data) |
| RQ2 measured roofline knee | §3.2, Fig. 2a TODO | the only calibration (defective backend) found a linear regime up to 2,560 processed tokens, ΔBIC +11 against a knee | **unsupported — do not claim a measured knee** unless recalibration on the corrected runtime finds one |
| RQ2 "verifier-token efficiency decreases with depth" (Prop. 2) and crossovers (Prop. 3) | §3, Fig. 3 | mathematically true; physical cost per draft token measured ≈ 0.4 ms against ≈ 38 ms per request (defective runtime) because the verifier recomputes the full context; positions `γ+1` are not the physical work | **true but physically irrelevant as stated**; must be restated in terms of physical verifier work (full-context tokens) or limited to a stylised regime |
| RQ3 Capacity-DPP enlarges the achieved region | §6.4, main result | no corrected-runtime policy comparison; baselines' verifier-latency proxy was 0 in fixed mode | **unsupported**; needs M1–M6, D1–D2, price normalisation |
| RQ3 "asynchronous DPP guarantee" | Prop. 4 is stated for the synchronised system; text separates it from the implementation | the implementation is asynchronous with non-causal state (D1–D2) and no bounded-error argument | keep as **synchronised reference result only**; no asynchronous guarantee, no "optimal" wording |
| RQ3 existential capacity from one controller | text distinguishes achieved vs existential | none | already well hedged; keep "controller-achieved"; never infer N*_existential from Capacity-DPP |
| RQ4 ablations: depth, queue-priced batching, bandwidth, learning | §6.5 | server price 0, knapsack slack, bandwidth vacuous, depth device-queue driven, α̂ nearly irrelevant | **unsupported/vacuous** as configured; three of four ablations would measure nothing |
| RQ5 robustness to heterogeneity | §6.6 | population redrawn per seed (M4) so heterogeneity is confounded with seed noise | **unsupported** until population and run seeds are separated |
| Area/ICA and duality | §4 area bounds, equation `area-duality` | bound logic flawed (M5, M6); dual not analysed (M8) | **unsupported**; fix then re-derive |

Also flag: "profiled roofline abstraction with θ_f per verifier-token position" (intro) — the profile exists only on the defective backend; "physical verifier-token efficiency" — see Prop. 2 row; "queue-priced verification batching" — currently not price-dependent.

## 6. Figure-first minimum viable SIGMETRICS evidence plan

| Priority | Figure | Needed | Cost |
|---|---|---|---|
| 1 | **Fig. 1** goodput vs per-client floor / N*(x) (RQ1) — start with the pilot x_min(N), Y(N) on a homogeneous population, then the certified N*(x) on the corrected runtime | CPU fixes M1–M8, D1, early stop; ≈ 1.65 node-h pilot, later ≈ 10–20 node-h for a certified 4–6-point N*(x) | pilot ≈ 2 h GPU wall |
| 2 | **Fig. 2** mechanism: calibrated verifier cost (fixed overhead, per-request context cost, draft-token cost), fixed-depth x_γ(N) vs model | recalibration on the corrected runtime (~20 min, ≈ 2,000 calls, plus a loaded condition); fixed-depth sweeps for γ ∈ {0,2,4} | ≈ 1 node-h calibration + pilot reuse |
| 3 | **Fig. 3** depth crossover / regime change (only if the calibration shows the model terms matter) | Fig. 2 data | 0 extra |
| 4 | **Fig. 4** policy frontiers N_A*(x): Capacity-DPP vs the two strongest baselines | price normalisation, causal state, baseline proxy fix, certified search | largest GPU item |
| 5 | **Fig. 5** one boundary-focused ablation set (only mechanisms shown to be active) with decision-change rates | instrumentation of decision-change rate | small |
| 6 | **Fig. 6** heterogeneity with fixed population seed | M4 | moderate |

A minimum viable paper evidence set is Fig. 1 (RQ1), Fig. 2 (RQ2 grounded on the corrected runtime), Fig. 4 (RQ3 against baselines) and a reduced Fig. 5/6; the structural claims should be written to match what the physical calibration supports.

## 7. Prioritised roadmap

1. **CPU, now (no authorization needed beyond review):** M1/M2/M3/M5/M6/M7/M8 with unit tests; population-seed separation (M4); optional early stop and per-client exhaustion flag (M1/M12); causal batch-formation barrier and server-state ledger (D1/D2) with mock-event tests; analytical exhaustion budget (`scripts/audit_c0/rq1_exhaustion_budget.py`).
2. **GPU #1 (needs authorization):** RQ1 minimal pilot on the corrected runtime (≈ 1.65 node-hours, three 35–40-minute jobs), which also supplies its own valid integration evidence.
3. **GPU #2:** corrected-runtime verifier calibration (isolated and loaded), no pricing.
4. **CPU then GPU #3:** price normalisation, baseline proxies, decision-change instrumentation; one certified N*(x) point per policy to test the harness.
5. **GPU #4+:** headline frontiers and ablations only after steps 1–4.

Excluded by instruction: no new runtime investigation, no alternative attention backends, no tie-break change, no pricing activation, no merge to main.
