# AUCspec C5-A — Literature veto and finite-cap theorem feasibility gate

Branch `research/c5a-novelty-gate` (base `research/c4-paper-viability` @ `e221070`). CPU only; no GPU, no calibration, no controller work, no manuscript edit, no historical-result change, no main merge.
New code: `scripts/c5a/{mini_des,des_claims,trace_comparability,heldout_check}.py` (an **independent** closed-system DES that shares no code with the project simulator), `legacy/tests/test_c5a_counterexamples.py` (19 tests); results `results/c5a/`.

## 0. Recommendation: **STOP**

The "finite-cap saturated capacity law" is (i) an instance of a standard deterministic cycle-time computation (max-plus / timed-event-graph cycle mean; operational analysis of a batch server), valid on a **narrow** domain (N ≤ 2B, deterministic, identical return delay, fixed hold rule, persistent cohorts), (ii) **false as stated in C4 outside that domain** (the ceiling law over-predicts the cycle for N > 2B; clients are served at different frequencies there, so no single cycle exists), and (iii) not distinguishable from the closed-queueing / bulk-service literature that I could verify — no source I read treats this exact deterministic regime, but none of them needs to for the result to be elementary. The one regime where the dynamics are not elementary (N > 2B, heterogeneous delays, unequal service frequencies) has no clean law, is not covered by any existing GPU trace, and is exactly where the C3.1/C4 scheduling study found only 3–13 % effects. Recommendation: stop the theorem development; keep the validated *accounting* (it explains the existing RQ1 data to < 1 % when measured batch times are used) as an explanatory section of a **methodological case study**; do not claim a structural-capacity contribution. Reasons and the explicit alternative in §7.

## 1. Literature (primary sources first) — reading status stated, not assumed

| source | read how | status |
|---|---|---|
| Kar, Rehrmann, Mukhopadhyay, Alt, Ciucu, Koeppl, Binnig, Rizk, "On the Throughput Optimization in Large-Scale Batch-Processing Systems", *Performance Evaluation* 144 (2020), arXiv:2009.09433 | full text extracted (§§1–5, App. A) | **read** |
| Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving", OSDI 2024, arXiv:2401.09670v3 | full text extracted (§1, §4 Alg. 1) | **read** |
| Sheng et al., "Fairness in Serving Large Language Models", OSDI 2024, arXiv:2401.00588v2 | full text extracted (Alg. 2, Def. 4.1–4.2, Lemma 4.3, Thm 4.4/4.8/4.13) | **read** |
| Ramani & Tantawi, "An Approximate Queueing Model of LLM Inference Serving for SLO-Driven Autoscaling", arXiv:2609.20957 | full text extracted | **read (model sections)** |
| Li, Dai, Peng, "Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents", arXiv:2504.07347 | HTML via summarising fetch (model, Thm 1) | read through a summary (first 100k characters) |
| "A Queueing Theoretic Perspective on Low-Latency LLM Inference with Variable Token Length", arXiv:2407.05347 | HTML via summarising fetch (Eqs. 1, 14, 18, 24–26) | read through a summary |
| Inoue, "Queueing analysis of GPU-based inference servers with dynamic batching: a closed-form characterization", *Perf. Eval.* 147 (2021), arXiv:1912.06322 | **abstract only** | **not read** |
| Bailey, "On Queueing Processes with Bulk Service", *J. Royal Stat. Soc. B* 16(1):80–87 (1954) | JSTOR not accessible | **NOT READ**; known only through secondary summaries (Wikipedia "Bulk queue"; Kar et al. §2) |
| Stankevich, Tananko, Dolgov, "Analysis of Closed Queueing Networks with Batch Service", *Izv. Saratov Univ. Math. Mech. Inform.* 20(4):527–533 (2020), doi:10.18500/1816-9791-2020-20-4-527-533 | full text blocked; abstract/summary only | **not read** |
| Van Dijk & Smeitink, "A nonexponential queueing system with independent arrivals and batch servicing", *J. Appl. Probab.* | abstract only | **not read** |
| Henderson, Pearce, Taylor, van Dijk (1990), "Closed queueing networks with batch services" | search summary only | **not read** |
| Baccelli, Cohen, Olsder, Quadrat, *Synchronization and Linearity* (1992) (max-plus cycle-time theorem) | not consulted | **not read — cited only as the place to verify** |
| LLM capacity tooling (vLLM `bench serve`, SGLang `bench_serving`, NVIDIA AIPerf, DataRobot profiling) | search summaries only | **secondary only**; DistServe Alg. 1 is the primary example read |

Anything marked "not read" below is a **question**, never a basis for a novelty or non-novelty claim.

### Comparison matrix

| topic | directly established (source, location) | general known results that would imply our special case | model differences | genuinely unresolved (after this gate) |
|---|---|---|---|---|
| Closed system with batching and per-job rate | Kar et al.: closed network, n clients each ≤ 1 job, batcher forms batches of exactly k, m servers, exponential times with k-dependent rates; CTMC (§3, eqs. (1)–(3)); mean-field limit (Thm 4.1(i)), exponentially fast convergence to the unique closed-form stationary point w\* (Thm 4.1(ii), eq. (9)), stationary measures converge (Thm 4.1(iii)); two job types (Thm 5.1) | product-form results for closed batch networks (Henderson et al.; Stankevich et al.; Van Dijk & Smeitink — **unread**) are the standard route; Kar et al. state product form fails for their model (§2) | exponential, large-n, **fixed batch size k** (no cap with partial batches), no hold time, no per-client rate floor, state-dependent routing; ours: deterministic, cap B with partial batches, hold W, return delay D | none of these treats the deterministic cohort regime; whether it is worth treating is §4 |
| Bulk service with a cap | Bailey (1954) is the classical bulk-service queue (**unread**); cap by Downton (1955) per a secondary source | operational analysis: busy-time per request = a/β̄ + d̄ for mean batch size β̄ (no theorem needed) | open Poisson arrivals vs closed clients | — |
| Affine batch time in LLM serving | Inoue (abstract): batch-service queue with batch-size-dependent time, closed-form latency **bound** (open); arXiv:2407.05347 Eq. 14 H[b]=αb+β, Eq. 18 H[b,l]=k₁b+k₂+k₃bl+k₄l, M/G/1 and M/D^b/1 (Eqs. 1, 24–26), no numbered theorems; Ramani–Tantawi: Markovian model with batch cap B, Poisson arrivals, validated on vLLM (H100) | Pollaczek–Khinchine, M/D^b/1 | all **open** arrivals; none has per-client token-rate floors or closed clients | closed finite-source version with cap and hold |
| Stability / throughput optimality | Li–Dai–Peng Thm 1(a)/(b): any work-conserving (K_p,K_d)-FCFS policy is stable iff λ(m_p+m_d) < b/t_b (open, discrete time, token-budget batches) | fluid / Foster–Lyapunov | open arrivals; token budget not batch cap; no closed-loop users | — |
| Capacity at an SLO | DistServe §1: per-GPU goodput = **maximum request rate** served within SLO attainment (e.g. 90 %); §4 Alg. 1 finds it by simulation + **binary search**; Fig. 1 | any SLO-capacity or load-testing methodology | open request rate λ, attainment fraction over requests; ours: closed N streams, **per-user token-rate floor** | whether a closed-N / per-user-floor capacity notion adds anything beyond "max sustainable concurrency" tooling |
| Fairness scheduling | VTC: Alg. 2 (virtual token counters), Def. 4.2 (|W_f−W_g| ≤ Δ for backlogged clients), Lemma 4.3, **Thm 4.4: ≤ 2·max(w_p L_input, w_q M)**, Thm 4.8 (≥ w_q M for any work-conserving scheduler), Thm 4.13 (non-overloaded clients) | least-attained-service, deficit round robin | VTC equalises *attained service* (with token weights); our Z_i is a Lindley deficit against a floor x | C4 already shows the effect carried by Z is a standard fair-share effect (+3–5 %) |
| Deterministic cyclic server | — | max-plus cycle-time theorem for timed event graphs (**unread, to verify**), plain operational laws | — | — |

**What this does and does not say.** I found no paper among those read that states Theorem A below or its corollaries for closed deterministic clients with a batch cap and a hold rule. That is **absence of a hit, not evidence of novelty**: the result is an application of a general cycle-time argument (§4) and the closed-network/bulk-service papers I could not read (Bailey, Stankevich et al., Van Dijk–Smeitink, Henderson et al.) are exactly where such special cases would appear. A paper not using the phrase "interactivity–concurrency capacity" proves nothing either way.

## 2. The mathematical claim, repaired

Fix a deployment ξ and nested populations I_N = {1..N}.

* **A. Existential capacity** x\*_exist(N) = sup over **all non-anticipating policies** (batching, hold, idling, depth) of min_i liminf_T E[tokens_i(0,T)]/T.
 **Lemma 1 (monotone).** x\*_exist(N+1) ≤ x\*_exist(N). *Proof.* Restrict an N+1-client policy to the first N clients, simulating the removed client's randomness internally; delete it from every batch and keep every original batch start and finish time (the server idles for the difference). Because T(S) = a + Σ_{i∈S} d_i is non-decreasing in S, deleting a member never lengthens a batch, so the idle padding is non-negative and every retained client's request/completion times are unchanged. ∎ (assumptions: batch time monotone in membership; idling admissible; test: `test_existential_restriction_lemma…`). **No nonmonotone existential claim is made or defensible.**
* **B. Achieved floor of a fixed policy** x_π(N) = min_i liminf E[tokens_i]/T under policy π. **Not monotone in general**: counterexamples (§3) — launch-when-full-or-timeout has R(16) < R(15); a threshold-k policy has x = 0 for N < k and > 0 for N = k. Monotonicity holds for hold-head FCFS in the deterministic domain (R(N) non-decreasing, tested for N = 1…40) but must be proved per policy.
* **C. Finite-window estimate** x̂_π(N; T_m) = min_i A_i/T_m — a random variable. A single sample reversal x̂(8) < x̂(12) is **not** evidence about A or B. Concretely (`results/c5a/trace_comparability.json`): the pilot reversal is 3.975 vs 4.033 (1.5 %), same catalog; cross-launch replicates of identical conditions (C1.2 ext vs C1.3 cap-16, same catalog/seed/γ/H, cap not binding) differ by 0.2 % (N=8) and 1.2 % (N=16); the N=8 pilot point is the first run of its job with 45 slow batches (documented in C1.1). It is an unexplained transient/outlier, not an expected-rate property.

### Exact conditions under which a saturated formula holds

**Theorem A (K ≤ 2).** Assume:
(A1) closed system, N clients, each client has at most one request, alternating *wait for service* → *batch completes* → *deterministic return delay D ≥ 0 (identical for all clients; device + RTT)* → *new request*; (A2) one server, atomic batches, batch time T(S) = a + Σ_{i∈S} d_i with a ≥ 0, d_i ≥ 0 deterministic (any per-client context/depth cost; heterogeneous d_i and acceptance φ_i allowed); (A3) batch cap B and **N ≤ 2B**, so K = ⌈N/B⌉ ≤ 2; (A4) hold-head work-conserving FCFS: with the server idle at t, the next batch starts at s = max(t, a_head + W) (W ≥ 0, a_head = earliest unserved arrival) and takes the first min(B, #arrived by s) requests in (arrival time, id) order; (A5) all clients start at t = 0 (ties by id); useful-token yield φ_i per round (any values, no distribution needed).
Then the batch partition is S₁ = {1..min(B,N)}, S₂ = the rest, **fixed from the first period on**, every client is served exactly once per period, and the period is
 R = max{ T₁ + T₂ , D + W + max(T₁, T₂) } (K = 2),  R = D + W + T₁ (K = 1, N ≤ B),
with T_j = a + Σ_{i∈S_j} d_i, so x_i = φ_i/R and the floor is x_π(N) = min_i φ_i / R.
The saturated law x = φ/(⌈N/B⌉a + Nd) holds **exactly iff** the server-limited term dominates: min(T₁, T₂) ≥ D + W, i.e. for identical d: **N ≥ N_c := B + (D + W − a)/d**. For N < N_c the cycle is client-limited, R = D + W + T₁ is **constant in N** (a plateau), not growing.
*Proof sketch.* Let s_j, f_j = s_j + T_j be the start/finish of the batch serving S_j in a period. Batch order and membership are preserved by FCFS because S₁'s members re-arrive at f₁ + D < f₂ + D (S₂'s), and at every start all members of the next batch have arrived (s ≥ f_j + D + W). The starts obey the max-plus recursion s₁′ = max(f₂, f₁ + D + W), s₂′ = max(f₁′, f₂ + D + W). A periodic solution with shift R exists with R equal to the maximum cycle mean of the precedence graph: the server loop (T₁ + T₂) and the two client loops (T_j + D + W). ∎ **Checked, not proved by simulation:** 352/352 parameter combinations with N ≤ 2B on the independent DES (`results/c5a/des_claims_summary.json` (rows regenerated by the script)) agree to < 0.5 %, with uniform service.

**Out of domain (N > 2B):** the formula with ⌈N/B⌉ holds in only 26 % of 480 tested combinations; the max-plus form with a fixed greedy partition in 61 %; a fractional-overhead form R = max{(N/B)a + Σd, D+W+T_full} in 93 %; service is uniform across clients in only 58 %. There is no single cycle time, hence no formula φ/R for the **floor**.

## 3. Counterexamples sought before the theorem (all executed; `legacy/tests/test_c5a_counterexamples.py`)

| # | test | outcome | false claim, or valid claim with incomplete assumptions? |
|---|---|---|---|
| 1 | N ≤ 2B, partial previous batch, > B eligible (N=20, B=16; 3 clients at t=0, 17 at t=10) | batch 2 starts at 142 ms with 17 arrived requests (16 served, 1 left over); settles into a (4, 16) rotation with eligible ≤ B | C3's "N ≤ 2B ⇒ cap rarely binds" is a **valid steady-state claim with an incomplete assumption** (needs |previous batch| ≥ N − B; false in start-up transients) |
| 2 | asynchronous start offsets / return jitter up to 100 ms, N = 24, 28, 32 | cycle identical to the lock-step value (824/…); jitter changes the partition but not R because the server never idles | the saturated law is **robust to arrival jitter** for N ≤ 2B when server-limited; it is not an emulator-lockstep artifact in that regime |
| 3 | heterogeneous return delays D_i ∈ [0, 300] ms, N > 2B (N = 40, B = 16) | per-client gaps 1064…1596 ms; mean-cycle law gives a floor that no client has | the single-cycle law is **false** beyond 2B |
| 4 | nonuniform contexts and acceptance (heterogeneous d_i) | N ≤ 2B: R = max{ΣT, D+W+T_max} exact; per-client x_i = φ_i/R, floor = min φ_i/R | valid |
| 5 | different partitions with the same total affine cost (cap 16 → 16+8; cap 12 → 12+12) | identical R (only K enters) | valid; this is algebra, not novelty |
| 6 | positive hold W and idle intervals | the D + W + T_max term; plateau for B < N < N_c; N = 16…20: R = 732 ms constant at (a,d,D,W)=(52,30,50,150) | the C4 formula without D, W is **incomplete**, not wrong in the server-limited limit |
| 7 | locally increasing fixed-policy rate | launch-when-full-or-timeout: R(16) = 582 < R(15) = 702; threshold-k: x = 0 for N < k; hold-head: R(N) non-decreasing for N = 1…40 | "x_π(N) is non-increasing" is **false in general**; true for hold-head in the tested domain |
| 8 | any nonmonotone *existential* capacity | none constructible; Lemma 1 proves monotonicity under idling | claim withdrawn |
| 9 | C4's N = 28 cohort "14+14" | observed in-window sizes {12: 131, 16: 131} (= greedy 16+12); the C4 code used [14,14] only to set K = 2, so its prediction was unaffected, the **text was wrong** | audit finding |

## 4. Smallest defensible theorem, and the novelty verdict

**Statement:** Theorem A (§2). **Corollary 1 (achieved capacity).** For N ≤ 2B, N\*_π(x) = max{N ≤ 2B : R(N) ≤ φ_min/x}; N ↦ R(N) is flat on [B, N_c) and strictly increasing afterwards, so N\*_π(x) has a **flat segment**: for every x ≤ φ/(D+W+T_B) the supported concurrency is at least ⌊N_c⌋ (no loss for the first ⌊N_c⌋ − B clients beyond the cap).
**Example the theorem predicts and a naive batch-cap model misses.** C1 GPU traces (cap 16, B < N ≤ 2B, W = 150 ms, D = 104.5 ms measured): the naive model R = ⌈N/B⌉a + Nd (no D, W; with the MEASURED batch times) predicts R(16) ≈ 534 ms (measured 784, −32 %) and a 51 % rise in R from N=16 to 24 (534 → 804); Theorem A with **measured** batch times gives R(16) = 788 ms (+0.5 %) and R(24) = 804 ms (−0.8 %), i.e. a 2 % rise, matching the measured 784 → 811 (+3.4 %) and the observed floor 3.425 → 3.283 (−4 %). `results/c5a/heldout_check.json`.
**Out-of-domain counterexample:** N = 40, B = 16, D heterogeneous (§3 #3).

**Why this does not survive the gate**
1. *Standardness.* The proof is a two-loop max-plus cycle mean plus FCFS order preservation. The cohort partition identity and ⌈·⌉ are not contributions (task rule), and the "plateau + step" structure is the elementary consequence of having a client-side delay loop in series with a server loop. The distinctive element — the vacuity of selection for N ≤ 2B in steady state — is the same two-cohort observation.
2. *Narrow validity.* Everything exact lives in K ≤ 2, deterministic, identical-D, fixed-rule systems. The first genuinely non-trivial regime (N > 2B) has no single cycle and no clean law; I did not find a short exact statement.
3. *Not testable where it would be non-trivial.* All valid GPU runs have N ≤ 2B (§5).
4. *Held-out accuracy comes from the accounting, not the model.* With the fitted affine cost the held-out goodput error is −1.7 / −3.5 / −6.0 % at N = 24 / 28 / 32 (constants from N ≤ 16: a = 11.5 ms vs the 52 ms of the all-data fit; the intercept is unidentified); with measured batch times the structure is within 0.0–0.8 %. The remaining error is the unexplained batch-time model error (loaded vs isolated), not the theorem.
5. *Literature position.* Open-arrival LLM batching queues (arXiv:2407.05347, Inoue, Ramani–Tantawi, Li–Dai–Peng), closed batch networks with exponential times and mean-field limits (Kar et al.) and SLO-capacity by simulation + binary search (DistServe) all differ in model, but none of them makes the deterministic lock-step cycle result *surprising*.

## 5. Which existing GPU traces can test what (no new runs)

| run | catalog | cap | γ | seed | protocol | in-window batch sizes | usable for |
|---|---|---|---|---|---|---|---|
| C1.1 pilot N = 1…16 | **GSM8K-test** | 32 (never binds) | 2 | 1 | original gate | N (single cohort) | within-N≤B structure on a **different catalog**; **no N ≥ 24** — it is not held-out for N ≥ 24 |
| C1.2 ext N = 8, 12, 16 | GSM8K-train | 32 (never binds) | 2 | 1 | original gate; N=24 OOM, no data | N | replicate of cap-16 N = 8, 16 (cross-launch noise 0.2–1.2 %) |
| C1.3 cap-16 N = 8, 16, 24 | GSM8K-train | 16 | 2 | 1 | original gate; **N = 24 failed it (3.07)** | {8}, {16}, {8:148, 16:149} | N ≤ 2B structure |
| C1.4 probe N = 28, 32 | GSM8K-train | 16 | 2 | 1 | exploratory_v2; **both fail the original gate** | {12:131, 16:131}, {16:236} | N ≤ 2B (K = 2) |
Same runtime H, run-seed 1, population seed 20261020 throughout. Consequences: (i) the only "held-out" points are N = 24, 28, 32 of the **same catalog and seed** as the fits (held out in N only; inputs ctx/φ/D measured at the target); (ii) no trace has N > 2B; (iii) no trace with a second catalog at N ≥ 24; (iv) no independent seed. A theorem about N > 2B or about asynchrony cannot be tested with the existing data.

## 6. Mathematical status of the older claims

| claim | status |
|---|---|
| C3: N ≤ 2B ⇒ selection vacuous | valid for steady cohort rotation; false in transients (§3 #1) |
| C4: saturated law φ/(⌈N/B⌉a + Nd) | exact only for K ≤ 2 and min(T₁,T₂) ≥ D+W; incomplete otherwise; wrong for N > 2B |
| C4: "14+14 at N=28" | wrong; observed and predicted partition is 16+12 |
| Lemma 1 (existential downward closure) | **valid** with idling and batch time monotone in membership |
| "feasibility may be nonmonotone" (C4 §5/F8) | valid only for **policy-achieved** floors (§3 #7) and for finite-window estimates (noise); not for existential capacity |

## 7. Decision and what to do instead

**STOP** (not PROCEED, not REVISE): the candidate theorem is standard, its exact domain is narrow, and the non-trivial regime is neither solved nor testable. A REVISE would ask for a theorem on N > 2B with unequal service frequencies; I could not state one, it would concern a deterministic-lock-step artifact of identical virtual clients, and the effect that matters there (scheduling) was measured at 3–13 %.
**Recommended repositioning:** a **methodological case study** — *what it takes to measure interactivity–concurrency capacity of speculative serving correctly*: the hybrid causal virtual-time emulator and its invariants, the measurement-validity audit (C0), the classifier's operating characteristics, the cohort-accounting explanation of the RQ1 curve (Theorem A as an explanatory lemma, with measured batch times), and the negative results (controller components inert; lock-step; N ≤ 2B vacuity with its transient exception). Novelty of this positioning is **unverified** and must be checked against the literature on benchmarking methodology and capacity testing before any commitment; it also still needs certified multi-seed measurement that the current budget cannot pay (≈ 10–14 node-h per certified x-value).
**Not recommended:** a full theorem project on finite-cap capacity; any GPU work; further controller development.

## 8. Reproduction and tests
`python scripts/c5a/des_claims.py out.json` (832 DES combinations), `scripts/c5a/heldout_check.py`, `scripts/c5a/trace_comparability.py`; `pytest legacy/tests/test_c5a_counterexamples.py` (19 tests). The mini DES is deliberately independent of the project simulator and was **not** altered to fit any equation; every formula was tested as stated, including the ones that failed.

## 9. Test status
Local full suite (Windows): 503 passed; the 4 known Windows-environment failures and 2 torch collection errors are unchanged. C5-A changed **no file under `legacy/src`** since C4 (`e221070`), whose Setonix per-module run (596 passed, 1 inherited failure) therefore still applies to the source; the Setonix job was not repeated for the added test file.
