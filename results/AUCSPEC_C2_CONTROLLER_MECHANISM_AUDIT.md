# AUCspec C2 — Capacity-aware controller: mechanism audit

Base `research/c1-scientific-correctness` @ `fa3fe1f`; branch `research/c2-controller-mechanism`. CPU only: no GPU job, no verifier calibration,
no server-price activation in the canonical campaign, no backend / V1-greedy change, no historical RQ1 data touched, no merge to main.

Evidence labels used throughout:
**[V]** implemented and verified (test or measurement in this branch) · **[U]** implemented but unvalidated · **[P]** proposed, not implemented · **[X]** unsupported paper claim.
All CPU closed-loop numbers come from `legacy/tests/closed_loop_harness.py`: the *real* simulator/controller/queue/batcher code with a CPU stand-in verifier
(`wall = 52 ms + 0.19 ms x processed full-context tokens`, fitted to C1.3 traces) and Bernoulli acceptance coupled across policies. It is a mechanism probe,
not a performance prediction; one run-seed per cell; settings were fixed before looking at outcomes.

## 1. Phase A — what the RQ1 evidence can and cannot support (`results/c2/rq1_evidence_inventory.json`, test-pinned)

| campaign | N | x_min | verifier busy | real/virtual | original gate | exploratory_v2 | use |
|---|---|---|---|---|---|---|---|
| C1.1 pilot (GSM8K-test, cap32) | 1–16 | declining | – | ≤3 | pass | – | low-load anatomy panel |
| C1.2 extension (train, cap32) | 8,12,16 | – | – | ≤3 | pass | not evaluated | N=24 aborted by HIP OOM, no data |
| C1.3 cap16 (train) | 8, 16 | 3.43 @16 | 0.535, 0.684 | ≤3 | pass | pass | descriptive |
| C1.3 cap16 | **24** | – | 0.992 | **3.07** | **FAIL** | pass | exploratory_v2 only, flagged |
| C1.4 probe | 28, 32 | – | 1.0 | 3.21, 3.28 | **FAIL** | pass | exploratory_v2 only, flagged |

Every point is one run-seed and one population per catalog. Supports: a *descriptive* RQ1 figure (cap16 train curve with N≥24 flagged; pilot as a separate
panel). Does not support: any CI/significance, a certified N*(x) or ICA, a controller-vs-baseline claim, causal explanation of the N=8 anomaly, or N≥24 under the
original protocol. All old verdicts (incl. the original N=24 failure) are preserved verbatim.

## 2. Phase B — Capacity-DPP mechanism audit

Score: `S_i(γ) = (V+Z_i)·φ̂_i(γ) − Q_s·θ_f·(γ+1) − Q_i·γ·(τ_d+κ/r)`, slots of 100 ms.

| state | declared unit | update | what the logs show (real C1.3/C1.4 traces) |
|---|---|---|---|
| Z_i | tokens (deficit vs x_req) | Lindley per slot | ≈0 until saturation (mean 0.05→0.7 for N=8→24), 8–33 for N=28–32. Meaningful only when floor is violated **[V]** |
| Q_s | ms, "verifier backlog" | per slot: Q←[Q+Σ θ_f·cost − slot]+ | R²=0.998 vs *last batch wall* at N=8, 0.89 at N=16, **0.0** at N≥24. It is a lumpy "last-batch-wall − slot" signal, not backlog; mean 190→1030 ms. **[V]** |
| Q_i | ms of device work | per slot | positive in **0 %** of homogeneous-trace rounds; in heterogeneous harness runs read ≈5–10 slots stale (age 5.6–10 slots) **[V]** |
| θ_f | ms per verifier position | config (0 in production) | the *canonical* run uses θ_f=0, so the Q_s term is **identically zero** **[V]** |

Unit problem: Z is in tokens, Q_s·θ_f·(γ+1) in ms²-ish (queue ms × ms/position × positions). The three terms are unnormalised; a 10× price-scale change flips
>20 % of decisions when the price is active (`qs_x0.1`), so the "equilibrium" depends on arbitrary scaling **[V]**. V=50 vs queues of 10²–10³ means any active
price is either inert (θ_f=0) or dominant (saturates at γ=0) — there is no regime where it trades off gracefully.

### 2.1 Does each mechanism change decisions? (open-loop on identical logged state, then closed loop)

Production config (θ_f=0, `capacity_knapsack`, cap16), N=16/24/32:

| population | γ distribution | no_Z flips | no_Q_i flips | no_Q_s flips | closed-loop vs constant γ=4 |
|---|---|---|---|---|---|
| homogeneous | 100 % γ=4 | 0 % | 0 % | 0 % | **identical trajectory** (x_min 4.87 / 4.69 / 3.57 equal to the digit) |
| heterogeneous α,HW | 70–76 % γ=4, 24–30 % γ=0 | 0–8 % | 25–30 % | **0 %** | x_min 2.79/1.89/1.51 vs γ=4 3.02/2.68/1.98 and γ=2 2.53/2.27/1.73 |

* **Z_i**: the only term that is exactly consistent with its definition; rarely active (≤8 % flips, only at N≥24 hetero). **Keep; verified as correct, weakly active.**
* **Q_s**: inactive in production (0 % flips even though mean Q_s = 345–600 ms). Positive state ≠ functioning mechanism **[V]**. With a hypothetical θ_f>0 (offline
  E2, not canonical): 94–99.9 % of batches are forced to a single request and x_min drops to 0.38–0.64 (θ=1…20) versus 1.5–1.9 unpriced. The implemented price
  *destroys* batching: `capacity_dpp_value<0` for every request, so `pop_batch` falls back to a forced one-request batch. Not a usable control as coded **[V]**.
* **Q_i**: changes 25–30 % of actions in heterogeneous runs but only through a stale snapshot (applying the missing idle slots removes the effect — `stale_drained` = `no_Q_i`),
  and it *lowers* x_min below constant γ=2 and γ=4 at N≥24. The term acts as stale noise, not as a device-load signal **[V]**.
* **Knapsack vs FCFS**: member sets are identical in **100 %** of multi-request batches, for N=24 and 32, x_req 3.0 and 4.5, homogeneous or heterogeneous hardware. The cap
  binds in 0.3 % of batches; the logical token budget in 0 %. Reason (structural): `pop_batch` only admits requests with arrival ≤ head+W, and cohorts phase-lock, so the
  eligible set is almost always ≤ cap. Reproduced even with cap=4 at 100 % utilisation (test `selection_is_structurally_inert…`). The mechanism is **inert**, not merely
  "not yet beneficial" **[V]**.
* **Wireless allocator**: for greedy 96 B payloads the maximum effect of any allocator on the upload is ≤2.1 ms (0.25 % of a 800 ms round) — irrelevant; the
  bandwidth at which uploads reach 10 % of a round is 0.0096 Mbit/s. For full-distribution uploads (1.2 MB) the allocator changes upload time by 2–13 s
  (square-root 77 ms vs proportional 1480 ms vs queue-weighted 3804 ms). A comm-limited regime exists only for full-distribution payloads, not for the V1-greedy
  protocol used in all RQ1 campaigns **[V]**.
* **Virtual-future dependence**: in causal mode decision logs are identical across three real-delay seeds (1200/1200 decisions, 0 mismatches); the historical real-order
  mode yields different decision *sets* (1200 vs 1416) — decisions there depend on real completion order **[V]**.

### 2.2 Why production Capacity-DPP degenerates

With θ_f=0 and Q_i≈0 the score reduces to `(V+Z)·φ̂(γ)`, which is increasing in γ, so the argmax is `γ_max` for every client. The "adaptive" controller is constant-γ=4
in homogeneous populations (bit-identical trajectory) and an α-dependent γ∈{0,4} switch (via the stale Q_i artefact) in heterogeneous ones. It cannot have a
"two-regime" behaviour, because the compute price that would reduce γ in the compute-bound regime is switched off **[V]**.

## 3. Phase C — offline physical verifier-cost model (`results/c2/verifier_cost_model_offline.json`)

4,250 batches from C1.3/C1.4 loaded runs (fixed γ=2). **Not an independent hardware calibration**; runs are loaded and share one γ, no depth variation.

| model | LORO MAE | MAPE | PI90 coverage |
|---|---|---|---|
| wall ~ 1 + B | 61.8 ms | 17.9 % | 0.89 |
| θ_f·(γ+1) logical positions (+ intercept) | 62.9 ms | 18.5 % | 0.89 |
| **full-context processed tokens** | **45.9 ms** | **12.5 %** | 0.89 |
| ctx_sum + B + γ_sum | 46.3 ms | 13.3 % | 0.89 |

* `wall ≈ 52 ms + 0.19 ms × processed tokens` (R² 0.74). The logical-position model is no better than the batch-size-only model: **θ_f·(γ+1) is not the physical cost**.
* Marginal cost of one extra request is context-dependent: 12→92 ms for ctx 100→450 at γ=0, while the θ_f abstraction gives a context-independent 10–50 ms.
* The γ coefficient's run-block bootstrap CI straddles zero: speculative depth is **not identified** (γ was constant). The compute price for γ cannot be set from these data.
* OOD: 90 % prediction intervals cover only 30–56 % under batch-size extrapolation (train B≤8 → test B≥12). Loaded-run, bounded-range model; do not activate **[U]**.
* No look-ahead leakage: predictors are request-level quantities known at decision time (context, γ, B); the fit is leave-one-run-out.

## 4. Phase D — D3 audit and repair (shared-bandwidth / active-client causality)

Finding: in causal virtual time, `_allocate_shared_bandwidth` read other clients' queue/demand state and the "active client" set at the *requester's* virtual time, but those
clients could be arbitrarily ahead in virtual time (D3). Repair (minimal, provable): causal mode accepts only allocators whose share of client *i* is a function of
*static* information (`equal`, `proportional`; `CAUSAL_SAFE_ALLOCATORS`), raises `ValueError(D3…)` otherwise at construction; `_active_clients()` returns the static client set;
other clients' demand is treated as 0. Queue-weighted / square-root allocation require a causal resource-information snapshot (design: broadcast the allocation at slot
boundaries from the same watermark snapshot the Q_s/Q_i coordinator already publishes; cycle analysis shows it needs the verifier horizon, so it is left **[P]**). Frozen verifier H untouched.

Also found and fixed (HIGH, liveness): the batcher's causal loop's deadlock check peeked the head of the asyncio queue the batcher was blocked on, "stealing" an item and
hanging N=24 cap16 FCFS closed-loop runs. Submissions now go to an internal list; `_absorb_submitted` is pure bookkeeping; waiting uses `coordinator._wait_until`. Regression test:
`test_liveness_regression_n24_cap16_fcfs_completes`. Determinism under permuted asyncio order verified (E5, 0 mismatches). 20 causal-wiring tests pass.

## 5. Claim–implementation–evidence matrix

| paper claim | status | evidence |
|---|---|---|
| Deficit queue Z_i enforces floor x_req | **[V]**, weak | correct Lindley; ≤8 % flips |
| Per-client speculation index γ_i adapts to load | **[X]** | γ=γ_max (homog) / stale-Q_i switch (hetero) |
| Server queue Q_s prices verifier capacity | **[X]** | inactive at θ_f=0; collapse of batching if θ_f>0; Q_s≠backlog |
| Device queue Q_i prices drafting load | **[U]** | acts only via stale reads; hurts x_min |
| θ_f·(γ+1) is verifier cost | **[X]** | positions model no better than B-only; full-context dominates |
| Knapsack batching beats FCFS | **[X]** | identical member sets in 100 % of batches; cap rarely binds |
| Radio waterfilling reduces interactivity loss | **[X]** for greedy-SD protocol, **[U]** for full-distribution | ≤2 ms effect vs seconds |
| Two-regime γ*(B) (Theorem 1) in the system | **[U]** | needs active compute price + depth-identified cost |
| Causal virtual-time decisions independent of real order | **[V]** | E5, D3 repair + tests |
| Drift-plus-penalty guarantee O(V) backlog for the implemented controller | **[X]** | units mismatch; Q_s not a queue of the modelled kind |
| Censored UCB learning | **[P]** | not exercised in C1 (oracle α) |
| Closed-form optimal index (13) | **[P]** | not the implemented score |

## 6. Defect list (priority)

1. **HIGH — fixed.** Batcher deadlock-check stole queue items (liveness; N=24 cap16 FCFS hang).
2. **HIGH — fixed (restriction).** D3: causal-mode allocator/active-client read of future state.
3. **HIGH — open.** Production Capacity-DPP ≡ constant γ_max (θ_f=0): the headline controller has no capacity-aware behaviour.
4. **HIGH — open.** Unit mismatch between Z (tokens) and Q_s,Q_i (ms²-like): arbitrary scale, no stable activation point; ms-unit activation collapses batching.
5. **MED — open.** Q_s is "last-batch wall − slot", not backlog; R²≈0 vs wall at saturation.
6. **MED — open.** Q_i read stale by 5–10 slots; effect disappears if idle slots applied.
7. **MED — open.** Knapsack inert (head+W eligibility + phase-locked cohorts).
8. **MED — open.** θ_f abstraction ignores full-context cost; γ cost unidentified.
9. **LOW.** Wireless allocator effect only relevant for full-distribution uploads.

## 7. Candidate corrected equations (proposed, **not implemented**)

* Verifier cost per request (ms): `ĉ_i(γ) = a_b + b·(ctx_i + γ + 1)`, with `b≈0.19 ms/token`, `a_b=52 ms/B_ref` as *placeholders to be re-identified*; the batch time `T̂ = a + b Σ_i (ctx_i + γ_i + 1)`.
* Normalised price: replace `Q_s·θ_f·(γ+1)` by `λ·ĉ_i(γ)/T_ref` with `λ` dimensionless (units tokens per unit cost-fraction) and `Q_s` redefined as *demand-based* backlog `[Q_s + Σ_i ĉ_i − slot]+` with time-aware drain.
* Device price: `μ_i·γ·(τ_d+κ/r)/T_slot`, `Q_i` read from the same causal snapshot the decision time uses (age ≤ 1 slot).
* Score unit check: `(V+Z_i)φ̂_i(γ)` [tokens] − `λ ĉ_i/T_ref` [tokens] − `μ_i γ(τ_d+κ/r)/T_slot` [tokens].

## 8. Recommendation per mechanism for the SIGMETRICS paper

| mechanism | decision |
|---|---|
| Causal virtual-time coordinator, exact occupancy, D3 guard | **keep** (methodology contribution; verified) |
| Z_i deficit queue | **keep**, state as floor-enforcement only |
| Q_s verifier price | **correct before any claim** (demand-based, normalised, context-aware); until then do not claim |
| Q_i device price | **correct or remove** (stale read; harmful) |
| Capacity-knapsack batching | **relegate to limitations** unless a work-conserving batching design (see RQ3/RQ4 doc) shows non-zero member-set change |
| Wireless allocation | **relegate to limitations** for greedy SD; keep only as full-distribution upload extension |
| Theorem 1/3 as properties of the implemented controller | **remove claim**; present as properties of the stylised model |
| "Adaptive γ" | no claim until θ_f/Q_s corrected and γ cost identified |

## 9. Test inventory (this branch)

`legacy/tests/test_c2_mechanism_activation.py` (13), `test_c2_cost_model.py` (5), `test_simulator_causal_wiring.py` (+D3, 20 total pass). Full local suite: only the 4 known Windows-environment failures.
