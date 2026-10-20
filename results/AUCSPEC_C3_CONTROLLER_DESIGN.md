# AUCspec C3 — Minimal capacity-aware controller: scientific contract

Branch `research/c3-minimal-controller` (base `research/c2-controller-mechanism` @ `7e802a1`). CPU only: no GPU job, no calibration run,
no price activation in the canonical policy, no frontier search, no historical-data rewrite, no merge to main.
Companions: `AUCSPEC_C3_CPU_MECHANISM_VALIDATION.md` (evidence), `AUCSPEC_C3_MINIMAL_RQ3_GPU_PLAN.md` (GO/NO-GO).

Status labels used throughout — **[I]** implemented, covered by a test; **[U]** implemented, not validated against real hardware;
**[P]** specified here, not implemented in the serving path; **[A]** assumed parameter (not measured).

The C3 controller is **opt-in** (`controller.policy: c3_depth`, `verification_batching.scheduler: c3_deficit | c3_fcfs`,
`verification_batching.work_conserving: true`). `capacity_dpp`, the legacy schedulers, the frozen runtime H, V1-greedy verification,
`batch_wall` accounting, the causal coordinator and all historical results are unchanged.

---------------------------------------------------------------------------------------------------------------------------------

## 1. Notation and units

Times are virtual milliseconds unless stated; "tokens" are committed (useful) output tokens or processed (context + candidate) tokens as marked.

| symbol | meaning | unit |
|---|---|---|
| i, r=(i,k) | client, its k-th verification request | – |
| γ_r ∈ {0..γ_max} | speculative depth of request r | candidate tokens |
| ctx_r | context length verified statelessly (prompt + committed history) | processed tokens |
| p_r = ctx_r + γ_r | processed tokens of r (what the engine computes: full context + candidates) | processed tokens |
| t_r | decision time (client virtual clock when γ_r is chosen) | ms |
| a_r | arrival time at the server = t_r + γ_r·c_i + RTT_i/2 + u_r  (u_r ≥ 0 uplink/jitter) | ms |
| c_i = τ_d,i + κ/r_i | device time per drafted token (draft + uplink) | ms / drafted token |
| W | batch-formation hold (legacy `batch_wait_ms`) | ms |
| B_max, M | batch-size cap, processed-token budget (memory) | requests, processed tokens |
| s_k, F_k | start / finish of verifier batch k (single shared verifier ⇒ s_{k+1} ≥ F_k) | ms |
| T_k = F_k − s_k | physical (measured, `batch_wall`) service time | ms |
| φ_r(γ) = Σ_{j=0}^{γ} Π_{l≤j} α_l  (= (1−α^{γ+1})/(1−α)) | expected useful tokens per call (incl. bonus), oracle/offline-profiled α | tokens / call |
| x_i | delivered rate of client i | tokens / s |
| x_req | SLO floor | tokens / s |
| V | throughput weight (canonical value 50) | tokens |

## 2. Event model, timestamps and causal reads

Events: *arrival* (a_r), *batch start* (s_k), *batch finish* (F_k), *decision* (t_r). The simulator is conservative-PDES:

* every idle client carries a lower bound lb_i on its next arrival (= t + RTT_i/2); a batch with eligibility cutoff C is formed only when every
  idle client has lb_i > C (no arrival ≤ C can still appear) — invariant **D1**, checked by the event ledger;
* a decision at virtual time t waits until the coordinator horizon H > t, where H = start of the in-flight batch, or, if none, max(F_prev, min queued
  arrival + W). Hence when a decision at t proceeds, **every batch with start ≤ t has been formed, verified and published** (and no unformed batch can
  start ≤ t) — invariant **D2**. Reads are therefore pure functions of the trace restricted to events with timestamp ≤ t; the order in which asyncio resumes
  concurrent clients, and the real wall-clock completion order, cannot change any decision. [I] (`test_decisions_and_batch_membership_are_identical…`,
  `test_every_depth_decision_saw_exactly_the_batches_started_and_finished_by_its_virtual_time`).

What a decision at t may read (nothing else):
* its own client state (Z_i, own completion history, α̂_i, device parameters);
* the **published verifier record** of every batch with *start* ≤ t: start s_k, predicted service T̂_k (from formation-time features), the work of the
  requests that had arrived by s_k and were left out, and — only for batches with *finish* ≤ t — the realized useful tokens and wall time.
It may **not** read: the realized finish of a batch still in service at t, arrivals later than t, other clients' state, the position of any client in
real-time order. (D3 for the wireless allocator: the minimal repair in C2 is kept — `equal`/`proportional` only in causal mode.)

## 3. State: physical queues, virtual queues, multipliers

| object | kind | definition | unit | update |
|---|---|---|---|---|
| W_srv(t) | **physical queue** | set of requests with a_r ≤ t not yet in a started batch | requests | arrival adds, batch start removes |
| in-service batch | physical | batch k with s_k ≤ t < F_k | – | – |
| U(t) | physical (post-hoc, evaluation only) | Σ_{r∈W_srv(t)} d_r + (F_k − t)^+  (waiting predicted demand + in-service remaining) | ms | eq. (S2) |
| Û(t) | **causal estimate** of U(t) | eq. (S3) | ms | recomputed from the published trace |
| Z_i | **virtual queue** | Lindley deficit of client i against x_req | tokens | eq. (Z1) |
| ω_i = (V+Z_i)/V | weight (function of Z_i) | dimensionless | – |
| μ(t) | **Lagrange-type multiplier** (shadow price of verifier time) | deficit-weighted tokens per ms | – | eq. (G), not a queue |
| V, T_ref, max_skips | design constants | tokens, ms, batches | – | fixed |

Only Z_i is a virtual queue that is integrated over time. There is **no** server virtual queue Q_s and no device virtual queue Q_i in C3 (C2 showed
they were a last-batch-wall signal and a stale read respectively). The server congestion state is the workload estimate Û, defined from demand (not from
the cost of the last completed batch).

## 4. Physical service time and its online estimate

Preliminary physical model (C1.3/C1.4 loaded runs, leave-one-run-out MAE 46 ms, 12.5 % MAPE — **not an independent calibration**):

    T̂(S) = a + Σ_{r∈S} d_r ,      d_r = c_req + b·ctx_r + b_γ·γ_r        [ms]                       (S1)

| parameter | value used | status |
|---|---|---|
| a (fixed batch overhead) | 52 ms | [A] preliminary, loaded-run fit |
| b (per context token) | 0.19 ms/token | [A] preliminary, loaded-run fit (bootstrap 90 % CI 0.183–0.197) |
| c_req (per-request fixed) | 0 | [A] confounded with b_γ·γ in the C1 traces |
| b_γ (per candidate token) | = b (0.19) | **[A] not identified**: γ ≈ 2 in every C1 batch; the M3 estimate 2.1 ms/token has 90 % CI [−1.7, 5.5] |

**Online estimate [P/I-component]** — `OnlineCostEstimator`: ridge-RLS on (1, Σctx, B, Σγ) with the assumed model as prior. Prediction uses only
formation-time features; the measurement updates it after completion. The estimator *reports* what the data cannot say: while γ is constant, only
c_req + γ·b_γ is learned and each coefficient keeps most of its prior spread; once γ varies across batches the split becomes identifiable (test
`test_online_estimator_learns_context_cost_and_reports_the_unidentified_gamma_coefficient`). It is not wired into the controller; every CPU result below
uses the static assumed model and tests sensitivity to its error.

Consequence for the scientific claim: any statement about γ-dependence of verifier cost needs a measurement with **varied γ at fixed membership**
(see the GPU plan). Nothing fitted on the C1 traces is used as validation of anything.

## 5. Server demand accounting and snapshots

Per request r the predicted demand d_r (S1) enters the demand ledger **once, at arrival**, and leaves **once, when its batch finishes**. The batch
overhead a is charged **to the batch** (its service interval), never to a request, so nothing is counted on both submission and completion.

    A(t) = Σ_{a_r ≤ t} d_r ,   D(t) = Σ_{F_batch(r) ≤ t} d_r ,   A(t) ≥ D(t),   A(∞) = D(∞)                           (S2a)
    U(t) = Σ_{r waiting at t} d_r + (F_k − t)^+                                                              (S2)

(`WorkLedger`; conservation, unit-rate drain and no double counting are tested on a deterministic trace, §10.)

The causal estimate used by a decision at t is built from the batch k* = the last batch with **start** s_{k*} ≤ t:

    Û(t) = (s_{k*} + T̂_{k*} − t)^+  +  Σ_{r∈L_{k*}} d_r + a·1[L_{k*}≠∅]                                    (S3)
           (predicted remaining service)   (work certainly waiting: arrived ≤ s_{k*}, not in the batch)

with L_k = {r : a_r ≤ s_k, r ∉ batch k} and T̂_{k*} the *prediction at formation*, never the realized finish. Û(t) ≤ U(t) when predictions are exact
(it ignores arrivals in (s_{k*}, t], which can only add work). The bound is an argument, not a tested property of the traces; with a mis-predicted T̂ the remaining-service term can be
wrong in either direction, and the effect of a wrong b_γ on decisions is what the misspecification sweep in the validation doc measures.
Weighted goodput of the finished batches (realized, finish ≤ t):

    G(t) = (1/H_w) Σ_{k: F_k ∈ (F_{k°}−H_w, F_{k°}]} Σ_{j∈batch k} (w_j/V)·u_j       [weighted tokens / ms]    (G)

(w_j = V + Z_j stamped on the request, u_j realized useful tokens, k° the last batch with F ≤ t, H_w = 5 s window, min(H_w, F_{k°}) at start-up.)

**Idle slots and reconstruction at any decision time.** Between publications nothing changes: Û(t) is recomputed from the last started batch (it drains at
unit rate through the predicted-service term and reaches the waiting-work floor), G(t) keeps the last finished batch, and Z_i is reconstructed lazily at
the client's next completion by applying every missed slot (+x_req·Δ per idle slot in (Z1)) — identical to a slot-by-slot update. A time slot with no
arrival therefore needs no event and costs nothing; the state at any t is the same function of the trace whatever the number of intermediate reads.

## 6. Per-client deficit

    Z_i(s+1) = [ Z_i(s) + x_req·Δ − u_i(s) ]^+ ,   Δ = 0.1 s slot, u_i(s) = tokens delivered in slot s            [tokens]   (Z1)
    Z_i^eff(s) = Z_i(τ_i) + x_req·(s − τ_i)/1000 ,   τ_i = client's last completion (stamp time)              [tokens]   (D2)

(Z1) is the existing slot Lindley recursion (unchanged). (D2) is the exact value of the same recursion at the batch start s for a request that has been
waiting since τ_i: the client has received no token since its last completion, so its deficit grows linearly; both operands are known at s.

## 7. Mechanism A — deficit-aware, work-conserving verifier scheduling [I][U]

**Formation (work conserving).** A batch starts at s = max(F_prev, a_head + W) and its eligible set is **everything that arrived by s**:
E(s) = {r waiting : a_r ≤ s}. The legacy rule closes the eligible set at a_head + W even when the verifier is busy until later, so requests arriving
between a_head+W and F_prev are held back and batches fragment under asynchronous arrivals (§ validation). The safety argument: the coordinator horizon
is already a lower bound on s, so a decision at t < horizon cannot be affected by a batch that has not started by t, and D1 holds with cutoff = s.

**Selection.** Greedy fill in rank order under the physical constraints (strict, never relaxed by the controller):

    |S| ≤ B_max ,   Σ_{r∈S} p_r ≤ M (processed tokens, memory) ,   T̂(S) ≤ T_max (optional, default ∞)                  (C)

* rank "density" (max-weight form): ρ_r(s) = (V + Z_i^eff(s))·φ_r / (a/B_max + d_r)           [weighted tokens / ms]       (D3a)
* rank "deficit" (floor-oriented): lexicographic (Z_i^eff(s), ρ_r(s))                                                    (D3b)
* starvation guard: a request left out of `max_skips` consecutive batches is ranked first.
* the first ranked request is always admitted (liveness; an oversize singleton is flagged, never silently dropped); the selection is never empty when E(s)≠∅
  (no forced-service escape hatch, no negative-value batches).

Fair baseline `c3_fcfs` = the same eligibility and the same constraints (C), arrival-order ranking. **The comparison of C3 scheduling against the legacy
FCFS therefore separates three effects: work-conserving formation (legacy → c3_fcfs), the physical constraints (included in both), and deficit-aware
ranking (c3_fcfs → c3_deficit).**

**Why the cap rarely binds (structural).** In a closed system with N decode streams the batch that just finished is in transit (its members re-arrive at
≥ F + RTT/2 + draft time > s when s = F), so |E(s)| ≤ N − |previous batch|. With a full previous batch, |E| ≤ N − B_max; selection has something to choose only
if N > 2·B_max (or the previous batch was partial). For the measured memory limit B_max = 16 that means N > 32; with a tighter limit B_max = 8, N > 16.

**Starvation bound.** A request waiting in E(s) is skipped at most max_skips times before it is ranked first; if several requests are simultaneously
forced they are served in arrival order, so a request is selected after at most max_skips + ⌈(N−1)/B_max⌉ batches. [I] (tested on synthetic requests; the
bound holds in all 3,400+ CPU runs.)

## 8. Mechanism B — context-aware speculative-depth control [I][U]

Per client, at its decision time t (all quantities causal):

    S_i(γ) = ω_i·φ_i(γ) − [ μ(t)·b_γ + ω_i·x̂_i·c_i ]·γ ,    γ_i = argmax_{γ∈{0..γ_max}} S_i(γ) (ties → smaller γ)      [weighted tokens]   (B1)
    μ(t)  = g(t)·G(t) ,   g(t) = Û(t)/(Û(t)+T_ref)  ∈ [0,1)                                                   [weighted tokens / ms]   (B2)

x̂_i = own delivered rate over the last 8 completions (tokens/ms; prior = x_req). Derivation: (B1) is the first-order condition of the weighted
renewal-reward objective Σ_j ω_j φ_j / T_cycle with respect to γ_i, with T_cycle = a + Σ_j d_j (saturated verifier) and own cycle time lengthened by the
drafting time γ_i c_i. An extra drafted token (i) lengthens the batch by b_γ for **everyone** — externality μ·b_γ with μ = weighted goodput per unit
verifier time — and (ii) lengthens client i's own round by c_i, a loss ω_i x̂_i c_i; its gain is ω_i·Δφ. With prefix acceptance Δφ(γ→γ+1) = α_i^{γ+1}, so

    γ_i* = max{ γ ≤ γ_max : α_i^γ > θ_i } ,    θ_i = μ·b_γ/ω_i + x̂_i·c_i      (dimensionless: tokens per drafted token)         (B3)

**Context enters only through the level of the cost, not through its γ-slope.** Under the affine model (S1) the verifier-time per useful token is
(a/B + c_req + b·ctx + b_γ·γ)/φ(γ); a larger ctx raises the numerator for every γ, which makes deeper speculation *more* attractive, not less (the cost
minimising γ is non-decreasing in ctx). A context-aware per-call cost therefore does **not** by itself yield a useful depth decision; depth is decided by
(α_i, b_γ, x̂_i c_i) via (B3). (`results/c3/verifier_economics.json`.)

**When (B3) can differ from γ_max.** γ_i* < γ_max iff α_i^{γ_max} ≤ θ_i. With the device term alone (x̂ ≈ 3 tok/s, c ≈ 22 ms ⇒ θ ≈ 0.07) this needs
α_i ≲ 0.5 for γ_max = 4; the verifier term adds μ·b_γ/ω: with b_γ = 0.19 ms/token it is ≈ 0.01 (inert); it matters only if b_γ ≳ 1–5 ms/token **[A:
unmeasured]**. Hence the controller is an activation-conditional safeguard, not an always-on gain.

## 9. Units and normalisation of every score term

| term | expression | unit | normalisation / invariance |
|---|---|---|---|
| benefit | ω_i φ_i(γ) | weighted tokens | ω dimensionless (V+Z)/V; φ in tokens/call |
| device latency price | ω_i x̂_i c_i γ | (tok/ms)(ms/tok_drafted)(tok_drafted) | invariant to the time unit (tested: ms→s) |
| verifier price | μ b_γ γ | (wtok/ms)(ms/tok_drafted)(tok_drafted) | μ is a measured rate, b_γ a measured/assumed cost: no free scale |
| congestion gate | g = Û/(Û+T_ref) | dimensionless | T_ref = a (one fixed batch overhead); sensitivity ×¼…×16 reported |
| scheduling density | (V+Z^eff)φ/(a/B_max+d_r) | weighted tokens / ms | ranks only; scale-free |
| deficit | Z^eff | tokens | x_req [tok/s]·s |

The deficit weight ω_i multiplies benefit **and** the device term, so it cancels in the own-latency trade-off (tested); it enters the argmax only through the
un-weighted-by-ω verifier price μ b_γ/ω_i — a client with a large deficit tolerates more verifier externality.

## 10. Invariants and proofs on a deterministic trace

Trace: r1 arrives 0 (d=10), r2 arrives 5 (d=20), r3 arrives 30 (d=5); batch 1 = {r1,r2} starts 5, T = 52+30 = 82, finishes 87; batch 2 = {r3} starts 87,
T = 57, finishes 144 (a = 52).

* **Conservation.** A(144) = 35 = D(144); A(86.9) = 35, D(86.9) = 0, D(87) = 30; A(t) ≥ D(t) for all t. Each key appears once in the arrival list and
  once in a batch (the ledger rejects overlapping service, the coordinator ledger rejects double service and unserved requests). ∎
* **Unit-rate drain.** U(5) = 82; U(40) − U(50) = 10 (no new arrival inside the interval ⇒ slope −1); U(144) = 0; U ≥ 0 everywhere.
* **Causal reads.** Adding a batch that starts after t changes neither Û(t) nor G(t) (prefix invariance, tested for t just before/at the start of the
  next batch); changing the realized wall of a batch still in service at t does not change Û(t) (it uses the prediction). On real event traces every depth
  decision saw exactly the batches with start ≤ t and finish ≤ t (tested on every decision of a 40-s heterogeneous asynchronous run).
* **Determinism.** Identical decision sets, batch memberships and per-request outcomes under three different asyncio resume orders (tested).
* **Lower bound** Û ≤ U under exact predictions: the physical remaining service equals the predicted one and arrivals in (s_{k*}, t] only add. (Argument only;
  not asserted on traces.)

## 11. What carries over from the manuscript, and what does not

| manuscript item | status for C3 |
|---|---|
| Prop. 2 (concave frontier, dual sweep) | independent of the controller; not exercised here |
| Theorem 1 (two-regime γ*(B) in the symmetric saturated instance) | (B3) is the corresponding first-order rule **only if** costs are affine and the verifier saturated; the compute-bound regime γ* = 0 needs b_γ large [A: unmeasured]. No claim for the heterogeneous asynchronous implementation |
| Theorem 3 (DPP optimality, backlog O(V)) | **not applicable as stated**: derived for a slotted synchronous model with a per-slot drift-minimising decision; the implementation is asynchronous, per-round, with batch-coupled service. No matching derivation exists. Do not claim the guarantee |
| Theorem 4 (censored learning) | not exercised (oracle α̂); the CPU study shows the absorbing γ = 0 failure of unoptimistic learned depth control |
| Capacity-DPP's Q_s, Q_i, θ_f(γ+1) | **dropped** in C3 (C2: Q_s ≠ backlog; Q_i stale; θ_f abstraction no better than batch size) |

## 12. Parameter inventory

measured-preliminary: a, b (C1 loaded runs). assumed: b_γ, c_req, T_ref, max_skips, H_w, M = 4096, B_max ∈ {8, 16}. fixed from the paper: V = 50, x_req = 3 tok/s,
slot 100 ms. static offline-profiled: α_i (oracle, equal for all arms). None is tuned on the CPU outcomes; sensitivity sweeps are reported.
