# AUCspec C3 — CPU mechanism validation

Branch `research/c3-minimal-controller`. **Every performance number below is model-derived** (real simulator / queue / batcher / controller code; the verifier
is a CPU stand-in with service time `T = 52 + 0.19·ctx + b_γ·γ  (±8 % noise)`, acceptance Bernoulli with draws coupled across arms). It is **not** a GPU
result and certifies no capacity. Raw outputs (3,588 runs, 4.7 MB) are on local scratch; the compact summary is `results/c3/validation_summary.json`
(+ `verifier_economics.json`). Re-run: `scripts/c3/mechanism_validation.py all <raw.json> 8` then `scripts/c3/report_tables.py <raw.json> <summary.json>`.

Design: matched **static** populations — `homog_sync` (identical clients, lock-step), `hetero_sync` (α_i∈[0.6,0.95], hardware, context 80–300 tokens),
`hetero_async` (same + start offsets 0–800 ms + non-negative arrival jitter ≤120 ms, which break permanent cohort lock-step without touching the lower-bound
rule). (N, B_max) ∈ {(24,16),(32,16),(48,16),(24,8),(32,8)}; 5 run-seeds (paired: same acceptance draws across arms); oracle α̂ for every arm (equal
information). Intervals are 95 % t-intervals over **5 run-seeds of one population** — they quantify run-seed noise only, not population variation.

## 1. Stage 0 — platform validation on Setonix (CPU `work` partition, per-module pytest, torch included)

| SHA | modules | passed | failed | note |
|---|---|---|---|---|
| `fa3fe1f` (C2 parent), job 51265347 | 58 | 525 | **1** | `test_payload_interface::test_full_vocab_payload_exposes_dense_distribution_support` |
| `7e802a1` (reviewed C2), job 51265346 | 60 | 547 | **1** | the same test, same assertion |
| `22def69` (C3 prototype), job 51265750 | all | see below | 1 (same) | |

* `test_network_allocator` (6/6) and `test_payload_interface` (2/3) were **run**, as were `test_simulator_causal_wiring` (20, incl. the C2 liveness and D3 tests) and
  the C2 mechanism tests. The only failure is **inherited**: float32 `-0.2` vs float64 `-0.2` in a full-vocab payload comparison, byte-identical on the parent SHA,
  in code C2 did not touch. **No C2 regression.** It is not hidden by an exclusion; it is a pre-existing test-precision defect (suggested separate fix).
* The 3 `test_runtime_guard` and 1 `test_verifier_service_time` failures seen on the Windows laptop pass on Setonix (platform-specific, not logic).
* Final-SHA result for C3 is appended at the end of this file (§13).

## 2. Integrity results over all 3,588 CPU runs

| condition | result |
|---|---|
| causal ledger (D1/D2) violations | **0** |
| invalid / lifecycle-invalid runs | **0** |
| empty batches; forced-service batches | **0 ; 0** |
| batch-size cap violations (B_max ∈ {8,16}) | **0** |
| processed-token budget (4096) violations by c3 / PF arms (singletons excluded) | **0** (the budget binds: max seen = 4096 in the long-context stress) |
| starvation bound `max_skips + ⌈(N−1)/B_max⌉` | **0 violations** (max skips seen 8, bound ≥ 10) |
| determinism under permuted asyncio order (3 resume orders; decisions, memberships, per-request outcomes) | identical |
| decision-time reads vs trace (every depth decision: started/finished counts equal the ledger's) | exact |
| queue conservation (each request served once, none left) | holds |
| dimensional consistency (score invariant to ms→s rescaling) | holds (test) |
| 31 + 1 tests in `test_c3_controller.py` | pass |

## 3. Mechanism A — scheduling (work-conserving formation, deficit-aware ranking)

**A1. Work-conserving formation vs legacy head+W windowing** (`PF` vs `legacy`, identical rank = FCFS). Lock-step populations: **bit-identical** (Δ = 0). Asynchronous
population (hetero_async), 5/5 seeds in every cell, γ=4: x_min +0.13…+0.24 tok/s, goodput +5.7…+10.0 tok/s (≈ +10–15 %); γ=2: +0.08…+0.17, +3.4…+6.4. Mean batch size
in (24,16): 6.5 → 12.3. **Reading:** the legacy rule fragments batches once arrivals de-synchronise (it closes the eligible set at head+W although the verifier is busy
until later). This is a *serving-policy / emulator-semantics* effect, not a controller gain; **it must be in the baseline**, otherwise any scheduler comparison is
inflated by 10–15 %.

**A2. Where selection can act.** `batches_eligible_gt_cap`: ≈ 0–1 % at (24,16), (32,16) in every population; 97–100 % at (48,16), (24,8), (32,8)(sync) — i.e. only when N exceeds ≈ 2·B_max
(structural argument, design §7). In the measured RQ1 regime (B_max=16, N ≤ 32) **deficit-aware ranking is vacuous**: differences vs PF are within seed noise (CIs span 0).

**A3. Ranking when the cap binds** (paired Δ vs `PF`, 5 seeds, γ=4):

| cell | DS (density) x_min | DS Y | DSz (deficit-first) x_min | DSz Y |
|---|---|---|---|---|
| hetero_async N=48 B=16 | −0.168 [−0.244,−0.092] 0/5 | +2.66 | **+0.318 [+0.260,+0.377] 5/5** | **−4.13 [−5.22,−3.04]** |
| hetero_async N=24 B=8 | +0.117 [−0.092,+0.325] | −0.31 | **+0.607 [+0.438,+0.775] 5/5** | **−2.77 [−3.41,−2.12]** |
| hetero_sync N=48 B=16 | −0.172 [−0.251,−0.093] | +3.10 | +0.315 [+0.231,+0.399] 5/5 | −3.72 |
| hetero_sync N=24 B=8 | +0.135 [−0.051,+0.321] | −0.08 | +0.632 [+0.470,+0.794] 5/5 | −2.58 |
| homog_sync N=48 B=16 | +0.180 [+0.115,+0.245] | −0.55 | +0.178 [+0.119,+0.238] | −0.95 |
| any cell with cap not binding | within noise | within noise | within noise | within noise |

Baseline levels (PF, hetero_async N=24 B=8, γ=4): x_min ≈ 1.65 tok/s, Y ≈ 62.7 tok/s ⇒ DSz is **+37 % floor for −4.4 % goodput**; Jain index 0.96 → 0.998.
Density ranking does the opposite (throughput up, floor down, Jain −0.03): it is the *throughput* end of a dial controlled by V (sensitivity: V=5/50/500 → x_min 2.13/2.04/1.87, Y 61.0/61.5/63.0).
**Reading:** deficit-first ranking is an *interactivity-for-throughput exchange* that exists only when the cap binds and clients are heterogeneous (homogeneous: Δx_min ≤ 0.25 and ≈ 0 Y).

**A4. Capacity view** (largest N with x_min ≥ x, first crossing, step 4 clients; paired over 5 seeds; CPU model): 

| population, B_max, floor | PF | legacy | DS | **DSz** | Δ(DSz−PF) |
|---|---|---|---|---|---|
| hetero_async, B=8, x≥2.0 | 19.2 | 16.8 | 20.0 | **25.6** | **+6.4 [+3.7,+9.1] (5/5)** |
| hetero_async, B=8, x≥2.5 | (see summary) | | | | +3.2 [+1.0,+5.4] (4/5) |
| hetero_async, B=16, x≥2.0 | 19.2 | 17.6 | 19.2 | 20.0 | +0.8 [−1.4,+3.0] (n.s.) |
| homog_sync, B=8/16, x≥2.0 | 48 / 53.6 | 48 / 52.0 | 48 / 56 | 48 / 56 | 0 / +2.4 [−0.3,+5.1] (n.s.) |

So the only supported capacity gain is **hetero_async with the tighter cap** (and it is a gain in N at a given floor, bought with ~4 % goodput). γ=2 vs γ=4: γ=4 raises N_c by 16–20 clients (homog) — the biggest single lever by far, and it needs no controller.

**A5. Does the gain depend on the cost model?** Predicted-cost scale errors (a×¼…4, b×¼…4) change DS/ranking outcomes by ≤ 0.1 tok/s; the optional predicted service-time cap (600/900 ms) never bound in these cells. The ranking mainly uses Z, not the cost, when in deficit-first mode.

## 4. Mechanism B — context-aware speculative depth

**B1. At the assumed cost (b_γ = 0.19 ms/token), depth control ≡ constant γ=4**: paired Δ vs `fix4+PF` is 0.000 in x_min and Y in all 15 (population, N, B) cells except one cell
with Y −0.012 [−0.044,+0.021] (a handful of decisions). The ablation without the verifier term (`depthNP`) is identical to the full rule. Analytic reason (design §8, `verifier_economics.json`): γ_i* < γ_max iff α_i^{γ_max} ≤ μ b_γ/ω_i + x̂_i c_i; with x̂ c ≈ 0.07 and μ b_γ ≈ 0.01 this needs α ≲ 0.5, and α ∈ [0.6, 0.95] here. The
verifier-time-per-useful-token argmin is γ=4 for every α ≥ 0.3, B ≥ 8, ctx ≤ 400 until b_γ exceeds a breakeven of **0.44 / 1.0 / 3.6 / 10 / 20 ms/token for α = 0.3/0.4/0.6/0.8/0.92** (B=16, ctx=160).
Context moves the **level** of cost, not the γ-slope: the cost-minimising γ is non-decreasing in ctx. A context-aware per-call cost does not by itself give a useful depth decision.

**B2. It acts only if b_γ is large and *known*** (open-loop replay vs closed loop, hetero_async N=24 B=8; fixed γ=4 acts in the open-loop run, the rule only logs):

| assumed b_γ | true b_γ | open-loop: decisions the rule would change | closed loop: decisions changed | x_min / Y vs γ=4 (1.80 / 62.3 or 1.275 / 44.3) |
|---|---|---|---|---|
| 0.19 | 0.19 | 0.0 % | 0.0 % | identical |
| 5.5 | 5.5 | 33 % | 32 % | 1.208 / 44.4 (−5 % floor, Y ≈) |
| 19 | 19 | — | 89 % | 0.725 / 27.9 vs γ=4 0.717 / 25.7 and **γ=2 0.886 / 26.7** |
| 5.5 | 0.19 | 75 % | 70 % | 1.62 / 57.5 (**−10 % floor, −8 % Y** vs γ=4) |
| 19 | 0.19 | 99 % | 99 % | 1.24 / 35.9 (**−31 % floor, −42 % Y**) |
| 19 | 5.5 | 99 % | 99 % | 1.14 / 33.9 (−10 % floor, −23 % Y) |

Open-loop (shadow) and closed-loop change rates agree (the rule changes a state-dependent minority of actions without feedback instability here). Even **well-specified**, the rule never beats the best
constant depth on the floor (b_γ=19: best constant γ=2 gives +22 % floor); a mis-assumed large b_γ costs up to 42 % goodput. The tested sweeps show no regime in which depth control is the
best arm; it is at best equal to a constant, at worst strongly harmful. Sensitivity of the congestion normaliser T_ref (13…832 ms): x_min 1.21–1.26 at assumed=true=5.5 — no qualitative change.

**B3. Learned α without optimism**: a rule that can output γ=0 is absorbed there when α̂ is learned online and no UCB bonus is used (no acceptance feedback at γ=0) — documented failure (test). C3 uses offline-profiled α̂.

## 5. Combined controller

`depth+DSz` vs `fix4+DSz`: |Δ| ≤ 0.11 tok/s (x_min and goodput) in all 15 cells (adaptive depth adds nothing on top of the scheduler); `depth+DSz` vs `fix4+PF` reproduces the DSz-vs-PF numbers of §3.
Valid to evaluate only because both components were first evaluated independently (§3, §4).

## 6. Adversarial tests (all with clean ledgers, 0 empty batches, 0 constraint violations)

| test | result |
|---|---|
| persistent overload (N=64, B=8) | no starvation (max skips 6 ≤ bound 10); arms equal in x_min (0.62); goodput 62.0–62.4 |
| weak client (α=0.35, ctx 700) among 39 strong | its rate 0.96 under all arms (limited by its own acceptance, not by scheduling); no starvation (max skips 3) |
| heterogeneous contexts up to 1500 tokens | token budget binds (4096 reached); no violation; DSz weak/long-context client rate 0.89 vs 0.78 (PF) |
| correlated arrivals (identical lock-step, N=32,B=8) | all arms bit-identical |
| 1 or 3 clients (partial batches) | batches of 1–3, no empty batch, all arms equal; **queue wait ≈ W=150 ms even when the verifier is idle** (the hold W is retained: "work-conserving" here means never leaving *eligible* work behind a busy verifier, not zero hold) |
| idle time slots (clients starting 0–20 s apart) | Z/snapshots well-defined, ledger clean; DSz max wait 359 vs PF 406 ms |
| delayed state updates | fault injection (depth rule reads a snapshot 0/0.25/1/4 s old, b_γ well specified 5.5): x_min 1.208/1.208/1.208/1.206, Y 44.4/44.5/44.3/44.4, changed-decision share 32→27 %: insensitive (`results/c3/read_lag_sensitivity.json`) |
| misspecified cost predictor | scheduler insensitive; depth rule highly sensitive (§4 B2) |

## 7. Prediction-model limitations

* (a, b) come from loaded C1 runs (LORO MAE 46 ms; 90 % intervals cover 30–56 % under batch-size extrapolation); b_γ and c_req are not identified (γ ≡ 2, collinear with batch size).
* The CPU "truth" is the same affine form as the controller's assumption (only b_γ is varied), so these results say nothing about non-affine effects (padding, chunking, KV pressure, scheduler steps beyond 4096 tokens).
* No look-ahead: predictions use formation-time features; the estimator tests show the unidentified split is *reported*, not hidden.
* Nothing fitted on the same C1 trace is presented as validation.

## 8. Recommendation per component (KEEP / REVISE / DROP)

| component | decision | reason |
|---|---|---|
| Causal coordinator, exact occupancy, D3 guard | **KEEP** | verified; required for any claim |
| Work-conserving batch formation (as *baseline* serving policy) | **KEEP (as baseline)** | removes a 10–15 % async artefact; must not be credited to AUCspec |
| Z_i deficit queue | **KEEP** | the only state that reaches actions, correctly defined |
| Deficit-first ranking under physical constraints (mechanism A) | **REVISE → candidate headline, conditional** | measurable floor/capacity gain only when the cap binds & clients are heterogeneous; costs ~4 % goodput; unconfirmed on GPU |
| Density ranking (V+Z)φ/d | **DROP** | pure throughput exchange; no capacity gain |
| Starvation guard (max_skips=2) | **KEEP** | bound holds; guard-free variants behave identically here (max_skips 2/8/∞ ⇒ x_min 2.39–2.41); max_skips=0 degenerates to FCFS |
| Context-aware depth rule (mechanism B), verifier price μ, Û gate | **DROP from headline** | ≡ constant γ_max at the assumed cost; best case ties the best constant; mis-specification harms up to −42 % goodput |
| Online cost estimator | **REVISE** (needed before any price is used) | reports unidentified γ coefficient; not wired |
| Capacity-DPP Q_s, Q_i, θ_f(γ+1); wireless allocation | **DROP / limitation** | C2 audit; wireless irrelevant for greedy payloads |
| Theorem 3 as a property of the implementation | **DROP the claim** | no matching asynchronous derivation (design §11) |

**Narrowing recommendation.** If the GPU confirms A, narrow AUCspec to *(i) the measurement of interactivity–concurrency capacity (RQ1) + (ii) a deficit-first, work-conserving
verifier scheduler under explicit physical constraints*, with adaptive depth removed from the headline and relegated to "a safeguard that needs a measured b_γ". If A is not confirmed, the paper is a measurement/characterisation paper.

## 9. Unsupported manuscript claims after C3 (non-exhaustive)

adaptive speculation depth improves capacity; Capacity-DPP's queue prices; knapsack/radio gains; Theorem 3 for the asynchronous controller; "two-regime γ*(B)" in real serving;
any claim that CPU capacity numbers transfer to the GPU; any controller-vs-baseline claim on GPU (none exists); statistical capacity certification (one seed/population in RQ1).

## 10. Implemented-and-verified / implemented-but-unvalidated / proposed / unsupported

* **[I] verified (CPU tests):** causal reads, determinism, conservation, constraint enforcement, no empty batch, starvation bound, unit invariance, estimator identifiability behaviour.
* **[U] implemented, unvalidated on hardware:** both mechanisms' effects; the cost model; the Û gate.
* **[P] proposed:** online estimator in the loop; launch-rule (hold W) optimisation; asynchronous Theorem-3 derivation.
* **unsupported:** §9.

## 11. Open issues / honest caveats

1. A3/A4 gains are against a baseline (`PF`) that I constructed; they depend on the population heterogeneity I chose (α∈[0.6,0.95], ctx 80–300, hardware ranges). Settings were fixed before results, but a different heterogeneity could shrink or grow them.
2. Capacity resolution is 4 clients; N_c is a first-crossing statistic on a noisy curve.
3. The B_max=8 regime is not the measured RQ1 regime; at B_max=16 the effect is not significant.
4. Run-seed intervals ignore population variance (1 population per cell).

## 12. Reproduction

`legacy/tests/test_c3_controller.py` (31 tests + asynchrony-knob test), `legacy/tests/c3_scenarios.py`, `legacy/tests/closed_loop_harness.py`, `scripts/c3/*`.

## 13. Final-SHA Setonix result
(appended after the last commit.)
