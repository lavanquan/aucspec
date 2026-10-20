# AUCspec C2 — Minimal RQ3/RQ4 experiment design (not launched)

Companion to `AUCSPEC_C2_CONTROLLER_MECHANISM_AUDIT.md`. Nothing here has been submitted; no GPU used in C2.

## 1. Starting point and what is allowed to be claimed

* RQ1 (descriptive anatomy) is closed: single seed, homogeneous, lockstep-oriented; not a certified capacity boundary.
* The audit shows the production controller ≡ constant γ_max in homogeneous runs and no better than constant γ in the heterogeneous harness. RQ3 ("does the capacity-aware
  controller raise the achievable interactivity capacity?") therefore **cannot yet be a headline** and the first experiment must be a *mechanism-validating policy comparison*,
  not a frontier search.
* High-load `batch_cap_16` region (N=16–32) is the natural place to start (verifier busy 0.68→1.0), but it is not an automatic headline: N≥24 is exploratory_v2-flagged.

## 2. Testable quantity

Controller-achieved interactivity capacity at floor x: `N_c(x; π) = max{N : x_min(N; π) ≥ x}` with seed-level bootstrap CI over ≥3 run-seeds × ≥2 population seeds, for a
**fixed set of floors** (e.g. x ∈ {2.5, 3.0, 3.5} tok/s) chosen *before* looking at data. Secondary: Y at N_c. No AUC/frontier search in the pilot.

## 3. Population and arrival structure

Permanent lockstep is an artefact (cohorts phase-lock; knapsack inert). Require an asynchronous heterogeneous population:
* staggered start offsets uniform in [0, mean round time];
* per-client α_i ∈ [0.6, 0.95] and heterogeneous draft hardware (τ_d spread ×1.5), prompt lengths drawn from the catalog (context heterogeneity is what makes verifier cost non-uniform);
* open-loop think-time jitter (±20 %) between requests to break phase locking.
Acceptance check before spending GPU: in the CPU harness the *eligible-set size* must exceed the cap in ≥10 % of batches for the batching policy comparison to be non-vacuous.

## 4. Fair baselines (equal information, same physical verifier)

| arm | speculation | batching |
|---|---|---|
| B0 | constant γ ∈ {2, 4} | FCFS, cap 16 |
| B1 | per-client α̂-greedy γ (oracle α, no queues) | FCFS |
| B2 | Z-only DPP (queues off except Z) | FCFS |
| B3 | corrected Capacity-DPP (design §7 of the audit: context-aware, normalised, fresh snapshot) | FCFS |
| B4 | B3 | work-conserving server-side batching policy (below) |

All arms use the same V1-greedy protocol, same runtime H, same physical constraints (max_batch 16, token budget 4096), same α information (oracle α̂ for fairness; learning
is a separate RQ).

**Compute control placement**: the audit finds the batching knob has no lever under head+W eligibility. Test it as a *batching policy* (work-conserving: launch when verifier
idle with all arrived requests, choose by context-cost knapsack) vs. an *inference-backend* knob (chunked prefill / max_num_batched_tokens), which is out of scope (no backend changes).

## 5. Wireless

For the greedy-SD protocol the allocator effect is ≤2 ms (0.25 % of a round): **reduce prominence** (limitation / appendix). A comm-limited regime requires full-distribution
uploads (1.2 MB per round, ≥121 Mbit/s needed to make uploads ≤10 % of a round); only then is allocation (square-root vs proportional: 77 ms vs 1480 ms mean upload) a lever.
No wireless arm in the pilot.

## 6. γ dose-response anchor (required before any price is set)

The cost model cannot identify the γ coefficient (γ constant in all traces). Pilot includes constant-γ ∈ {0,1,2,4} at N∈{16,24} with matched seeds to estimate the depth
dependence of batch wall time and acceptance; this is *calibration for the controller's cost term*, kept separate from the policy comparison and not reused as test data.

## 7. Minimal GPU pilot (estimate only — NOT submitted)

| block | runs | note |
|---|---|---|
| γ dose-response | 4 γ × N∈{16,24} × 1 seed = 8 | cost-term identification |
| policy comparison | 5 arms × N∈{16,24,32} × 2 run-seeds = 30 | async hetero population |

Reference cost ≈0.2–0.25 node-h per N≥24 run (0.1–0.15 for N≤16). Estimated ≈ 38 runs × ~0.2 ≈ **7–9 node-h** — **exceeds the remaining 1.474 node-h budget by ~5×**.
Smallest affordable cut (1.2 node-h): drop to 3 arms (B0 γ=4, B2, B3) × N∈{16,24} × 2 seeds = 12 runs ≈ 2.4 nh → still over; truly budget-feasible pilot: 3 arms × N=24 × 1 seed + γ=0/1/2 at N=16 ≈ 6 runs ≈ 1.2 nh. Such a pilot yields **no statistics** and is only a smoke test of the corrected controller; it should not be launched until the corrected controller passes the CPU harness criteria below. Budget increase needs user approval.

## 8. Pre-registered CPU gates before any GPU launch

1. Corrected controller changes γ vs constant γ in ≥10 % of decisions in the *homogeneous* regime *and* beats constant γ=4 floor in the harness (reported either way).
2. Batching policy changes member sets vs FCFS in ≥10 % of multi-request batches (currently 0 %).
3. Price is dimensionless; scale perturbation ×0.5/×2 changes x_min by <10 %.
4. Decisions identical across permuted asyncio order (existing E5 test).
5. Stale-state age ≤ 1 slot for Q_i/Q_s reads.

If gate 1–2 fail again, the mechanism is reported as inactive and moved to limitations (audit §8).

## 9. What stays unclaimed

No frontier, no AUC, no ICA, no "capacity boundary", no wireless benefit under greedy SD, no Theorem 1/3 properties for the implemented controller.
