# AUCspec RQ1 minimal pilot plan (Task C0) — plan only, nothing launched

**RQ1 question.** Can aggregate goodput Y(N) hide a fall in per-client capacity x_min(N)? Pilot: fixed policy, corrected runtime H (`V1-batchwall-tritonfa33`, see `results/runtime_identity/`), vary N only. Descriptive result; **not** a certified N*(x), no pricing, no calibration dependence.

## 1. Design
| Item | Choice |
|---|---|
| Policy | `fixed_gamma`, γ = 2 (one simple policy; no learning, no server price) |
| Population | homogeneous symmetric clients (same RTT, uplink, SNR, draft speed), fixed `population_seed` (after fix M4) so client i is the same device in every N and seed |
| N | {1, 2, 4, 8, 12, 16} (N=1 gives isolated cycle time, 4–8 expected near saturation, 12–16 past it) |
| Seeds | 3 run-seeds per N (descriptive CIs only; no Bonferroni claim) |
| Timing | warm-up 30 s, measurement window 120 s (edge noise ≈ 0.4 round/ 120 s), early stop at window end + 10 s |
| Verifier | physical batch wall time (`batch_wall`), `verifier_price_mode=zero`, budget 100, max batch 32, wait 150 ms (unchanged) |
| Workload | `questions_per_client` = 16 (see section 2) |

## 2. Exhaustion budget (analytical, `scripts/audit_c0/rq1_exhaustion_budget.py`)
Per-client supply must satisfy questions × 96 tokens ≥ 1.5 · x_max(N) · (30 + 120) s with x_max = min(φ(2,0.9)/T_cycle_min, Y_cap/N), T_cycle_min = 0.495 s (draft 150 + RTT 50 + wait 150 + B=1 verify 145 ms), Y_cap = 30 tok/s (generous upper bound).

| N | x_max (tok/s) | tokens needed | questions needed |
|---|---|---|---|
| 1, 2, 4 | 5.47 | 1232 | 13 |
| 8 | 3.75 | 844 | 9 |
| 12 | 2.50 | 563 | 6 |
| 16 | 1.88 | 422 | 5 |

16 questions/client (≥ 13, 23% above the worst case) covers every N. Dry-run on CPU (no GPU): run the existing simulator workload loader with the per-client assignment to confirm ≥ 16 prompts exist and total decode-token supply ≥ the table; the per-client `workload_exhausted_before_end` flag (fix M1) must be false for every client, else the point is flagged INVALID and not interpreted.

## 3. Measurements per (N, seed)
Per-client useful-token rate over the exact window T_m (absent = 0), x_min, mean, Jain index, aggregate Y = Σx, N·x_min; verifier utilization (busy/ T_m), batch-size histogram, batch wall time per size, queue wait, round-latency decomposition; workload completion and exhaustion flags; virtual-time consistency (missed-arrival fraction D1, future-state fraction D2 after fix) and wall vs virtual clock ratio; edge-round counts.

## 4. Support / falsification of RQ1
- **Supports RQ1:** Y(N) plateaus (within ±10% for N ≥ N_sat, e.g. N=8→16) while x_min(N) keeps falling with N·x_min within ±15% of constant and log-log slope of x_min vs N in [−1.3, −0.7]. Then aggregate goodput is blind to the capacity loss and the per-client floor is the informative axis.
- **Falsifies the strong form:** Y(N) keeps rising ~linearly with N (x_min flat) over the range — the system is not saturated and N* is above 16; extend N, not claim.
- **Falsifies the interpretation:** x_min collapses faster than 1/N (slope < −1.3) or Y falls with N → contention/overhead pathology to diagnose before any boundary claim.
- Results are descriptive; no "certified N*(x)" language. Not dependent on calibrated verifier pricing.

## 5. GPU cost
Run length ≈ warm-up+window+margin (~160 s virtual) but real time ≈ 3–4× per round (draft on shared GPUs). With early stop: 6 N × 3 seeds ≈ 18 runs; N-weighted ≈ 1.65 node-hours (3 jobs of ~35–40 min each on 1 node with gres gpu:8, one seed per job, N loop inside). Without early stop ≈ 3.4 node-hours. Reduced option (N ∈ {1, 4, 8, 16}, 2 seeds) ≈ 0.75 node-hour.

## 6. Conditions to authorize the pilot
1. CPU fixes landed and unit-tested: M1 (true exhaustion flag), M2 (exhausted ⇒ INVALID), M12 (early stop with flag), M4 (population seed fixed), M3 (Student-t classifier, only needed if a classifier is used; pilot is descriptive) and D1/D2 either fixed (causal barrier) or declared as a limitation with the missed-arrival fraction reported.
2. Dry-run of the workload budget passes for all N.
3. Provenance guard (`canonical_guard`) unchanged; runtime H identity file committed.
4. Review of this plan by the user.

## 7. RQ3/RQ4 readiness (plan only)
Before RQ3/RQ4: (a) corrected-runtime calibration, isolated vs loaded, with physical full-context work as a regressor (explain the ≈ +86 ms/batch loaded gap); (b) normalise server price and Q_i into tokens/ms-consistent units and instrument decision-change rates per price; (c) baseline verifier-latency proxies non-zero in fixed mode; (d) causal virtual-time barrier; (e) choose an operating point where batching/bandwidth actually bind (otherwise the ablations are null). Do not enable pricing or tune κ_s until (a)–(e) exist.
