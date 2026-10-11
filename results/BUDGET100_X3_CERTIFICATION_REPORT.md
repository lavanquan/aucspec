# budget100, x=3.0 boundary certification (capacity_dpp)

**Conclusion: N*(3.0) in [10, 11], unresolved (N=10 certified feasible; N=11 remains uncertain after the maximum number of 480 s seeds, so it is neither confirmed feasible nor infeasible; N=12 was not run under the 480 s protocol)**

| N | status | seeds | measurement_s | min-rate range | min-rate mean | CI/decision | mean gamma | mean batch | goodput |
|---|---|---|---|---|---|---|---|---|---|
| 10 | feasible | 9 | 480 | 2.971-3.167 | 3.069 | LCB 3.004 / UCB 3.132 vs x=3.0 | 2.45 | 2.69 | 33.77 |
| 11 | uncertain_at_max_seeds | 12 | 480 | 2.904-3.035 | 2.962 | LCB 2.949 / UCB 3.034 vs x=3.0 | 2.41 | 2.85 | 35.80 |

Decision rule: shortest seed prefix k in [5, 12] whose exact-SLO Bonferroni-bootstrap bounds (delta=0.05) are decisive; feasible iff min_i LCB >= 3.0, infeasible iff min_i UCB < 3.0, else uncertain.

## Provenance

- git commit used for runs: `921dedd4020c6dceac720f27a4a44463b959d095`
- config hash: `78e092235456de2e` (includes measurement_seconds_final=480.0, so 240 s records cannot be reused)
- population fingerprint: `7d873c2e94ab447b`
- warmup_s=30, measurement_s=480.0, initial_seeds=5, max_seeds=12
- N=10: seeds run [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; seeds used for verdict [1, 2, 3, 4, 5, 6, 7, 8, 9]
  - secondary check, classifier on all 12 seeds: feasible (LCB 3.026 / UCB 3.170)
- N=11: seeds run [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; seeds used for verdict [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  - secondary check, classifier on all 12 seeds: uncertain (LCB 2.949 / UCB 3.034)
- workload exhaustion occurred: no
- candidates still uncertain: [11]

## Procedure trace

1. Step 1: N=10 with seeds 1-5 (480 s): LCB 2.974 / UCB 3.114, uncertain. Seeds 6-12 added; the shortest decisive prefix is k=9 (LCB 3.004 >= 3.0): feasible.
2. Step 2B (N=10 feasible): N=11 with seeds 1-5: LCB 2.914 / UCB 3.043, uncertain. Seeds 6-12 added; still uncertain after 12 seeds x 480 s, so no verdict is forced and the search stops. N=12 and N=9 were not run (neither is required by the procedure in this branch).
3. Seeds beyond the decisive prefix were run in parallel (all 12 seeds exist for N=10 and N=11) and are reported as a secondary check only; verdicts use the shortest decisive prefix.

## Observations / anomalies

- N=10 is feasible only narrowly: min-rate mean 3.069 tok/s against x=3.0, per-seed range 2.971-3.167, and the verdict needs 9 seeds. Two of the 12 seeds (seed 1: 2.971, seed 3: 2.979) are below 3.0 on their own.
- N=11 has min-rate mean 2.962 (range 2.904-3.035), i.e. slightly below 3.0, and 9 of 12 single-seed min-rates are below 3.0, but the classifier works on per-client means across seeds with a Bonferroni-adjusted bootstrap interval, and its upper bound (UCB 3.034) still contains 3.0, so N=11 is not declared infeasible. The boundary lies between N=10 and N=11 up to finite-sample noise; whether N=11 is truly infeasible is not established.
- The earlier 240 s sweep (results/BUDGET100_SWEEP_REPORT.md) reported N=9 feasible, N=10 uncertain, N=12 infeasible. The 480 s runs confirm N=10 feasible, so the earlier N*=9 was a conservative lower bound caused by the shorter window. Those 240 s records were not mixed into any verdict here.
- No workload exhaustion in any of the 24 runs. No classifier, cache, or measurement bug was observed.
- Run code revision is 921dedd (harness commit). A later commit changes only the reporting script (interval wording and a secondary all-seed check) and its test; the runner and simulator are unchanged.
