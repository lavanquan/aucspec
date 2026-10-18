# budget100, x=3.0 boundary certification (capacity_dpp)

**Conclusion: N*_capacity_dpp(3.0) in [10, 11], certified interval under the 480-s protocol (N=10 feasible, N=11 uncertain after the maximum number of seeds and not resolved, N=12 infeasible)**

| N | status | seeds | measurement_s | min-rate range | min-rate mean | CI/decision | mean gamma | mean batch | goodput |
|---|---|---|---|---|---|---|---|---|---|
| 10 | feasible | 9 | 480 | 2.971-3.167 | 3.069 | LCB 3.004 / UCB 3.132 vs x=3.0 | 2.45 | 2.69 | 33.77 |
| 11 | uncertain_at_max_seeds | 12 | 480 | 2.904-3.035 | 2.962 | LCB 2.949 / UCB 3.034 vs x=3.0 | 2.41 | 2.85 | 35.80 |
| 12 | infeasible | 6 | 480 | 2.712-3.025 | 2.831 | LCB 2.745 / UCB 2.988 vs x=3.0 | 2.37 | 3.04 | 38.01 |

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
- N=12: seeds run [1, 2, 3, 4, 5, 6, 7, 8]; seeds used for verdict [1, 2, 3, 4, 5, 6]
  - secondary check, classifier on all 8 seeds: infeasible (LCB 2.740 / UCB 2.967)
- workload exhaustion occurred: no
- candidates still uncertain: [11]

## Boundary statement

- N=10 feasible (certified, 9 seeds), N=12 infeasible (certified, 6 seeds), N=11 unresolved after 12 seeds x 480 s.
- N*_capacity_dpp(3.0) is therefore bracketed as [10, 11]; the upper end relies on feasibility being monotone in N (as assumed by the search). N=10 and N=11 are not distinguished, and no equality is claimed.
- No 240 s record entered any verdict in this report.

## Procedure trace

1. N=10, seeds 1-5 (480 s): LCB 2.974 / UCB 3.114, uncertain. Seeds 6-12 added; shortest decisive prefix k=9 (LCB 3.004 >= 3.0): feasible.
2. N=11, seeds 1-5: LCB 2.914 / UCB 3.043, uncertain. Seeds 6-12 added; still uncertain after 12 seeds x 480 s, no verdict forced.
3. N=12, seeds 1-5: LCB 2.741 / UCB 3.009, uncertain (UCB barely above 3.0). Wave seeds 6-8 added; k=6 gives LCB 2.745 / UCB 2.988 < 3.0: infeasible. Seeds 7-8 ran in the same wave and are shown only in the secondary check (all 8 seeds: infeasible, UCB 2.967). No further seeds were run.
4. N=12 per-seed min-rates (tok/s): 3.025, 2.846, 2.783, 2.713, 2.844, 2.775, 2.775, 2.700. Only seed 1 reaches 3.0.

## Observations / anomalies

- Workload exhaustion: none in any of the 32 runs (N=10, 11 with 12 seeds each, N=12 with 8 seeds).
- The first N=12 submissions (10 jobs plus 2 diagnostic jobs) failed after ~8 s with no log: the per-user file-count limit on /software was exceeded (250,378 files, 100.2%), so Slurm could not create log files. No simulator ran, so no N=12 data was produced by those jobs. About 2,144 raw-trace files (~20 GB) of finished aucspec runs were archived with tar to /scratch/pawsey1257/quanla/aucspec_raw_archive/ (each tar verified with tar --compare, file counts matched) and then removed from /software; compact results in git are unaffected. /scratch has a purge policy, so those tars are not permanent storage.
- The N=12 jobs ran from a clean git worktree at 921dedd, with the run directory symlinked in, so the runner revision, config hash and population fingerprint are identical to the N=10/11 runs. The main working tree was dirty (uncommitted edits to legacy/src/edge_specsim/target_client.py and scripts/collect_trace.py, HEAD detached at cf5f434); those edits were made after the N=10/11 runs finished (2026-10-11 14:22 vs 21:17 and 2026-10-12 08:00) and were not used for any run here. certify_x3_run.py checks only HEAD, not a dirty tree, so it would not have caught such edits.
- The uncommitted target_client.py edit changes the greedy tie-break in the verifier and max_logprobs; if it is merged, this certification should not be assumed to carry over unchanged.
- Run code revision is 921dedd. The later commits change only reporting and tests (the conclude() logic of the report script and its regression test).
