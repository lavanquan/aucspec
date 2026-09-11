# Capacity-AUC N* — Baseline Comparison (compact run)

Follow-up to `results/CAPACITY_AUC_NSTAR_IMPLEMENTATION_REPORT.md`. Every
policy is run through the identical outer capacity search (Section 18),
same deployment, same nested population, same seed.

**Deliberately fast settings** (per explicit request to keep this a quick
check, not the final matrix): `x_grid = [2.0, 3.5]`, `n_search_cap = 14`,
**1 seed per candidate**, `measurement_seconds = 60`, homogeneous GSM8K,
`N_ref = 30` for ICA. With 1 seed the feasibility classifier has no
variance band — a candidate is feasible iff its single measured `min_rate`
clears the tolerance, infeasible iff it doesn't; there is no `uncertain`
escalation. This trades statistical robustness for speed; treat the exact
`N*` values as indicative, not final (Phase 5 should re-run with ≥3 seeds
per the implementation report's recommendation).

## Results

| policy | N\*(2.0) | N\*(3.5) | ICA (extrapolated to [0.5,4.0]) | mean γ | mean batch |
|---|---|---|---|---|---|
| capacity_dpp | 14 (cap) | 14 (cap) | 0.467 | 2.34–2.36 | 2.40–2.43 |
| goodspeed | 14 (cap) | 14 (cap) | 0.467 | 2.33–2.34 | 2.40–2.42 |
| turbospec | 14 (cap) | 14 (cap) | 0.467 | **0.07–0.09** | **6.9–8.4** |
| fixed_slo | 14 (cap) | **8** | **0.438** | — | — |

(`ICA` integrates the step function over the configured `[x_L, x_U] =
[0.5, 4.0]` window, which extends beyond the measured `[2.0, 3.5]` grid —
the step function is carried flat outside the measured range, so the
absolute ICA values are an extrapolation. Only the *within-grid* `N*`
values and the *relative* ranking are load-bearing here.)

## What this run shows

1. **`capacity_dpp`, `goodspeed`, and `turbospec` all saturate the
   `n_search_cap = 14` at both `x = 2.0` and `x = 3.5`** — this grid/cap
   combination is too easy for three of the four policies to
   differentiate them; a real comparison needs a higher `x` (or a lower
   `n_search_cap`) so the frontier actually bends for all of them within
   the swept range, as it already does for `capacity_dpp` at `x = 4.5–6.0`
   in the pilot (`N* = 6, 2`).
2. **`fixed_slo` is the first policy whose frontier bends inside this
   grid** (`N* = 14 → 8` from `x = 2.0` to `3.5`) — a concrete, measured
   difference: the fixed per-round SLO heuristic can't hold the
   interactivity floor at higher concurrency as well as the queue-based
   policies here, giving it the lowest ICA of the four.
3. **`turbospec`'s mechanism looks structurally different** even though
   its `N*` matches `capacity_dpp`/`goodspeed` at this grid: mean `γ` is
   near zero (0.07–0.09, i.e. it mostly runs target-only) while its mean
   batch size is 3× larger (6.9–8.4 vs ~2.4) — it is trading speculation
   depth for batch size to hit the same capacity numbers. Worth a closer
   look in Phase 5 rather than reading its equal `N*` as "equivalent
   policy".

## Recommendation

This 1-seed / 2-point run is a sanity/orientation pass, not a paper-ready
baseline table. Before drawing conclusions for the writeup:
- widen the `x`-grid past the point where `capacity_dpp` bends (the pilot
  put that around `x ≈ 4.5`), so all policies are compared where the
  frontier is actually informative, not capped;
- restore ≥3 seeds per candidate (Phase 5 default) so `N*` differences of
  1–2 aren't within single-run noise;
- keep the per-round trace for the `turbospec` runs (`mean_batch_size`
  outlier) for a follow-up mechanism figure.

## Files

`results/capacity_frontier/{nstar_frontier_<policy>.csv,
feasibility_decisions_<policy>.csv, ica_summary_<policy>.json,
policy_comparison.csv, candidate_cache.jsonl, fig1_nstar_frontier.png,
fig2_ica_comparison.png}`. Raw per-round traces gitignored.
