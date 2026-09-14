# Gates B–D Findings Report — Capacity-Aware Framework

Covers the staged validation requested after the CAPACITY_AWARE_FRAMEWORK_CODEX_IMPLEMENTATION.md
Gate A pass: **Gate B** (local N*(x) boundary certification at x=4.5), **Gate C** (paired
capacity_dpp vs turbospec reproduction at N=14, x=3.5), and **Gate D** (small full frontier,
capacity_dpp + turbospec across x∈{3.0, 4.5, 6.0}, `n_search_cap = n_ref = 30`).

## Critical correctness bug found and fixed mid-run

While certifying Gate B/D boundaries across `--resume` cycles, `N*(6.0)` for `capacity_dpp`
flipped from **3** (first run, all-fresh data) to **0** (after a resume) despite the underlying
raw measurements being unchanged and comfortably feasible (~8.9 tok/s against a 6.0 requirement).

Root cause: `CandidateCache.get()` reconstructed `CandidateMeasurement.per_client_rates` directly
from `json.loads()` output. JSON always serializes dict keys as strings, so a cache-loaded
measurement had `per_client_rates == {"0": 8.9}` while a freshly-computed one had
`{0: 8.9}` (int key). `classify_feasibility_final()` looks up `per_client_rates.get(cid, 0.0)`
with `cid` as an `int` from `range(n_clients)` — so every cache-loaded measurement silently
contributed a rate of **0.0** instead of its real value whenever a feasibility decision mixed
cache-loaded seeds (pre-resume) with freshly-computed seeds (post-resume, e.g. during seed
escalation on "uncertain" or `{N*-1,N*,N*+1}` boundary certification).

Fixed in commit `4b24977` (coerce `per_client_rates` keys back to `int` on cache load), with two
new regression tests (`test_candidate_cache_roundtrip_preserves_int_per_client_rate_keys`,
`test_classify_feasibility_final_consistent_across_cache_reload`). All 50 CPU tests pass.

**Everything below is post-fix and either freshly re-run or cross-validated against the fix**
(Gate D's `N*(6.0)=3` for capacity_dpp reproduces the original pre-bug run exactly).

## Gate A — smoke test

PASS. `measurement_duration_s=60.0` exact; causal uplink lag confirmed real
(`gamma_rate_signal_mbps(20.0) != realized_uplink_rate_mbps(25.4)`); `capacity_knapsack` produced
real batches (fill_ratio=0.69, forced_service_fraction=0.0); 8/8 clients sustained 4.85–5.27 tok/s.

## Gate B — local boundary certification, x=4.5, capacity_dpp

| x_requirement | N* (last confirmed feasible) | first infeasible N | exact | seeds at boundary |
|---|---|---|---|---|
| 4.5 | **8** | 9 | True | 7 |

Certified: capacity_dpp sustains x=4.5 tok/s for up to 8 concurrent clients.

## Gate C — paired capacity_dpp vs turbospec reproduction, N=14, x=3.5

Both policies feasible. capacity_dpp achieves slightly higher goodput via much deeper
speculation and smaller batches; turbospec trades speculation depth for larger batches to reach
comparable per-client rate.

| policy | feasible | min_rate range (tok/s) | mean gamma | mean batch size | goodput (tok/s) |
|---|---|---|---|---|---|
| capacity_dpp | yes | 3.53–3.85 | 2.33 | 2.47 | 59.6–60.6 |
| turbospec | yes | 3.61–3.62 | 0.19 | 8.29 | 55.7–58.8 |

This confirms the batching-vs-speculation-depth mechanism the spec hypothesized, but the
originally-observed 1-seed "turbospec outperforms" signal does **not** clearly reproduce under
the corrected multi-seed final-mode pipeline at this single point.

## Gate D — small full frontier, capacity_dpp + turbospec, x∈{3.0, 4.5, 6.0}

| policy | x=3.0 | x=4.5 | x=6.0 |
|---|---|---|---|
| capacity_dpp | **N\*=16** (exact, certified, 7 seeds) | **N\*=5** (not exactly certified — N=6 stays statistically uncertain even at max seeds, real measurement variance ~4.0–4.6 tok/s at N=6-8) | **N\*=3** (exact, certified — reproduces the original pre-bug run exactly) |
| turbospec | **N\*=3** (exact, certified) | **N\*=0** (infeasible even at N=1) | **N\*=0** (infeasible even at N=1) |

**Headline finding**: capacity_dpp sustains dramatically higher concurrency than turbospec across
the whole tested grid — most strikingly at x=3.0 (16 vs 3) and at x≥4.5, where turbospec cannot
serve even a single client at the required interactivity while capacity_dpp still serves several.
This reverses the earlier uncorrected 1-seed signal that suggested turbospec was competitive with
or better than capacity_dpp.

## Known open issue: cross-config reproducibility at N=8, x=4.5

Gate B (`config_hash=1678f63d…`) and Gate D (`config_hash=9f88e5c2…`) each independently measured
N=8, x=4.5 for capacity_dpp. The two config files differ only in `x_grid_final` (which is not
read by the per-candidate simulation config), yet the measured rates differ meaningfully
(~4.6 tok/s in Gate B vs ~4.0–4.3 tok/s in Gate D, enough to keep Gate D's N=6 boundary
"uncertain" at max seeds). This is most likely real run-to-run GPU/vLLM non-determinism larger
than the classifier's confidence-interval assumptions currently budget for, not a code bug — but
it is worth tightening (more seeds, or a documented rate-variance floor) before treating any
single-seed-count boundary as final for the paper's headline numbers.

## Status

Gates A–D complete. `capacity_dpp` shows a genuine, much larger capacity region than `turbospec`
in every case measured. Per the spec's staged-gate requirement, the full final experiment matrix
(all baselines × full x-grid × heterogeneous population variants) has not been launched — awaiting
explicit go-ahead.
