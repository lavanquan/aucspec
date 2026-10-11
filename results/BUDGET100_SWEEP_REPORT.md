# verify_token_budget = 100 sweep: N*(x) for capacity_dpp

Single policy (`capacity_dpp`), `scheduler: capacity_knapsack`, `verify_token_budget: 100`
(all other settings identical to `configs/capacity_frontier_gate_d.yaml`, where the budget is 10).
Seven x requirements, one independent run per x (`results/capacity_frontier/budget100_x<x>/`),
run in parallel on the Setonix `gpu` partition. Final-mode exact-SLO classifier, 3-7 seeds per candidate.

## N*(x) at budget=100

| x_req | N* | exact | boundary certified | first infeasible N | uncertain N | seeds at boundary |
|---|---|---|---|---|---|---|
| 2.0 | 21 | yes | yes | 22 | 20;22 | 5 |
| 3.0 | 9 | no | no | 12 | 10 | 3 |
| 4.0 | 6 | yes | yes | 7 | - | 5 |
| 4.5 | 5 | yes | yes | 6 | - | 5 |
| 5.0 | 4 | yes | yes | 5 | - | 5 |
| 5.5 | 3 | yes | yes | 4 | - | 5 |
| 6.0 | 3 | yes | yes | 4 | - | 5 |

## Comparison with budget=10 (Gate D, `results/capacity_frontier/gate_d_small_frontier/`)

| x_req | N* budget=10 | N* budget=100 |
|---|---|---|
| 3.0 | 16 (certified, 7 seeds) | 9 (not certified; N=10 uncertain) |
| 4.5 | 5 (not certified; N=6 uncertain) | 5 (certified) |
| 6.0 | 3 (certified) | 3 (certified) |

x = 2.0, 4.0, 5.0, 5.5 have no budget=10 counterpart (outside the Gate D grid).

## Observations

- Raising the verifier budget 10 -> 100 did not raise N* anywhere it can be compared.
  At x = 4.5 and 6.0 the result is unchanged; at x = 3.0 it is lower (9 vs 16).
- The frontier is strictly decreasing in x: 21, 9, 6, 5, 4, 3, 3. The drop from x=2 to x=3 is the
  steepest; from x=5.5 to x=6.0 it is flat at N=3.
- The x = 3.0 point at budget=100 is the only uncertified one: N=10 stayed statistically
  uncertain through all 7 seeds, so N*=9 should be read as a lower bound with N in [9, 11] possible.

## Caveats

- Cross-run GPU/vLLM non-determinism was already seen between Gate B and Gate D at N=8, x=4.5
  (~4.6 vs ~4.0 tok/s for identical simulation settings). The x=3.0 difference (16 vs 9) is larger than that
  noise would suggest, but it is a single comparison from separately-run searches; it has not been
  re-measured with matched seeds.
- Why the lower N* at x=3.0 is not established. Larger batches lengthening per-round latency is a
  hypothesis; no per-batch latency or mean-gamma breakdown was analysed for these runs.
- `capacity_dpp` only; turbospec was not run at budget=100.

## Provenance

- Configs: `configs/capacity_frontier_budget100.yaml` and per-x `configs/capacity_frontier_budget100_x*.yaml`
- Runs: `results/capacity_frontier/budget100_x2p0` ... `budget100_x6p0` (compact outputs only; `raw/` stays gitignored)
- Fixed code at the time of these runs: commit `4b24977` (CandidateCache per_client_rates key fix) and later
