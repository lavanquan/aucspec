# Capacity-Aware Framework -- Run Report

run_tag: `budget100_x2p0`  
config_hash: `0041cb7f75956f77`  
git_sha: `e5670f3a2b52b411652f52dba16c21b66f9af4b4`


## capacity_dpp

ICA (lower, upper): None, None  
points: 1 (exact: 1, cap-limited: 0)

```
      policy  x_requirement  last_confirmed_feasible_N  first_confirmed_infeasible_N  N_lower  N_upper  exact_boundary  hit_search_cap  boundary_certified uncertain_n  seeds_at_boundary
capacity_dpp            2.0                         21                            22       21       22            True           False                True       20;22                  5
```

---

This run used the FINAL exact-SLO classifier (classify_feasibility_final, Bonferroni-bootstrap confidence bounds) and NStarResult search semantics. Cap-limited points are reported as lower bounds, never as N*=N_cap. See CAPACITY_AWARE_FRAMEWORK_CODEX_IMPLEMENTATION.md for the full contract.
