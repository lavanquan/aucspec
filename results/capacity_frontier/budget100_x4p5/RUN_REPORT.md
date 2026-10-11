# Capacity-Aware Framework -- Run Report

run_tag: `budget100_x4p5`  
config_hash: `8a033e1fea8e2388`  
git_sha: `e5670f3a2b52b411652f52dba16c21b66f9af4b4`


## capacity_dpp

ICA (lower, upper): None, None  
points: 1 (exact: 1, cap-limited: 0)

```
      policy  x_requirement  last_confirmed_feasible_N  first_confirmed_infeasible_N  N_lower  N_upper  exact_boundary  hit_search_cap  boundary_certified  uncertain_n  seeds_at_boundary
capacity_dpp            4.5                          5                             6        5        6            True           False                True          NaN                  5
```

---

This run used the FINAL exact-SLO classifier (classify_feasibility_final, Bonferroni-bootstrap confidence bounds) and NStarResult search semantics. Cap-limited points are reported as lower bounds, never as N*=N_cap. See CAPACITY_AWARE_FRAMEWORK_CODEX_IMPLEMENTATION.md for the full contract.
