# Capacity-Aware Framework -- Run Report

run_tag: `gate_d_small_frontier`  
config_hash: `9f88e5c288a37154`  
git_sha: `aef4d9a092bb441b2035b77cd245d62aecca7bd6`


## capacity_dpp

ICA (lower, upper): None, None  
points: 3 (exact: 2, cap-limited: 0)

```
      policy  x_requirement  last_confirmed_feasible_N  first_confirmed_infeasible_N  N_lower  N_upper  exact_boundary  hit_search_cap  boundary_certified  uncertain_n  seeds_at_boundary
capacity_dpp            3.0                         16                            17       16       17            True           False                True         16.0                  7
capacity_dpp            4.5                          5                             8        5        8           False           False               False          6.0                  3
capacity_dpp            6.0                          3                             4        3        4            True           False                True          NaN                  5
```

## turbospec

ICA (lower, upper): None, None  
points: 3 (exact: 3, cap-limited: 0)

```
   policy  x_requirement  last_confirmed_feasible_N  first_confirmed_infeasible_N  N_lower  N_upper  exact_boundary  hit_search_cap  boundary_certified  uncertain_n  seeds_at_boundary
turbospec            3.0                          3                             4        3        4            True           False                True          NaN                  5
turbospec            4.5                          0                             1        0        1            True           False                True          NaN                  0
turbospec            6.0                          0                             1        0        1            True           False                True          NaN                  0
```

---

This run used the FINAL exact-SLO classifier (classify_feasibility_final, Bonferroni-bootstrap confidence bounds) and NStarResult search semantics. Cap-limited points are reported as lower bounds, never as N*=N_cap. See CAPACITY_AWARE_FRAMEWORK_CODEX_IMPLEMENTATION.md for the full contract.
