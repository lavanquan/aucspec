# Capacity-Aware Framework -- Run Report

run_tag: `gate_b_x4p5`  
config_hash: `1678f63d784c1adb`  
git_sha: `aef4d9a092bb441b2035b77cd245d62aecca7bd6`


## capacity_dpp

ICA (lower, upper): None, None  
points: 1 (exact: 1, cap-limited: 0)

```
      policy  x_requirement  last_confirmed_feasible_N  first_confirmed_infeasible_N  N_lower  N_upper  exact_boundary  hit_search_cap  boundary_certified  uncertain_n  seeds_at_boundary
capacity_dpp            4.5                          8                             9        8        9            True           False                True            8                  7
```

---

This run used the FINAL exact-SLO classifier (classify_feasibility_final, Bonferroni-bootstrap confidence bounds) and NStarResult search semantics. Cap-limited points are reported as lower bounds, never as N*=N_cap. See CAPACITY_AWARE_FRAMEWORK_CODEX_IMPLEMENTATION.md for the full contract.
