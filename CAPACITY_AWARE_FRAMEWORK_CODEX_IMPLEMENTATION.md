# AUCspec Capacity-Aware Control and Characterization Framework

## Codex implementation specification

This document is the implementation contract for the framework described in the current paper Section 4, **Capacity-Aware Control and Characterization Framework**. It supersedes the old experimental shortcuts in `CAPACITY_AUC_NSTAR_IMPLEMENTATION.md` wherever the two disagree.

The goal is to turn the current prototype into a paper-faithful, reproducible implementation of the pipeline

\[
(N,x)
\rightarrow \text{Capacity-DPP}
\rightarrow \text{fixed-window measurements}
\rightarrow \widehat{\mathsf{FEAS}}(N,x)
\rightarrow \text{monotone boundary search}
\rightarrow \{N^\star(x),x^\star(N)\}
\rightarrow \mathrm{ICA}.
\]

The implementation must preserve the existing real Qwen/vLLM simulator and baseline policies. Do **not** rewrite the simulator from scratch. Extend the current code with small, testable components and keep all old policies reproducible.

---

# 0. Non-negotiable semantics

These rules define correctness. Do not change them for convenience.

1. **The primary object is the interactivity--concurrency capacity region.**

   \[
   \mathcal C(\xi)=\{(N,x): \exists\pi,\ \bar x_i(\pi)\ge x,\ \forall i\le N\}.
   \]

2. **A candidate is a pair `(N, x)`** for a fixed deployment/configuration `xi`. The runtime controller serves that fixed candidate. The outer procedure changes `N` or `x` only between candidate runs.

3. **`V` is fixed while tracing a capacity frontier.** Never sweep `V` to create `N*(x)` or `x*(N)`.

4. **Concurrency `N` and realized verifier batch size `B_t` are different quantities.** Never use one as a synonym for the other.

5. **Every admitted client remains in the service statistic.** A client that receives zero useful tokens in the measurement window has rate zero. Never drop slow/idle clients from `min_i x_i`.

6. **Use a common, fixed measurement denominator.** For a configured measurement interval of exactly `T_measurement` seconds,

   \[
   \widehat x_i=\frac{A_i}{T_{\rm measurement}}.
   \]

   Do not use `(last_client_receive_ms - first_client_receive_ms)` as the denominator.

7. **The SLO is exactly `x`.** Statistical/numerical tolerance must not redefine the requirement as `(1-eps)*x`.

8. **If the search reaches a cap and the cap is feasible, report a lower bound**, e.g.

   \[
   N^\star(x)\ge N_{\rm cap},
   \]

   not `N*=N_cap`.

9. **Do not extrapolate a measured capacity frontier outside the measured `[x_L,x_U]` interval** when reporting the headline ICA.

10. **Every baseline must run through the same outer capacity-search and measurement pipeline.** A baseline operating point is not comparable to a proposed-policy frontier.

11. **All final result directories are immutable and provenance-complete.** Do not overwrite a canonical `nstar_frontier.csv` with a later quick run.

12. **Do not claim an exact offline oracle.** `KnownAlpha` means the same online controller with a known/static acceptance profile. If an actual offline oracle is implemented later, give it a different name.

---

# 1. Current repository state and known problems

The current implementation already contains useful pieces:

- `legacy/src/edge_specsim/controller.py`
  - `capacity_dpp`
  - interactivity deficit queue `Z_i`
  - server queue `Q_s`
  - device queue `Q_i`
  - exact enumeration of `gamma_choices`
  - censored acceptance learning
- `legacy/src/edge_specsim/verification_queue.py`
  - exact cardinality/token-budget knapsack
- `legacy/src/edge_specsim/network.py`
  - square-root allocation weight
- `legacy/src/edge_specsim/capacity.py`
  - candidate data structures
  - monotone N search
  - queue-tail slopes
  - step-area utilities
- `scripts/run_capacity_frontier.py`
  - nested candidate execution
  - per-policy sweeps
  - cache/resume
- `configs/capacity_frontier_base.yaml`
  - current experimental config

The following prototype behaviors must be fixed before the next expensive GPU sweep:

### 1.1 Wrong service-rate denominator

Current `scripts/run_capacity_frontier.py::measure_candidate()` computes

```python
start_ms = win["client_receive_ms"].min()
end_ms = win["client_receive_ms"].max()
t_meas_s = (end_ms - start_ms) / 1000
```

This excludes idle time at the beginning/end of the configured measurement interval and can inflate rates near the boundary. Replace this with the exact configured measurement duration.

### 1.2 Current feasibility classifier relaxes the SLO

`classify_feasibility()` currently uses a floor `x*(1-rate_tolerance_fraction)` and worst-seed/median heuristics. Final paper runs must classify the exact `x`, with uncertainty represented by confidence bounds and an `uncertain` state.

### 1.3 Gamma selection uses nominal uplink instead of the causal allocation signal

The current `capacity_dpp` device/network cost uses `client.uplink_mbps`. The paper algorithm uses a causal rate estimate `r_tilde_i` and then applies the square-root allocator to the realized payload. Add an explicit feedback path from the wireless allocator to the controller.

### 1.4 Proposed batching does not include the server queue price

The current `knapsack` value is essentially `(V+Z_i)*phi_i`. For Capacity-DPP, the batch subproblem must use

\[
v_i=(V+Z_i)\widehat\phi_i-Q_s\theta_f(\gamma_i+1).
\]

Do this without changing the semantics of legacy baseline schedulers.

### 1.5 Search result semantics at a cap are currently ambiguous

A feasible candidate at `n_search_cap` is only a lower bound. Persist this explicitly in result files.

### 1.6 Current output paths are mutable

`results/capacity_frontier/nstar_frontier.csv` has already been overwritten by runs with different configs. Introduce immutable run directories keyed by a run tag/config hash and make reports point to those directories.

---

# 2. Target architecture

Keep the framework in four layers:

```text
Capacity search
    |
    v
Candidate experiment (N, x, policy, seed)
    |
    v
Capacity-DPP runtime controller
    |-- gamma selection
    |-- verifier batch selection
    `-- wireless allocation
    |
    v
Fixed-window measurement
    |
    v
Feasibility classifier
    |
    v
Search-bracket update / boundary certification
```

The intended file split is:

```text
legacy/src/edge_specsim/controller.py
legacy/src/edge_specsim/verification_queue.py
legacy/src/edge_specsim/network.py
legacy/src/edge_specsim/metrics.py
legacy/src/edge_specsim/capacity.py
legacy/src/edge_specsim/simulator.py
scripts/run_capacity_frontier.py
configs/capacity_frontier_base.yaml
configs/capacity_frontier_final.yaml        # add
results/capacity_frontier/<run_tag>/        # add immutable layout
tests/test_capacity*.py                     # extend
```

Do not move unrelated code unless necessary.

---

# 3. Inner controller: Capacity-DPP

For a fixed `(N,x)`, the runtime objective is to satisfy

\[
\bar x_i\ge x,\qquad i=1,\ldots,N,
\]

while using residual capacity for useful-token goodput.

## 3.1 Interactivity deficit queue

Preserve/update

\[
Z_i(t+1)=\left[Z_i(t)+x\Delta-a_i(t)\right]^+.
\]

Requirements:

- `x` is the exact candidate requirement.
- `Delta` uses the controller slot duration in consistent units.
- Empty service slots must be accounted for, not skipped merely because no completion callback fires.
- Multiple completions in one slot must sum correctly.
- Add tests for long empty gaps and multiple completions in one slot.

## 3.2 Server resource queue

Use

\[
Q_s(t+1)=\left[Q_s(t)+\theta_f\Gamma_t-\Delta\right]^+,
\qquad
\Gamma_t=\sum_{i\in\mathcal B_t}(\gamma_i+1).
\]

Important distinction:

- `theta_f * Gamma_t` is the **analytical compute-work charge** used by the DPP price.
- Measured/profiled verifier latency is a separate quantity and must continue to be logged.
- Do not silently replace measured verifier latency with this linear model.

## 3.3 Device/network resource queue

Use the realized uplink rate when updating work:

\[
Q_i(t+1)=\left[
Q_i(t)+\gamma_i(t)\left(\tau_i^d+\frac{\kappa_i}{r_i(t)}\right)-\Delta
\right]^+.
\]

The queue update after a completed round already has access to the realized `uplink_rate_mbps`; use that value.

---

# 4. Causal gamma selection and bandwidth coupling

## 4.1 Required gamma score

For Capacity-DPP, enumerate all configured `gamma_choices` and score

\[
S_i(\gamma,t)
=(V+Z_i)\widehat\phi_i(\gamma,t)
-Q_s\theta_f(\gamma+1)
-Q_i\gamma\left(\tau_i^d+\frac{\kappa_i}{\widetilde r_i(t)}\right).
\]

Choose

\[
\gamma_i^\star(t)=\arg\max_{\gamma\in\Gamma_i}S_i(\gamma,t),
\]

breaking exact/nearly exact ties toward smaller `gamma`.

Do not use the roofline regime multiplier as an undocumented extra factor in this score. `theta_f` is the DPP compute-work price; the roofline remains the measured/profiled latency model.

## 4.2 Add an explicit causal uplink-rate signal

Add controller state such as

```python
self._last_uplink_rate_mbps: dict[int, float]
```

and methods

```python
observe_uplink_allocation(client_id: int, rate_mbps: float) -> None
estimated_uplink_rate_mbps(client: ClientProfile) -> float
```

Semantics:

- after `_allocate_shared_bandwidth()` returns the realized uplink rate for a speculative payload, call `observe_uplink_allocation()`;
- the next depth decision uses that last observed allocated rate as `r_tilde_i`;
- if a client has no prior realized allocation, fall back to a documented initialization, preferably its nominal link estimate or an equal-share estimate;
- never use future allocation information.

Log for every speculative round:

```text
gamma_rate_signal_mbps
realized_uplink_rate_mbps
gamma_rate_signal_age_rounds
```

This makes the causal approximation inspectable.

## 4.3 Scalar sanity condition

Keep a unit-tested scalar helper matching

\[
(V+Z_i)\alpha_i^{\gamma+1}
\ge
Q_s\theta_f+Q_i\left(\tau_i^d+\frac{\kappa_i}{\widetilde r_i}\right).
\]

For a scalar i.i.d. acceptance profile and an action set containing consecutive depths, the helper and exact enumeration should agree except at tie-breaking boundaries.

---

# 5. Acceptance learning

Use the existing censored positional estimator, but make the paper-facing semantics explicit.

For position `k`, an observation is valid only if the speculative execution reaches/inspects position `k`. Maintain separate opportunity and success counts per `(client, position)`.

Compute

\[
\alpha^{\rm UCB}_{i,k}(t)
=\min\{1,\widehat\alpha_{i,k}(t)+\rho_{i,k}(t)\}
\]

with the existing Bernstein-style radius (or the current equivalent if already implemented). Then

\[
\widehat\phi_i(\gamma,t)
=1+\sum_{k=1}^{\gamma}\prod_{j=1}^{k}\alpha^{\rm UCB}_{i,j}(t).
\]

Modes:

```text
capacity_dpp_ucb       # proposed learning controller
capacity_dpp_known     # same controller, static known/empirical acceptance
```

Backward compatibility: existing `capacity_dpp` may alias `capacity_dpp_ucb`, but result metadata must record the resolved mode.

Do not call `capacity_dpp_known` an oracle in paper-facing outputs.

Required logs:

```text
alpha_hat_position_1...
alpha_ucb_position_1...
acceptance_opportunities_position_1...
acceptance_successes_position_1...
```

If wide per-position columns are inconvenient, write a compact JSON field or a separate learning trace.

---

# 6. Verification batch selection

## 6.1 Capacity-DPP value

For a ready request `i` with verifier-token cost

\[
c_i^v=\gamma_i+1,
\]

use

\[
v_i=(V+Z_i)\widehat\phi_i(\gamma_i)-Q_s\theta_f c_i^v.
\]

The batch problem is

\[
\max_{\mathcal B\subseteq\mathcal R_t}\sum_{i\in\mathcal B}v_i
\]

subject to

\[
\sum_{i\in\mathcal B}c_i^v\le G_t,
\qquad
|\mathcal B|\le B_{\max}.
\]

## 6.2 Do not silently change legacy `knapsack`

Add a Capacity-DPP-specific path, for example:

```text
verification_batching.scheduler: capacity_knapsack
```

or add an explicit optional `capacity_value` field to `VerificationRequest` and use it only when the scheduler is `capacity_knapsack`.

Recommended request fields:

```python
@dataclass
class VerificationRequest:
    ...
    service_weight: float                  # V + Z_i
    expected_useful_tokens: ...            # existing property
    server_queue_price: float = 0.0
    theta_f_ms_per_token: float = 0.0

    @property
    def capacity_dpp_value(self) -> float:
        return (
            self.service_weight * self.expected_useful_tokens
            - self.server_queue_price * self.theta_f_ms_per_token * self.verifier_token_cost
        )
```

Do not reuse `diagnostic_priority_multiplier` in the proposed controller. That field remains diagnostic-only.

The exact DP knapsack must allow an empty batch mathematically when every value is non-positive. The server loop may need a separate liveness rule so that a request is not deadlocked forever; if a forced-service escape hatch is used, log it explicitly as `forced_service=true` and do not pretend it is the exact DPP optimum.

## 6.3 Required batch logs

```text
batch_scheduler
batch_size
batch_token_cost
batch_token_budget
batch_fill_ratio
sum_capacity_value
server_queue_before
server_queue_after
theta_f_ms_per_token
measured_verify_ms
modeled_verify_ms
roofline_regime
forced_service
```

---

# 7. Shared bandwidth allocation

Keep the current square-root solution for Capacity-DPP:

\[
b_i^\star
=W\frac{\sqrt{Q_i\gamma_i\kappa_i/s_i}}
{\sum_j\sqrt{Q_j\gamma_j\kappa_j/s_j}}.
\]

Requirements:

- use current channel/spectral-efficiency state;
- a client with `gamma=0` has zero speculative uplink weight;
- normalize robustly when every raw weight is zero;
- log both allocated bandwidth and realized rate;
- call `controller.observe_uplink_allocation()` after allocation.

Add a deterministic unit test that compares the implementation with the KKT formula for at least three clients with unequal `Q`, `gamma`, payload, and spectral efficiency.

---

# 8. Candidate experiment and exact measurement window

## 8.1 Make window boundaries explicit

`run_simulation.py` / simulator must expose the exact virtual-time boundaries used by the experiment:

```text
measurement_start_ms
measurement_end_ms
measurement_duration_s
```

The capacity driver must not infer these from event timestamps.

Preferred implementation:

- include the three fields in a small sidecar metadata JSON for every candidate run; or
- include constant `measurement_start_ms` and `measurement_end_ms` columns in the detailed trace.

Then `measure_candidate()` receives `measurement_duration_s` explicitly.

## 8.2 Sustained service rate

For every admitted client id `0..N-1`, compute useful committed tokens whose delivery timestamp lies inside the measurement interval, then divide by the exact configured duration:

```python
rates = compute_sustained_service_rates(
    measurement_rows,
    t_measurement_s=metadata.measurement_duration_s,
    admitted_client_ids=range(N),
)
```

If `measurement_rows` is empty, all rates are zero.

Required invariant test:

- measurement interval = 100 s;
- client gets tokens only from second 20 to second 80;
- denominator remains 100 s, never 60 s.

---

# 9. Final feasibility classifier

The final classifier must estimate the capacity definition, not a relaxed SLO.

## 9.1 Statistical object

For a fixed nested population and candidate `(N,x)`, let `r[i,s]` be client `i`'s sustained rate in independent seed `s`. The target is each client's expected long-run service rate.

For final runs, use **simultaneous per-client confidence bounds** across seeds. A concrete implementation that is simple and auditable is Bonferroni-adjusted Student-t bounds:

For client `i`, with `K>=2` seeds,

\[
\bar r_i=K^{-1}\sum_s r_{i,s},
\qquad
SE_i=s_i/\sqrt K.
\]

For overall confidence `1-delta`, define

\[
t_{\rm crit}=t_{1-\delta/(2N),K-1},
\]

and

\[
L_i=\bar r_i-t_{\rm crit}SE_i,
\qquad
U_i=\bar r_i+t_{\rm crit}SE_i.
\]

Clamp only the lower bound at zero if desired; never clamp it to the SLO.

Then

```text
FEASIBLE    iff min_i L_i >= x
INFEASIBLE  iff min_i U_i <  x
UNCERTAIN   otherwise
```

This directly encodes the exact requirement `x_i >= x`.

Add `scipy` only if already available in the environment. If avoiding a new dependency is important, implement the same interface with bootstrap-over-seeds, but document the method and add deterministic tests. Do not label worst-seed/median as a confidence interval.

## 9.2 Seed policy

Final configuration:

```yaml
capacity_search:
  confidence_delta: 0.05
  initial_seeds: 3
  max_seeds: 7
  min_final_seeds: 5
```

A candidate may begin with 3 seeds for search speed. Near the final certified boundary, force at least `min_final_seeds` even if the preliminary classifier is decisive.

A one-seed mode may remain for `--pilot`, but it must be labelled heuristic/sanity-only and must not produce headline ICA.

## 9.3 Queue-tail diagnostic

Estimate `Z_i` tail slopes over the latter half of the measurement interval using time, not row index.

Queue slopes are secondary diagnostics. They should trigger one of:

- extend measurement / add seeds if the rate CI is marginal and `Z_i` is still growing;
- mark a strong divergence as infeasible if the rate CI also does not support feasibility.

Do not reproduce the old behavior where a tiny transient positive `Z_i` slope vetoes a rate estimate that is comfortably above `x`.

Persist:

```text
max_z_slope
server_queue_slope
max_device_queue_slope
```

but only the service-rate CI is the primary exact-SLO classifier.

---

# 10. Capacity search

## 10.1 `N*(x)` search

For fixed `x`, implement:

1. obtain/verify a feasible lower point;
2. exponential probing to find an infeasible upper point or hit `N_cap`;
3. integer binary search while the bracket width exceeds 1;
4. refine uncertain candidates with more seeds/longer window before moving the bracket;
5. locally certify the boundary with at least `{N_hat-1, N_hat, N_hat+1}` where valid.

Return a structured result, not only an integer:

```python
@dataclass
class NStarResult:
    x_requirement: float
    last_confirmed_feasible_n: int
    first_confirmed_infeasible_n: int | None
    lower_bound_n: int
    upper_bound_n: int | None
    exact: bool
    hit_search_cap: bool
    uncertain_n: list[int]
    trace: list[SearchDecision]
```

Semantics:

- exact boundary when `first_infeasible = last_feasible + 1` and both are certified;
- if cap is feasible: `exact=False`, `lower_bound_n=N_cap`, `upper_bound_n=None`, `hit_search_cap=True`;
- if an unresolved uncertain point lies next to the boundary, return an interval.

## 10.2 `x*(N)` search

Add a second search axis using the **same** candidate runner and classifier:

```bash
python scripts/run_capacity_frontier.py --search-axis n ...   # current N*(x)
python scripts/run_capacity_frontier.py --search-axis x ...   # x*(N)
```

For fixed `N`, bisect an `[x_lo, x_hi]` feasible/infeasible bracket until

```text
x_hi - x_lo <= x_tolerance_tps
```

Return

```python
@dataclass
class XStarResult:
    n_active: int
    lower_feasible_x: float
    upper_infeasible_x: float
    estimate_x: float
    exact_within_tolerance: bool
    trace: list[SearchDecision]
```

The purpose is both presentation and cross-validation:

\[
N^\star(x)=\max\{N:x^\star(N)\ge x\}.
\]

## 10.3 Search monotonicity is theoretical, not a license to hide noisy violations

Do not apply post-hoc isotonic correction to raw policy results and present it as measurement. If local finite-run decisions violate monotonicity:

- log the violation;
- increase seeds/window around the conflicting points;
- report unresolved uncertainty if it remains.

---

# 11. Capacity area / ICA

For the common evaluation window `[x_L,x_U]` and `N_ref`, the area is

\[
A_C=\int_{x_L}^{x_U}\min\{N^\star(x),N_{\rm ref}\}\,dx.
\]

and

\[
\mathrm{ICA}=\frac{A_C}{N_{\rm ref}(x_U-x_L)}.
\]

Rules:

- the final measured/search grid must cover `x_L` and `x_U`;
- do not carry the first/last observed `N*` flat beyond the measured range;
- if a boundary is interval-valued or cap-limited, compute lower and upper area bounds;
- report `ICA_lower`, `ICA_upper`, and an estimate only when the estimate is justified;
- every compared policy uses identical `x_L`, `x_U`, `N_ref`, population, seeds policy, and search semantics.

Also implement the dual column-sum check from `x*(N)`:

\[
A_C
=\sum_{N=1}^{N_{\rm ref}}
[\min\{x^\star(N),x_U\}-x_L]^+.
\]

Do **not** trapezoidally integrate linearly interpolated `x*(N)` over fractional `N`.

---

# 12. Immutable output layout and provenance

Replace mutable headline outputs with

```text
results/capacity_frontier/<run_tag>/
  config.yaml
  config_hash.txt
  git_commit.txt
  environment.json
  population_fingerprint.txt
  candidate_cache.jsonl
  raw/
  feasibility_decisions_<policy>.csv
  nstar_frontier_<policy>.csv
  xstar_frontier_<policy>.csv
  capacity_area_<policy>.json
  policy_comparison.csv
  figures/
  RUN_REPORT.md
```

`run_tag` default:

```text
YYYYMMDD-HHMMSS_<config_hash>_<short_git_sha>
```

A resume operation must point to an existing run directory explicitly. Never discover a cache by silently using a global mutable path.

`environment.json` should include at least:

```text
python version
pytorch version
vllm version
CUDA/ROCm version
GPU model(s)
target model id
draft model id
hostname/cluster identifier if available
```

Every CSV/JSON should also contain `run_tag`, `config_hash`, `git_sha`, and `policy` where practical.

---

# 13. Final experiment configuration

Do not overwrite `configs/capacity_frontier_base.yaml`; keep it as historical/quick configuration.

Add `configs/capacity_frontier_final.yaml` with paper-oriented defaults. Start with:

```yaml
capacity_search:
  x_L: 2.0
  x_U: 6.0
  x_grid_final: [2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
  n_ref: 30
  n_search_cap: 30
  n_lo_feasible: 1
  confidence_delta: 0.05
  initial_seeds: 3
  max_seeds: 7
  min_final_seeds: 5
  warmup_seconds: 30
  measurement_seconds_final: 240
  max_measurement_seconds: 480
  x_tolerance_tps: 0.10
```

This is a starting point, not a sacred parameter choice. If runtime is prohibitive, preserve correctness first: reduce the number of non-boundary exploratory points before reducing final boundary seeds/window.

The final paper should include at least:

- homogeneous workload sanity;
- heterogeneous acceptance classes;
- heterogeneous device/network classes;
- the same full frontier for all baselines;
- `KnownAlpha` learning ablation;
- at least one verifier/batching ablation;
- capacity-vs-aggregate-goodput comparison.

---

# 14. Baselines

Preserve current baselines and run each through the same outer search:

```text
capacity_dpp_ucb / capacity_dpp
capacity_dpp_known
GoodSpeed
turboSpec / TurboSpec
FixedSLO
Gelato (if stable enough for final matrix)
target_only
fixed_gamma variants as ablations
```

Do not tune each baseline using a different `N_ref`, measurement window, population, or ICA range.

If a baseline cannot satisfy the common-floor semantics by design, that is an evaluation result, not a reason to change the classifier.

---

# 15. Mandatory tests before GPU Phase 5

Extend/add CPU tests so all of the following pass before expensive runs.

### Measurement

- fixed 100 s denominator despite activity only during a 60 s subinterval;
- zero-token admitted client appears with rate zero;
- all clients use identical denominator;
- warmup rows never contribute useful tokens to the measurement window.

### Deficit queues

- constant service exactly equal to `x` keeps `Z` bounded;
- service below `x` makes `Z` grow linearly;
- long empty gaps are accounted for;
- multiple completions in one slot aggregate correctly.

### Gamma control

- exact enumeration matches brute force over `gamma_choices`;
- tie goes to lower `gamma`;
- increasing `Q_s` cannot increase the chosen gamma in a controlled scalar test;
- decreasing causal uplink rate cannot increase gamma in a controlled scalar test;
- last observed uplink allocation is used, not `client.uplink_mbps`, after the first observation.

### Batch control

- exact `capacity_knapsack` matches brute force on small random instances;
- server queue penalty changes the selected batch in a constructed example;
- cardinality and token-budget constraints always hold;
- legacy `knapsack` behavior remains unchanged.

### Bandwidth

- square-root allocator matches the KKT formula;
- total allocated bandwidth never exceeds `W`;
- all-zero weights use a documented fallback without NaN/divide-by-zero.

### Statistical classifier

- a client whose upper confidence bound is below `x` yields infeasible;
- all client lower bounds above `x` yields feasible;
- overlapping bounds yield uncertain;
- classifier never compares to `(1-eps)*x`;
- `K=1` final-mode candidate cannot be reported as statistically certified.

### Search

- exact monotone synthetic boundary recovered;
- cap-feasible case returns `N* >= cap`, not equality;
- uncertain neighbor produces interval-valued boundary;
- local boundary certification invokes adjacent points;
- `x*(N)` bisection returns the expected bracket on a synthetic oracle.

### Area

- known staircase has identical row-slice and column-slice area;
- cap/uncertain points propagate area intervals;
- no extrapolation outside `[x_L,x_U]`;
- no trapezoidal interpolation over fractional concurrency.

---

# 16. GPU validation sequence

Do not immediately run the full final matrix. Use the following gates.

## Gate A — one candidate correctness

Run one moderate candidate, e.g. `(N=8, x=4)`, and manually inspect:

```text
fixed measurement denominator
gamma_rate_signal vs realized uplink rate
Z_i evolution
Q_s/Q_i evolution
capacity batch values
selected batch and token budget
measured verifier latency
per-client sustained rates
```

The run must produce a self-contained immutable directory.

## Gate B — local boundary at one x

At an `x` near the observed bend (start around 4--5 tok/s), search `N` and then explicitly re-run the local neighborhood. Confirm a pattern consistent with

```text
N*-1 : feasible
N*   : feasible
N*+1 : infeasible
```

within statistical uncertainty.

## Gate C — reproduce the TurboSpec signal

The previous compact one-seed run showed TurboSpec outperforming Capacity-DPP at `(N=14,x=3.5)` while using much shallower speculation and larger batches. Re-run this phenomenon with the corrected controller and >=3 seeds before drawing conclusions.

Log enough information to test the hypothesis:

\[
\text{deeper speculation}
\rightarrow \text{larger verifier-token cost}
\rightarrow \text{smaller effective batches}
\rightarrow \text{lower concurrency capacity}.
\]

If the effect persists, it is a result to explain, not something to hide.

## Gate D — small full frontier

Run Capacity-DPP and TurboSpec across a reduced but boundary-crossing grid with `n_search_cap=n_ref`. Only after this is sensible should the remaining baselines and heterogeneity matrix be launched.

---

# 17. Required result fields

Each candidate summary should contain at least:

```text
run_tag
config_hash
git_sha
policy
seed
population_fingerprint
N
x_requirement
measurement_duration_s
min_rate_tps
per_client_rates
min_rate_lcb
min_rate_ucb
feasibility_status
max_z_slope
server_queue_slope
max_device_queue_slope
goodput_tps
mean_gamma
p10_gamma / p90_gamma if easy
mean_batch_size
mean_batch_token_cost
mean_batch_fill_ratio
mean_measured_verify_ms
fraction_memory_bound
fraction_roofline_knee
fraction_compute_bound
mean_uplink_rate_mbps
mean_gamma_rate_signal_mbps
forced_service_fraction
```

Each `N*(x)` row should additionally include:

```text
last_confirmed_feasible_N
first_confirmed_infeasible_N
N_lower
N_upper
exact_boundary
hit_search_cap
boundary_certified
seeds_at_boundary
```

Each `x*(N)` row should include:

```text
N
x_lower_feasible
x_upper_infeasible
x_estimate
x_tolerance_tps
boundary_certified
```

---

# 18. Reporting and figures

The driver should be able to regenerate aggregate outputs from the immutable candidate cache without GPU execution.

Produce at least:

```text
N*(x) step plot with uncertainty/cap arrows
x*(N) plot with uncertainty bands
aggregate goodput vs x at the same boundary points
mean gamma vs x or N
mean batch size/fill vs x or N
roofline-regime fraction vs load
ICA comparison with lower/upper error interval
```

Do not create a headline plot that labels cap-limited points as exact `N*`.

The generated `RUN_REPORT.md` should state:

- exact config and git SHA;
- which points were exact, interval-valued, or cap-limited;
- any monotonicity conflicts and how they were resolved;
- whether area is exact/estimated/bounded;
- baseline comparison summary;
- warnings about pilot/sanity-only modes.

---

# 19. Code quality and backward compatibility

- Keep old result readers usable when possible.
- New dataclass fields should have defaults if old JSON cache loading depends on them.
- Version the candidate-cache schema, e.g. `schema_version: 2`.
- If an old cache lacks fixed-window metadata, reject it for final analysis rather than silently reusing biased rates.
- Preserve existing policy names or provide explicit aliases.
- No giant refactor in the same commit as the statistical/algorithmic correctness changes.
- Add docstrings that state units (`ms`, `s`, `Mbps`, `MHz`, tokens/s).
- Fail loudly on unit mismatches or missing required final-mode metadata.

---

# 20. Suggested implementation order

Implement in this order so failures are isolated:

1. immutable run directories + metadata schema;
2. exact measurement-window denominator + tests;
3. final exact-SLO confidence classifier + tests;
4. structured `NStarResult` and cap/uncertainty semantics;
5. `x*(N)` bisection path;
6. causal uplink-rate feedback into gamma scoring;
7. Capacity-DPP-specific queue-priced batch knapsack;
8. logging/provenance additions;
9. dual area/ICA bounds;
10. final config;
11. CPU test suite;
12. GPU Gates A--D;
13. only then launch the final baseline/heterogeneity matrix.

Make separate commits for the correctness-critical steps above. Do not combine all changes into one unreviewable commit.

---

# 21. Acceptance criteria

The implementation is complete only when all of the following are true:

- `measure_candidate()` uses the exact configured window duration;
- no final classifier contains a relaxed floor such as `(1-eps)*x`;
- Capacity-DPP gamma scoring uses a causal allocated-rate signal after the first observation;
- Capacity-DPP batch selection includes the `-Q_s theta_f (gamma+1)` price without changing legacy baseline schedulers;
- final-mode candidates use statistically interpretable confidence bounds and can remain uncertain;
- cap-limited capacity points are represented as lower bounds;
- both `N*(x)` and `x*(N)` can be reconstructed through the same candidate runner;
- area can be computed/bounded from either representation and the two agree within numerical/search tolerance;
- results live in immutable provenance-complete run directories;
- all mandatory CPU tests pass;
- a real-GPU boundary point is locally certified with multiple seeds;
- the corrected framework can reproduce or refute the previously observed Capacity-DPP vs TurboSpec batching/capacity effect.

Only after these criteria pass should the outputs be treated as paper-ready Phase-5 capacity measurements.
