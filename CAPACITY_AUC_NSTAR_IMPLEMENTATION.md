# Capacity-AUC / N* Implementation Specification for AUCspec

## 0. Purpose and scope

This document is the implementation specification for the next AUCspec direction targeted at a SIGMETRICS-style performance-modeling/system paper.

The previous primary frontier used aggregate useful-token goodput as the vertical axis:

\[
Y^\star(x)=\max_{\pi:\min_i x_i(\pi)\ge x}Y(\pi).
\]

In a work-conserving verifier, `Y*(x)` is often nearly constant until the per-user requirement becomes infeasible. The old area therefore becomes close to a rectangle and has low discriminative power.

The new primary object is the **maximum sustainable active concurrency** at a required per-user interactive service rate:

\[
\boxed{
N^\star(x;\xi)
=
\max\left\{N:\exists\pi\text{ such that }\bar x_i(\pi;\xi)\ge x,\;\forall i\le N\right\}
}
\]

for fixed deployment/configuration `xi`.

The corresponding capacity-area metric is

\[
A_N(\xi)=\int_{x_L}^{x_U}N^\star(x;\xi)\,dx.
\]

For cross-system comparisons, also report the normalized area

\[
\boxed{
\mathrm{ICA}(\xi)
=
\frac{1}{N_{\rm ref}(x_U-x_L)}
\int_{x_L}^{x_U}
\min\{N^\star(x;\xi),N_{\rm ref}\}\,dx
\in[0,1]
}
\]

where `x_L`, `x_U`, and `N_ref` are COMMON across every compared policy/system. `x_L` must be strictly positive unless an explicit finite `N_ref` cap is applied. Do not integrate an idealized `N*(x) ~ 1/x` to zero without a concurrency cap.

Interpretation: ICA is the fraction of the common `(interactivity requirement, offered concurrency)` rectangle that the system can sustain.

This document uses **N-star / capacity-AUC / ICA** as provisional names. Do not spend implementation time renaming the project yet.

---

## 1. Important conceptual separation

There are two distinct layers.

### Layer A: inner serving controller

For a FIXED candidate `(N, x)`, run the serving system and try to satisfy

\[
\bar x_i\ge x,\qquad i=1,\dots,N,
\]

while using resources efficiently. The controller retains the three key subproblems of the original AUCspec design:

1. per-client speculative draft-length selection `gamma_i`;
2. server verification-batch selection;
3. shared wireless-bandwidth allocation.

Unknown draft/target acceptance is learned online.

### Layer B: outer capacity search

The outer harness does NOT control per-round resources. It asks whether a candidate `(N,x)` is sustainable. Because supported concurrency is monotone under a nested population construction, it searches for the maximum feasible `N`.

Do **not** add dynamic admission control to the production controller in this phase. `N` is the offered active concurrency chosen by the capacity-search harness. This preserves the original three-subproblem algorithmic structure.

The result of the outer search is `N_hat_star(x)` for each requirement `x`.

---

## 2. Reuse the existing implementation; do not rewrite the simulator

Build on the current real-GPU asynchronous simulator under:

```text
legacy/src/edge_specsim/
```

Relevant existing code paths already contain most required mechanisms:

```text
legacy/src/edge_specsim/controller.py
    OnlineController
    paper_index_gamma(...)
    choose_gamma(...)
    observe_completion(...)
    observe_server_batch(...)
    observe_device_service(...)

legacy/src/edge_specsim/verification_queue.py
    VerificationRequest
    VerificationQueue.pop_batch(...)
    weighted_utility
    knapsack

legacy/src/edge_specsim/verification_batcher.py
    VerificationBatcher

legacy/src/edge_specsim/simulator.py
    EdgeSpecSimulator
    _make_clients(...)
    _allocate_shared_bandwidth(...)
    _bandwidth_allocation_score(...)

legacy/src/edge_specsim/network.py
    paper_square_root_allocation_weight(...)

legacy/src/edge_specsim/models.py
    ClientProfile
    censored / Bernstein acceptance learning
    positional_acceptance_profile(...)

legacy/src/edge_specsim/metrics.py
    per-sample/per-client/system metrics
```

The existing `adaptive_index` implementation is a useful reference but should NOT be silently changed. Add a new policy name, preferably:

```yaml
controller:
  policy: capacity_dpp
```

so old experiment results remain reproducible.

---

## 3. System model

### 3.1 Clients

There are `N` continuously active decoding streams. Client `i` has:

- a draft model/device;
- a prompt/current context;
- speculative depth `gamma_i(t)`;
- draft compute latency;
- uplink/downlink link state;
- target/draft acceptance process;
- a long-run committed useful-token service rate `x_i`.

The capacity requirement is a COMMON minimum interactive token rate:

\[
\bar x_i
=
\liminf_{T\to\infty}
\frac{1}{T}
\sum_{t<T}a_i(t)
\ge x.
\]

Here `a_i(t)` is the number of committed useful tokens delivered to client `i` in control slot `t` and the units of `x` are tokens/s/client.

### 3.2 Speculative decoding reward

For an i.i.d. scalar acceptance approximation `alpha_i`, expected useful committed tokens from speculation depth `gamma` are

\[
\phi(\gamma,\alpha_i)
=
\frac{1-\alpha_i^{\gamma+1}}{1-\alpha_i}.
\]

For positional acceptance probabilities `alpha_{i,1},...,alpha_{i,gamma}`, use

\[
\phi_i(\gamma)
=
1+\sum_{k=1}^{\gamma}
\prod_{j=1}^{k}\alpha_{i,j}.
\]

The implementation already supports positional profiles through `ClientProfile.positional_acceptance_profile()` and `expected_useful_tokens_from_profile()`; the new controller should use these when available.

### 3.3 Verifier cost

Keep the existing measured/profiled verifier implementation. For analysis and queue accounting, use the roofline abstraction

\[
T^v(\mathcal B)
=
\theta_0+
\max\left\{
T_{mem}(\mathcal B),
\theta_f\Gamma(\mathcal B)
\right\},
\]

where

\[
\Gamma(\mathcal B)=\sum_{i\in\mathcal B}(\gamma_i+1)
\]

is verification-token work.

Do not call `theta_0` network RTT. It is target/verifier overhead. Network RTT remains a client/network quantity.

### 3.4 Shared wireless

For the analytical allocator use the same linearized bandwidth-rate model already implicit in the square-root allocator:

\[
r_i=b_i s_i,
\]

where `b_i` is allocated MHz and `s_i` is current spectral efficiency in bit/s/Hz. Total bandwidth satisfies

\[
\sum_i b_i\le W.
\]

The existing channel/SNR model and actual payload-size functions should be reused.

---

## 4. Nested population construction — REQUIRED for valid N* search

Naively calling `_make_clients(N)` independently for every `N` can change the random client composition. Then `N=18` and `N=21` are not nested systems and feasibility need not be monotone.

Implement a **fixed maximum population catalog per seed**.

### 4.1 Required behavior

For a capacity-search seed:

1. construct a deterministic population of `N_ref` or `N_max` client profiles once;
2. freeze client class, dataset assignment, prompt shard, channel parameters, draft-speed parameters, etc.;
3. candidate concurrency `N` uses a prefix/subset of that SAME catalog;
4. increasing concurrency only ADDS clients; it never resamples or replaces existing clients.

### 4.2 Preserve workload mix

For heterogeneous experiments, use a population template / class mix. Example for three equal groups:

```text
q = (1 easy_gsm8k, 1 medium_cnn, 1 hard_math)
N = m * 3
```

Search over integer scale factor `m`, not arbitrary `N`, so every candidate preserves exactly the same class proportions.

General definition:

\[
\mathbf N(m)=m\mathbf q,
\qquad
N(m)=m\sum_c q_c.
\]

Then define

\[
m^\star(x)=\max\{m:\mathbf N(m)\text{ is feasible at }x\}
\]

and report

\[
N^\star(x)=m^\star(x)\sum_c q_c.
\]

Homogeneous experiments simply use `q=(1)`.

### 4.3 Implementation suggestion

Do not mutate global random state differently for each candidate. Add a reusable population-spec generator, e.g.

```text
legacy/src/edge_specsim/population.py
```

or equivalent helpers inside the existing simulator. Persist enough metadata to prove candidate populations are nested.

Log a `population_fingerprint` and per-client stable ID/seed for every run.

---

## 5. Inner problem at fixed (N, x)

For each candidate concurrency and interactivity requirement, the inner controller solves the operational problem

\[
\max_{\pi}\;\bar Y(\pi)
\]

subject to

\[
\bar x_i(\pi)\ge x,\quad\forall i,
\]

plus server, draft-device, and wireless constraints.

Goodput is a SECONDARY efficiency objective inside the feasibility region. The paper-level capacity metric is `N*(x)`, not `Y*(x)`.

Keep `V` as the drift-plus-penalty trade-off parameter. Do not sweep `V` to trace the capacity frontier. Main capacity-frontier experiments sweep/search `N` at fixed `V` for each `x`.

---

## 6. Virtual queues and drift accounting

Use the existing control slot duration `Delta_s = slot_ms / 1000` seconds.

### 6.1 Interactivity-deficit queue

For each active client:

\[
\boxed{
Z_i(t+1)=\left[Z_i(t)+x\Delta_s-a_i(t)\right]^+
}
\]

If `Z_i(t)` is rate-stable, the long-run average service constraint is satisfied.

The current `observe_completion()` already reconstructs missed empty slots when a completion arrives. Preserve that accounting, but add tests specifically for long gaps and multiple completions in one slot.

### 6.2 Server resource queue

Use the existing server queue abstraction:

\[
\boxed{
Q_s(t+1)
=
\left[Q_s(t)+\theta_f\Gamma_t-\Delta_{ms}\right]^+
}
\]

with consistent time units. Existing implementation uses `slot_ms` and `theta_f_ms_per_token`.

Measured verifier service time should be logged separately from analytical `theta_f * Gamma`; do not silently substitute one for the other.

### 6.3 Per-device/network queue

For client/device `i`:

\[
\boxed{
Q_i(t+1)
=
\left[
Q_i(t)
+\gamma_i(t)\left(\tau_i^d+\frac{\kappa_i}{r_i(t)}\right)
-\Delta_{ms}
\right]^+
}
\]

Reuse `observe_device_service()` and the existing marginal uplink payload calculation.

### 6.4 Drift-plus-penalty weight

The one-slot idealized objective has the form

\[
\max
\left\{
\sum_i (V+Z_i)\,\mathbb E[a_i]
-Q_s c_s
-\sum_i Q_i c_i
\right\}.
\]

This should lead to the same three-way decomposition below.

Do not introduce hand-written fairness multipliers in the proposed controller unless they can be derived as a queue scaling. The previous diagnostic `diagnostic_priority_multiplier` remains diagnostic-only.

---

## 7. Subproblem 1 — speculative draft-length selection

### 7.1 Proposed implementation

For `capacity_dpp`, use exact enumeration over the configured discrete action set `gamma_choices`. The set is small, so `O(|Gamma|)` per client is preferable to relying only on an i.i.d.-alpha closed form.

For client `i`, given current queue prices and current estimated acceptance profile, score

\[
\boxed{
S_i(\gamma)
=(V+Z_i)\hat\phi_i(\gamma)
-Q_s\theta_f(\gamma+1)
-Q_i\gamma\left(\tau_i^d+\frac{\kappa_i}{r_i}\right)
}
\]

and choose

\[
\gamma_i^\star
=\arg\max_{\gamma\in\Gamma}S_i(\gamma).
\]

Tie-break toward smaller `gamma` unless there is a clear reason to keep the current behavior. Smaller-gamma tie-breaking avoids gratuitous verifier work at equal utility.

Use UCB positional acceptance for the learning controller; use static empirical/true acceptance for KnownAlpha mode.

### 7.2 Closed-form special case

Keep `paper_index_gamma()` as a fast special-case/reference. For scalar i.i.d. acceptance, the marginal benefit of one more speculative token is

\[
(V+Z_i)\alpha_i^{\gamma+1},
\]

and marginal cost is approximately

\[
Q_s\theta_f
+Q_i\left(\tau_i^d+\frac{\kappa_i}{r_i}\right).
\]

Thus add one more speculative token while

\[
(V+Z_i)\alpha_i^{\gamma+1}
\ge
Q_s\theta_f
+Q_i\left(\tau_i^d+\frac{\kappa_i}{r_i}\right).
\]

Unit-test that discrete enumeration and `paper_index_gamma()` agree under the scalar model when their assumptions match.

### 7.3 Required logs for every decision

Log:

```text
client_id
slot/round_id
x_requirement
N_active
Z_i
Q_i
Q_server
alpha_hat
alpha_ucb
positional alpha_hat/ucb if available
gamma_choices
score for each gamma (detailed/debug mode)
selected_gamma
expected_useful_tokens(selected_gamma)
estimated draft cost
estimated uplink cost
estimated verifier marginal cost
exploration flag/reason
```

---

## 8. Subproblem 2 — verification batch selection

### 8.1 Ready-request set

At a target scheduling opportunity, let `R_t` be requests eligible under the batch waiting rule. Do not return to the old `fcfs` default for the proposed controller.

Use the existing batch-wait mechanism but log whether a batch launches because of timeout, token budget, max batch size, or another reason. Add `launch_reason` if it is still missing.

### 8.2 Exact knapsack for the proposed controller

For each ready request, define

\[
v_i=(V+Z_i)\hat\phi_i(\gamma_i),
\qquad
c_i=\gamma_i+1.
\]

Select

\[
\boxed{
\max_{\mathcal B\subseteq\mathcal R_t}
\sum_{i\in\mathcal B}v_i
}
\]

subject to

\[
\sum_{i\in\mathcal B}c_i\le G_t,
\qquad
|\mathcal B|\le B_{max}.
\]

Use the existing `knapsack` implementation in `verification_queue.py` unless inspection shows a mismatch with this value/cost definition. For the proposed controller, prefer exact knapsack when the configured token budget is small enough. Keep `weighted_utility` as an approximation/ablation.

Important: `B_t = |B_t|` is the ACTUAL verification batch size. Do not call total active concurrency `B`; call it `N` or `N_active`.

### 8.3 Prevent the previous low-alpha diagnostic confusion

Low-acceptance clients naturally have smaller `phi_i`, but if they miss the interactivity floor then `Z_i` must grow and eventually dominate the scheduling value. Do not compensate using a static manual class multiplier in the proposed algorithm.

The implementation must make it possible to verify this mechanism directly:

\[
Z_i\uparrow
\Rightarrow
v_i=(V+Z_i)\phi_i\uparrow
\Rightarrow
\text{batch inclusion probability should rise}.
\]

Add a diagnostic plot of batch-inclusion probability vs `Z_i` quantile and acceptance quantile.

### 8.4 Required batch logs

For every batch log:

```text
batch_id
batch_ready_ms
batch_start_ms
verify_finish_ms
launch_reason
eligible_count
selected_client_ids
selected_gamma list
selected_Z list
selected_weight/value list
selected verifier-token costs
Gamma_batch
verify_token_budget
fill_ratio
batch_size
mean/max context length
scheduler_name
modeled verifier service ms
measured verifier service ms
compute-bound proxy ms
theta_f_ms_per_token
verifier regime
```

Also log rejected/non-selected eligible requests in debug mode with reason/value/cost.

---

## 9. Subproblem 3 — shared uplink bandwidth allocation

### 9.1 Analytical subproblem

For active uploading clients, approximate

\[
r_i=b_i s_i
\]

and minimize queue-weighted communication time

\[
\min_{b_i\ge0}
\sum_i
Q_i\frac{\gamma_i\kappa_i}{b_i s_i}
\]

subject to

\[
\sum_i b_i\le W.
\]

KKT gives the square-root allocation

\[
\boxed{
b_i^\star
=
W\frac{
\sqrt{Q_i\gamma_i\kappa_i/s_i}
}{
\sum_j\sqrt{Q_j\gamma_j\kappa_j/s_j}
}
}
\]

for positive active weights.

The existing `paper_square_root_allocation_weight()` already computes the unnormalized square-root term. Reuse it.

### 9.2 Zero-weight handling

If all analytical weights are zero, fall back to equal allocation over active transmitters. Do not let all clients receive zero bandwidth.

If only some weights are zero, make starvation behavior explicit. Preferred implementation is either:

- a tiny configured bandwidth floor `b_min` followed by square-root allocation of the residual bandwidth; or
- a documented epsilon in the queue price.

Whichever is used must preserve `sum_i b_i <= W` and be tested.

### 9.3 Downlink

The paper's main analytical resource-allocation result may focus on uplink because speculative payload grows with `gamma`. Keep existing downlink allocator for the implementation and log downlink behavior, but do not invent a theorem for it unless derived.

### 9.4 Required wireless logs

Per transfer/client:

```text
N_active
client_id
direction
allocated_bandwidth_mhz
total_bandwidth_mhz
snr_db
spectral_efficiency
rate_mbps
payload_bytes
transfer_time_ms
Q_i
Z_i
gamma_i
allocator_name
square_root_raw_weight
```

---

## 10. Acceptance learning

Do not replace the current censored-learning implementation.

The repository already maintains:

- scalar `alpha_hat` / `alpha_ucb`;
- positional estimates/UCBs;
- right-censored rounds;
- discounted and sliding-window modes.

For the primary proposed controller use `bernstein_censored` or the current canonical censored-Bernstein mode.

### 10.1 Proposed variants

Every capacity experiment should distinguish:

```text
Capacity-UCB
    capacity_dpp + online censored-UCB acceptance estimates

Capacity-KnownAlpha
    same controller/action logic, but use a static acceptance profile estimated offline
    from a disjoint calibration trace or known simulation truth
```

Do NOT call `Capacity-KnownAlpha` an exact oracle. It only removes the learning error.

For small analytical/simulation cases, optionally implement `OfflineOracle` / exhaustive search separately to estimate the true feasible boundary.

### 10.2 Learning measurements

Log and summarize:

```text
observed proposal positions
censored rounds
alpha_hat / alpha_ucb by client and position
acceptance calibration error when offline truth is available
UCB coverage rate
selected gamma vs KnownAlpha selected gamma
capacity gap N*_KnownAlpha - N*_UCB
```

---

## 11. Empirical service-rate definition — IMPORTANT

The capacity metric requires a **sustained per-active-client service rate over a common measurement window**.

Do not rely only on existing per-sample `active_e2e_ms` denominators.

Add an explicit per-client sustained rate:

\[
\boxed{
\hat x_i
=
\frac{\text{useful committed tokens delivered to client i during measurement}}
{T_{measurement}}
}
\]

where all admitted clients use the SAME `T_measurement` denominator.

If an admitted client gets no tokens, its rate is zero. Do not drop it from the minimum.

Primary empirical interactivity metric:

\[
\hat x_{min}=\min_{i\le N}\hat x_i.
\]

Keep existing sample-level TTFT, TPOT/inter-token latency, completion, and fairness metrics as secondary diagnostics.

Add these to `metrics.py` without breaking the old metric names.

Suggested new fields:

```text
sustained_service_rate_tps per client
min_sustained_service_rate_tps
p10_sustained_service_rate_tps
median_sustained_service_rate_tps
mean_sustained_service_rate_tps
max_sustained_service_rate_tps
```

---

## 12. Feasibility oracle for a candidate (N, x)

The theoretical criterion is virtual-queue rate stability plus the long-run rate constraint. A finite GPU run needs an explicit empirical classifier.

Implement:

```text
class FeasibilityResult:
    status: feasible | infeasible | uncertain
    min_rate_tps
    rate_margin_tps
    max_z_slope
    max_device_queue_slope
    server_queue_slope
    seeds_used
    reason
```

### 12.1 Per-seed measurements

After warmup, for each seed compute:

\[
m_s=\hat x_{min,s}-x.
\]

Also estimate a linear slope over the last half of the measurement interval for:

- each `Z_i(t)`;
- server queue `Q_s(t)`;
- device queues `Q_i(t)`.

Normalize queue slope by the relevant slot time so runs of different duration are comparable.

### 12.2 Classification

Use configurable tolerances, for example:

```yaml
capacity_search:
  rate_tolerance_fraction: 0.03
  queue_slope_tolerance: ...
  initial_seeds: 3
  max_seeds: 5
```

Suggested logic:

**Feasible** if:

- the seed-aggregated lower confidence bound / conservative estimate of `min_rate` is at least `x * (1 - tol)`; and
- no resource/deficit queue has a clearly positive persistent tail slope.

**Infeasible** if:

- the upper confidence estimate of `min_rate` is below `x * (1 - tol)`; or
- deficit queues show clear sustained positive growth.

**Uncertain** if the point lies inside the statistical/tolerance band. For uncertain points, automatically add seeds up to `max_seeds` and/or extend measurement duration before classifying.

Do not force a binary answer from a noisy 3-seed point exactly on the boundary.

Persist the classification evidence.

---

## 13. Outer monotone search for N*(x)

### 13.1 Search variable

For homogeneous workloads search integer `N`.

For heterogeneous fixed-mix workloads search population scale `m`, then convert to total `N`.

### 13.2 Binary search

For one requirement `x`:

```text
lo = known feasible population scale
hi = known infeasible scale or configured maximum
while hi - lo > 1:
    mid = floor((lo + hi)/2)
    result = feasibility_oracle(mid, x)
    if feasible:
        lo = mid
    elif infeasible:
        hi = mid
    else:
        refine the same point; do not move the bracket yet
return lo
```

If no initial infeasible `hi` is known, use exponential bracketing:

```text
1, 2, 4, 8, ...
```

in population-scale units until infeasible or `N_ref` is reached.

### 13.3 Reuse monotonicity across x

Search the `x` grid in increasing order. Since

\[
x_2>x_1\Rightarrow N^\star(x_2)\le N^\star(x_1),
\]

the previous `N_hat_star(x_{k-1})` is the new upper bound for `x_k`.

This substantially reduces GPU runs.

### 13.4 Cache/resume

Every `(x, N, policy, seed, config hash)` run must be cacheable. The driver must resume after interruption and never rerun completed expensive GPU points unless `--force` is passed.

Suggested driver:

```text
scripts/run_capacity_frontier.py
```

with modes:

```text
--pilot
--search
--resume
--analyze-only
```

---

## 14. Capacity area / ICA computation

Because `N*(x)` is integer-valued and nonincreasing, do not blindly spline it.

For grid

\[
x_1<x_2<\cdots<x_K,
\]

report monotonic Riemann bounds:

\[
A_{lower}
=
\sum_{k=1}^{K-1}
N^\star(x_{k+1})(x_{k+1}-x_k),
\]

\[
A_{upper}
=
\sum_{k=1}^{K-1}
N^\star(x_k)(x_{k+1}-x_k).
\]

A step-envelope estimate can be used as the headline, but store both bounds. Do not use trapezoidal interpolation as the only result for an integer capacity frontier.

For stochastic runs, bootstrap the WHOLE pipeline:

```text
resample seeds -> classify candidate points -> reconstruct N*(x) -> integrate area
```

rather than putting independent error bars on already-maximized points.

Report raw area `A_N` and normalized `ICA`.

---

## 15. Symmetric analytical sanity model

Implement a small CPU-only analytical model for correctness and theorem validation.

For `N` identical clients in a gated/synchronous approximation:

\[
x(N,\gamma)
=
\frac{\phi(\gamma,\alpha)}
{T_d(\gamma)+T_{net}(\gamma)+T_v(N,\gamma)}.
\]

In the compute-bound verifier regime:

\[
T_v(N,\gamma)
\approx
\theta_0+\theta_fN(\gamma+1).
\]

Let `T0(gamma)` collect draft/network/non-amortized terms. Feasibility requires

\[
x\le
\frac{\phi(\gamma,\alpha)}
{T_0(\gamma)+\theta_fN(\gamma+1)}.
\]

Therefore

\[
\boxed{
N^\star(x)
\approx
\max_{\gamma\in\Gamma}
\left\lfloor
\frac{\phi(\gamma,\alpha)/x-T_0(\gamma)}
{\theta_f(\gamma+1)}
\right\rfloor_+
}
\]

under these assumptions.

The code should demonstrate the expected high-load scaling

\[
N^\star(x)=\Theta(1/x)
\]

rather than the old aggregate-goodput rectangle.

Suggested file:

```text
sim/analysis/nstar_symmetric.py
```

or a location consistent with the repository's current analysis structure.

Do not claim this closed form for the asynchronous real system; it is a structural sanity model.

---

## 16. Proposed configuration schema

Create a canonical config, e.g.

```text
configs/capacity_frontier_base.yaml
```

with explicit sections similar to:

```yaml
controller:
  policy: capacity_dpp
  V: 50.0
  gamma_choices: [0, 1, 2, 3, 4]
  min_interactivity_tps: 1.0   # overridden by capacity-search driver
  use_virtual_queues: true
  learning:
    mode: bernstein_censored
    confidence_delta: 0.05

verification_batching:
  scheduler: knapsack
  max_batch_size: 32
  batch_wait_ms: 150.0
  budget_mode: fixed           # or roofline_knee in a separate experiment
  verify_token_budget: 10

clients:
  shared_wireless:
    total_uplink_bandwidth_mhz: 40.0
    total_downlink_bandwidth_mhz: 80.0
    uplink_allocator: square-root
    downlink_allocator: queue-weighted

capacity_search:
  x_grid: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
  N_ref: 60
  population_template: [1]    # homogeneous default; [1,1,1] for 3-class equal mix
  initial_seeds: 3
  max_seeds: 5
  rate_tolerance_fraction: 0.03
  warmup_seconds: 30
  measurement_seconds_pilot: 60
  measurement_seconds_final: 300
  resume: true
```

The numeric values above are starting defaults, not sacred constants. The implementation must allow overrides from CLI/YAML. If the current hardware cannot support `N_ref=60`, use a lower common cap and record it.

---

## 17. Experiment plan

### Exp 0 — correctness and calibration

No headline result before this passes.

1. Exact greedy speculative verification returns the same target output as target-only greedy decoding.
2. `phi(gamma,alpha)` / positional expectation functions match Monte Carlo simulation.
3. discrete `capacity_dpp` gamma enumeration matches brute force by construction.
4. scalar special case matches `paper_index_gamma()` under matching assumptions.
5. knapsack batch selection matches brute force on tiny request sets.
6. square-root bandwidth allocation satisfies KKT and total-bandwidth constraints.
7. virtual queues update correctly across empty slots and multiple completions.
8. population candidates are nested and keep workload mix exactly fixed.
9. sustained per-client rate uses a common measurement denominator.
10. symmetric CPU model recovers `N*(x) ~ 1/x` in compute-bound conditions.

### Exp 1 — main N*(x) capacity frontier

Run the proposed controller at fixed deployment and search `N*(x)` over a common `x` grid.

Primary plot:

```text
x-axis: required interactivity x (tok/s/client)
y-axis: maximum sustainable concurrency N*(x)
```

Also show normalized ICA.

Start homogeneous (e.g. GSM8K) for clean interpretation, then use a fixed heterogeneous workload mix.

### Exp 2 — component ablations

Trace `N*(x)` and ICA for:

```text
Full capacity_dpp
- adaptive gamma (fixed gamma)
- knapsack batching (FCFS or max-throughput baseline)
- square-root bandwidth (equal bandwidth)
- learning (KnownAlpha vs UCB)
```

This experiment should show whether all three internal subproblems matter to capacity.

### Exp 3 — capacity regimes / roofline

Vary offered concurrency and verifier budget/profile to move through:

```text
underloaded / memory-bound
roofline knee
compute-bound / saturated
```

Measure how:

```text
N*(x)
gamma distribution
mean batch size
Gamma_batch
verifier utilization
useful/verified-token efficiency
```

change across regimes.

### Exp 4 — heterogeneity

Use fixed population templates with measured acceptance/task heterogeneity and network/device heterogeneity. Do not resample class composition between candidate `N` values.

### Exp 5 — V scaling

At several fixed interior `(N,x)` points, sweep `V` to validate the expected efficiency/backlog trade-off. `V` is NOT the main frontier axis.

### Exp 6 — old metric vs new metric motivation

For the same deployment show side-by-side:

```text
old: Y_controller or Y_hat_star vs x, often near horizontal
new: N*(x) vs x, expected to expose scalability/interactivity trade-off
```

This is a motivation figure, not a proof that every system must have a curved `N*(x)`.

---

## 18. Baselines

Every baseline must be evaluated through the SAME outer capacity-search procedure; do not compare one proposed frontier against one baseline operating point.

Initial policies already available in the repo include:

```text
TargetOnly
FixedGamma / BestFixed
LoadOnly
GoodSpeed
TurboSpec
Fixed-SLO heuristic
GELATO-style heuristic
```

For each policy `p`, measure

\[
N_p^\star(x)
=
\max\{N:\text{policy p sustains requirement x}\}.
\]

Then compute `ICA_p` on the exact same `(x_L,x_U,N_ref)` domain.

Keep baseline naming faithful to what is actually implemented. Do not call simplified heuristics exact reproductions of external systems if they are not.

---

## 19. Mandatory raw logging

The current per-round CSV already logs many useful fields. Extend it so every capacity run can reconstruct decisions without rerunning GPUs.

At minimum every round/request record must include:

```text
run_id
policy
seed
config_hash
population_fingerprint
N_active
x_requirement
client_id
client_class/dataset
round_id
slot_id
sample_id
Z_i
Q_i
Q_server
gamma
alpha_hat
alpha_ucb
expected_useful_tokens
proposed_tokens
accepted_length
useful_tokens
committed_token_count
draft_start/finish
draft_queue_wait_ms
draft_service_ms
uplink bandwidth/rate/payload/time
RTT
server_arrival_ms
batch_id
batch_wait_ms
batch_size
Gamma_batch
verify_token_budget
fill_ratio
batch scheduler
batch launch reason
batch ready/start/finish
modeled verifier service ms
measured verifier service ms
downlink bandwidth/rate/payload/time
client_receive_ms
round_latency_ms
virtual_system_time_ms
in_measurement_window
```

Do not commit multi-GB raw trace dumps to git. Keep large raw logs gitignored and commit compact aggregate CSV/JSON reports.

---

## 20. Aggregate outputs

Create at least:

```text
results/capacity_frontier/
    candidate_points_raw.csv
    candidate_points_summary.csv
    feasibility_decisions.csv
    nstar_frontier.csv
    ica_summary.json
    queue_stability_summary.csv
    policy_comparison.csv
    fig1_nstar_frontier.png
    fig2_ica_comparison.png
    fig3_gamma_vs_capacity.png
    fig4_batching_regime.png
    fig5_queue_stability.png
    fig6_old_vs_new_frontier.png
```

Suggested `nstar_frontier.csv` columns:

```text
policy
x_requirement
N_hat_star
N_lower_conf
N_upper_conf
boundary_last_feasible_N
boundary_first_infeasible_N
seeds
min_rate_at_last_feasible
queue_stability_at_last_feasible
Y_at_last_feasible
mean_gamma
mean_batch_size
mean_fill_ratio
server_utilization
useful_per_verify_token
```

Suggested `ica_summary.json`:

```json
{
  "x_L": 0.5,
  "x_U": 4.0,
  "N_ref": 60,
  "area_lower": 0.0,
  "area_upper": 0.0,
  "area_step_estimate": 0.0,
  "ICA": 0.0,
  "bootstrap_ci_95": [0.0, 0.0]
}
```

---

## 21. Required figures and what they must answer

### Figure 1 — N*(x) frontier

Does stricter per-user interactivity reduce sustainable concurrency? Show proposed + main baselines with uncertainty.

### Figure 2 — ICA comparison

Does the proposed controller support a larger fraction of the common interactivity-concurrency region?

### Figure 3 — gamma mechanism

At the last feasible boundary for each `x`, plot gamma distribution/mean. Does optimal speculation become shallower as load/interactivity pressure changes?

### Figure 4 — verifier mechanism

Show batch size, Gamma, fill ratio, measured service time, and roofline regime along the frontier.

### Figure 5 — queue stability

Compare an adjacent feasible and infeasible concurrency at the same `x`. Feasible queues should stabilize; the bottleneck deficit/resource queue should grow at the infeasible point.

### Figure 6 — why old AUC was degenerate

For identical configurations, show the old aggregate-goodput-vs-x view and the new concurrency-vs-x view. The point is to demonstrate that work conservation can hide scalability structure in aggregate goodput.

---

## 22. Unit/integration tests Codex must add

Add tests for at least:

```text
1. nested population construction
2. fixed class proportions for template scaling
3. N monotonic search bracket logic
4. cache/resume key correctness
5. feasibility classifier: feasible/infeasible/uncertain synthetic cases
6. sustained-rate metric with zero-service client
7. queue-tail slope estimator
8. capacity_dpp gamma score exhaustive argmax
9. scalar closed-form gamma equivalence
10. knapsack exactness vs brute force
11. bandwidth allocation sum <= W and KKT ratios
12. ICA lower/upper step integration
13. normalized ICA in [0,1]
14. symmetric N*(x) monotonicity and ~1/x scaling
```

Do not unit-test a theorem by asserting an asymptotic bound on one stochastic GPU run.

---

## 23. Theoretical properties the implementation should support measuring

The code does not need to write the paper proof, but it must expose measurements needed to validate these claims.

### Property P1 — monotonicity

\[
x_2>x_1
\Rightarrow
N^\star(x_2)\le N^\star(x_1).
\]

### Property P2 — queue-based feasibility

For a feasible `(N,x)` in the interior, interactivity deficit queues should be rate-stable and empirical service should satisfy the floor.

### Property P3 — three-way decomposition

One-slot drift decisions decompose into:

```text
gamma selection
verification batch knapsack
wireless square-root allocation
```

coordinated by queue prices.

### Property P4 — symmetric capacity law

In a compute-bound symmetric regime, `N*(x)` should follow the derived inverse-in-x form and exhibit changes in `gamma*` across regimes.

### Property P5 — learning cost

`Capacity-UCB` should approach `Capacity-KnownAlpha`; report the gap in sustainable concurrency/ICA, not only token regret.

---

## 24. Implementation order

Codex should implement in this order and commit incrementally.

### Phase 1 — metrics/population/search plumbing

1. add sustained per-client rate metric;
2. add deterministic nested population catalog/template scaling;
3. implement feasibility result + queue slope diagnostics;
4. implement cached outer N search;
5. implement step-area / ICA computation;
6. add CPU-only tests.

At the end of Phase 1, use an existing controller as a smoke test. Do not run the expensive final sweep yet.

### Phase 2 — proposed capacity_dpp controller

1. add new `capacity_dpp` policy without changing `adaptive_index`;
2. exact discrete gamma score using positional UCB profile;
3. proposed server scheduler uses exact knapsack;
4. ensure server/device queue prices feed the gamma decision;
5. square-root uplink allocation with explicit zero-weight behavior;
6. add detailed decision logs and tests.

### Phase 3 — analytical sanity model

Implement symmetric CPU model and compare numerical `N*(x)` against closed form in the compute-bound special case.

### Phase 4 — pilot GPU frontier

Run a small `x` grid, short measurement windows, and low `N_ref` to validate:

- search monotonicity;
- queue classifier;
- nested populations;
- no OOM/deadlock;
- meaningful `N*(x)` variation.

### Phase 5 — final experiment harness

Only after pilot results are sane, run longer measurement windows, all seeds, main baselines, and ablations.

---

## 25. Non-goals / avoid these mistakes

1. **Do not sweep V to create the N*(x) frontier.** Sweep/search concurrency `N`; keep V fixed in the headline frontier.
2. **Do not dynamically drop hard clients** to satisfy `min_i x_i >= x`. All clients in the candidate population count.
3. **Do not resample the population when N changes.** Candidates must be nested.
4. **Do not call `B` both concurrency and batch size.** Use `N_active` and `B_t` separately.
5. **Do not call KnownAlpha an exact oracle.**
6. **Do not use a single baseline point to compute an area.** Every baseline needs its own N*(x) frontier.
7. **Do not integrate to x=0 without the explicit N_ref cap.**
8. **Do not use trapezoidal interpolation as the only area estimator** for an integer monotone capacity curve.
9. **Do not remove the real Qwen/vLLM path.** Add capacity-search tooling around the working system.
10. **Do not hide flat or rectangular results.** If N*(x) is also flat in some range, report and explain the physical regime.
11. **Do not tune parameters separately using the test seed.** Any BestFixed/baseline tuning must use a separate validation set/trace.
12. **Do not commit huge raw logs.** Commit aggregate evidence and reproducible configs/scripts.

---

## 26. Expected SIGMETRICS-level narrative supported by this implementation

The implementation should make it possible to substantiate the following story, without assuming the result in advance:

> Aggregate goodput is a poor vertical axis for interactivity robustness in a work-conserving speculative-decoding server because the verifier tends to remain saturated. We instead characterize the interactivity-concurrency capacity region: for each required per-user token rate, how many simultaneous streams can the system sustain? A queue-based online controller coordinates speculative depth, target verification batching, and shared wireless bandwidth under unknown acceptance. The resulting N*(x) frontier exposes capacity regimes hidden by aggregate throughput and its area summarizes robustness over uncertain interactivity requirements.

Whether experiments ultimately support all parts of this narrative must be determined by measurements, not forced by the code.

---

## 27. Final Codex deliverable

After implementation and pilot validation, create:

```text
results/CAPACITY_AUC_NSTAR_IMPLEMENTATION_REPORT.md
```

The report must answer, with exact code paths and measured evidence:

1. How is the nested population constructed and verified?
2. What exact empirical definition of per-client sustained rate is used?
3. What is the feasibility classifier and how often does it return `uncertain`?
4. Does N-search produce a monotone frontier without post-hoc monotonic correction?
5. What exact score selects `gamma_i` in `capacity_dpp`?
6. Does the scalar special case match `paper_index_gamma()`?
7. Does batch selection solve the intended queue-weighted knapsack?
8. Does the bandwidth allocator satisfy the analytical square-root/KKT solution?
9. What are `N*(x)` and ICA for the proposed policy in the pilot?
10. Which queue/resource becomes unstable immediately above the measured boundary?
11. How does `N*(x)` compare with the old nearly-rectangular goodput-vs-x view?
12. Are the results strong enough to justify running the full SIGMETRICS experiment matrix, or is another modeling/controller issue visible first?

If a flaw is found, report it explicitly before spending GPU time on final baselines.