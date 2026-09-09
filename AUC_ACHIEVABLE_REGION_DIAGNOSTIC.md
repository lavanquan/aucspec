# AUC Achievable-Region Diagnostic — Distinguish a Flat Physical Frontier from Controller Failure

## Purpose

The latest `results/AUC_FRONTIER_DIAGNOSTIC_REPORT.md` shows that the current controller produces an almost flat goodput curve across `x_requirement`, even after:

- fixing weighted server scheduling,
- increasing batching delay so real batches form,
- forcing verifier scarcity (`verify_token_budget=6`, fill ratio ~78–80%),
- introducing strong latency/network heterogeneity,
- introducing real acceptance heterogeneity across GSM8K / CNN-DailyMail / MATH clients.

However, this does **not yet prove that the physical achievable frontier**

\[
Y^\star(x)=\max_{\pi:\min_i x_i(\pi)\ge x}Y(\pi)
\]

is rectangular.

What has actually been measured so far is closer to

\[
Y_{\pi_{\rm controller}}(x),
\]

that is, the behavior of the current controller when `x_requirement` changes.

The controller may simply fail to move to a different feasible operating point when the requirement tightens.

This diagnostic must answer the following decisive question:

> **Does there exist any reachable operating point with higher minimum interactivity if we deliberately sacrifice system goodput, or is the observed ~0.9 token/s/client ceiling a true physical feasibility boundary?**

Do not run another ordinary `x_requirement` sweep. The purpose here is to probe the **achievable operating region independently of the AUC controller**.

---

## 1. Why this experiment is necessary

The latest acceptance-heterogeneity diagnostic reported approximately:

- `Y(x)` = 25.63–26.04 tokens/s,
- useful/verify efficiency `eta(x)` = 0.778–0.783,
- hard-MATH service share = 0.303–0.305,
- achieved `min_x` ≈ 0.88–0.91,
- requirements >= 1.0 violated.

The critical observation is that service allocation barely changes as `x_requirement` increases.

Thus there are two competing explanations.

### Hypothesis A — True rectangular achievable region

No policy can materially increase `min_x` beyond ~0.9 under this fixed deployment.

Then the true frontier is approximately

\[
Y^\star(x)\approx C,\qquad 0\le x\le \bar x\approx0.9,
\]

followed by infeasibility.

In this case AUC effectively reduces to

\[
\mathrm{AUC}\approx C\bar x.
\]

### Hypothesis B — Controller failure

There exist policies that trade some aggregate goodput for higher minimum interactivity, e.g.

\[
(x_{\min},Y)=(0.9,26),
(1.0,24),
(1.15,22),
(1.25,20),
\]

but the current virtual-queue/index controller does not discover them.

If so, the AUC formulation is still meaningful; the controller implementation/algorithm is the problem.

This diagnostic must distinguish A from B.

---

## 2. Core experimental principle

Instead of asking the controller to react to `x_requirement`, directly create a family of deliberately biased scheduling policies and map the reachable point cloud

\[
\mathcal R
=
\{(x_{\min}(\pi),Y(\pi)):\pi\in\Pi_{\rm probe}\}.
\]

Then reconstruct the empirical upper envelope

\[
\widehat Y^\star(x)
=
\max_{\pi\in\Pi_{\rm probe}:x_{\min}(\pi)\ge x}Y(\pi).
\]

The probe policies should intentionally favor the current bottleneck group, even if doing so is globally inefficient.

This is not a production controller. It is a diagnostic search over the reachable region.

---

## 3. Fixed experimental configuration

Start from the **scarce, contended, acceptance-heterogeneous** configuration used in the latest diagnostic.

Keep fixed:

- target model: Qwen2.5-7B-Instruct,
- draft model: Qwen2.5-1.5B-Instruct,
- N = 21 clients,
- three equal client groups (7 each):
  - `easy_gsm8k`,
  - `medium_cnn_summarize`,
  - `hard_math`,
- identical latency/network ranges across the three groups,
- real measured acceptance heterogeneity only,
- `scheduler: weighted_utility` or `knapsack`,
- `batch_wait_ms = 150`,
- `verify_token_budget = 6`,
- same seeds and prompt assignments across all probe policies,
- same `gamma_choices` initially as the latest diagnostic,
- same run duration / question count.

Do **not** change deployment parameters while sweeping priority. The only thing that should change is the artificial service-priority bias.

---

## 4. Required implementation: explicit class-priority multipliers

Add a diagnostic-only multiplicative priority factor to the server-side request weight.

Current conceptual weight is approximately

\[
w_i = V + Z_i.
\]

For this diagnostic define

\[
\widetilde w_i
= m_{c(i)}\,(V+Z_i),
\]

where `c(i)` is the client's dataset/class.

At minimum support:

```yaml
clients:
  diagnostic_priority_multiplier:
    easy_gsm8k: 1.0
    medium_cnn_summarize: 1.0
    hard_math: 1.0
```

The multiplier must affect the **server-side batch selection / verification priority** directly.

It must not alter:

- acceptance observations,
- draft latency,
- network latency,
- dataset sampling,
- reward accounting,
- `x_requirement`,
- virtual queue update equations.

This is a diagnostic lever only.

If the current scheduler normalizes or transforms utility before selection, confirm that the class multiplier survives that transformation and genuinely changes ranking.

---

## 5. Main sweep

### Sweep A — prioritize the current bottleneck group

Fix:

\[
m_{easy}=1,
\qquad
m_{medium}=1.
\]

Sweep:

\[
m_{hard}\in\{1,2,4,8,16,32,64\}.
\]

For each multiplier, run at least 3 seeds with exactly the same workload assignments as the baseline.

The purpose is to force increasingly more verifier service toward the hard-MATH group.

### Sweep B — two-dimensional priority search

If Sweep A increases hard-group service share but does not clearly maximize `min_x`, run:

\[
m_{medium}\in\{1,2,4,8\},
\qquad
m_{hard}\in\{1,2,4,8,16,32\}.
\]

Keep `m_easy = 1`.

This creates a small grid of deliberately distorted service allocations.

### Optional Sweep C — direct fixed service quotas

If weighted priority does not reliably produce distinct group shares because of scheduler details, implement a stronger diagnostic mode with explicit per-group verifier-token or batch-slot quotas, e.g.

```yaml
verification_batching:
  diagnostic_group_share:
    easy_gsm8k: 0.2
    medium_cnn_summarize: 0.3
    hard_math: 0.5
```

The point is not algorithmic elegance. The goal is to determine whether the physical system can reach a higher `min_x` when more scarce verifier capacity is forcibly given to the bottleneck group.

---

## 6. Do not use `x_requirement` as the sweep variable

For this diagnostic, set `x_requirement` to a benign fixed value that does not itself dominate behavior, for example:

\[
x_{req}=0
\]

or a small feasible value such as 0.1.

We are not testing whether the controller responds to a requirement.

We are testing the reachable operating region directly.

The independent variable is the forced scheduling bias / service quota.

---

## 7. Mandatory metrics

For each policy point, record the following aggregate metrics:

- system goodput `Y`,
- achieved `x_min`,
- p10 / median / max per-client interactivity,
- useful tokens / verification token `eta`,
- verifier tokens/s,
- verifier fill ratio,
- mean batch size,
- target utilization,
- mean selected gamma,
- class-level service shares,
- class-level verifier-token shares,
- class-level committed useful-token shares.

For each class record:

- mean `x_i`,
- minimum `x_i` within the class,
- mean acceptance rate / alpha estimate,
- mean UCB alpha if active,
- mean selected gamma,
- number of verification opportunities,
- number of verification token positions,
- committed useful tokens,
- average wait time,
- average Z queue.

For the hard-MATH group specifically, verify that increasing `m_hard` actually increases at least one of:

- verification-opportunity share,
- verifier-token share,
- batch inclusion frequency.

If priority changes do not change real service allocation, stop and fix the diagnostic implementation before interpreting the results.

---

## 8. Required plots

### Figure 1 — Achievable point cloud

Scatter plot:

- x-axis: achieved `x_min`,
- y-axis: system goodput `Y`,
- one point per `(priority setting, seed)` or one mean point with CI,
- annotate representative `m_hard` values.

This is the most important figure.

### Figure 2 — Empirical upper envelope

Construct

\[
\widehat Y^\star(x)
=
\max_{j:x_{\min,j}\ge x}Y_j.
\]

Plot the stepwise upper envelope over the common domain.

Do not use arbitrary spline interpolation.

If confidence intervals are needed, bootstrap across seeds and recompute the envelope per bootstrap sample.

### Figure 3 — Hard-group service response

- x-axis: `m_hard`,
- y-axis: hard-group service share / verifier-token share.

This verifies that the diagnostic lever is actually moving scarce capacity.

### Figure 4 — Mechanism plot

Against `m_hard`, plot:

- achieved `x_min`,
- `eta`,
- system goodput `Y`.

The expected non-trivial-tradeoff signature is:

\[
m_{hard}\uparrow
\Rightarrow
\text{hard service share}\uparrow
\Rightarrow
x_{min}\uparrow,
\eta\downarrow,
Y\downarrow.
\]

### Figure 5 — Per-class interactivity

Plot mean/min interactivity for easy, medium, hard groups vs `m_hard`.

This reveals whether the bottleneck is actually being lifted or whether capacity is wasted without improving the slowest users.

---

## 9. Decision criteria

### Outcome A — Non-trivial achievable frontier exists

Suppose increasing `m_hard` produces points such as

\[
(0.90,26),
(1.02,25),
(1.15,23),
(1.25,20).
\]

Then conclude:

> The physical achievable region contains a genuine goodput–interactivity trade-off. The current x-requirement controller fails to trace it.

Implication:

- keep AUC formulation,
- redesign/fix the controller,
- investigate why `Z_i` / weighted scheduling does not generate the same reallocation automatically,
- compare the controller against this empirical upper-envelope reference.

This outcome means the current flat `Y_controller(x)` should **not** be interpreted as evidence against AUC.

### Outcome B — Service shifts, but `x_min` does not improve

Suppose hard-group verifier share rises strongly, goodput drops, but `x_min` remains around ~0.9.

Then conclude:

> The ~0.9 ceiling is not caused by insufficient verifier priority; the bottleneck lies elsewhere on the per-client critical path.

Next inspect:

- draft generation throughput,
- RTT / communication round structure,
- sequential dependency between speculative rounds,
- maximum sustainable token rate of the hard group even under exclusive service.

Run an additional **single-group hard-MATH capacity test**:

- only hard-MATH clients,
- reduced N,
- optionally one client at a time,
- measure maximum achievable per-client token rate.

If even exclusive service cannot exceed ~0.9, then the ceiling is genuinely physical for that client class.

### Outcome C — `x_min` improves without significant `Y` loss

If hard prioritization raises `x_min` while `Y` stays nearly constant, then the previous controller was simply using a poor allocation even within the same capacity region.

Implication:

- AUC still has limited curvature,
- but the controller should be fixed because it is not finding efficient allocations.

### Outcome D — Priority multiplier fails to change service share

This is an invalid diagnostic.

Do not interpret the frontier.

Fix the scheduler coupling or use explicit group quotas until service allocation clearly moves.

---

## 10. Additional physical-ceiling test

If Outcome B is observed, run a direct capacity experiment to estimate the maximum sustainable per-client rate of each class.

For each dataset/class separately:

- N in `{1,2,4,7}` clients,
- same target/draft pair,
- no competing classes,
- verifier budget sufficiently high not to throttle unless intentionally testing contention,
- measure each client's maximum sustained useful token rate.

This gives a class-specific ceiling

\[
\bar x_c^{\rm isolated}.
\]

If

\[
\bar x_{hard}^{\rm isolated}\approx0.9,
\]

then no system-level scheduler can satisfy a common floor above ~0.9 while serving hard-MATH clients under this deployment.

That would strongly support a truly rectangular frontier with

\[
\bar x\approx\min_c \bar x_c^{\rm isolated}.
\]

---

## 11. Important interpretation rule

Do not conflate

\[
Y_{controller}(x)
\]

with

\[
Y^\star(x).
\]

The monotonicity statement

\[
x_2>x_1
\Rightarrow
Y^\star(x_2)\le Y^\star(x_1)
\]

is a property of the **optimal constrained value function**, not of an arbitrary controller run.

The latest diagnostic only proves that the current controller produces a nearly fixed allocation as `x_requirement` changes.

This new experiment is required to determine whether the **reachable system region itself** contains lower-goodput / higher-interactivity points.

---

## 12. Requested Codex deliverables

Implement the diagnostic and commit:

```text
results/AUC_ACHIEVABLE_REGION_REPORT.md
results/auc_achievable_region/points_raw.csv
results/auc_achievable_region/points_summary.csv
results/auc_achievable_region/frontier_points.csv
results/auc_achievable_region/mechanism_summary.json
```

Also add reproducible scripts/configs, preferably:

```text
scripts/run_auc_achievable_region.py
configs/auc_achievable_region_base.yaml
```

or equivalent paths consistent with the repository structure.

The report must include:

1. exact code path where the diagnostic class multiplier is applied;
2. proof from logged data that increasing `m_hard` changes actual hard-group service share;
3. the achievable `(x_min, Y)` point cloud;
4. the empirical upper envelope `Y_hat_star(x)`;
5. mechanism plots showing service share, `x_min`, `eta`, and `Y`;
6. one of Outcome A/B/C/D above;
7. if Outcome B, the isolated hard-MATH capacity test;
8. a final answer to:

> **Is the current near-rectangular AUC frontier a true property of the achievable system region, or is it primarily a failure of the current controller to move along that region?**

Do not optimize the current AUC controller during this diagnostic. The purpose is to independently establish the shape of the reachable operating region first.