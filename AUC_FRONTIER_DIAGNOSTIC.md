# AUC Frontier Diagnostic — Verify Whether the Frontier Is Structurally Flat

## Purpose

Before investing further in the AUC-maximization story, verify whether the current system actually has a **non-trivial goodput–interactivity trade-off**, or whether the frontier is structurally almost flat because the verifier remains work-conserving and saturated.

This is a diagnostic task. **Do not change the controller logic first.** Inspect the current implementation, run controlled experiments, and determine whether the observed flat frontier is:

1. an implementation artifact,
2. a consequence of the current scheduler/batching policy,
3. or a genuine structural property of the modeled system.

---

## 1. Core concern

The paper defines

\[
Y^\star(x)=\max_{\pi:\min_i x_i(\pi)\ge x} Y(\pi),
\]

where

- \(x\) is the minimum per-user interactivity requirement,
- \(x_i\) is the achieved interactivity of stream \(i\),
- \(Y=\sum_i x_i\) is system goodput.

By definition, if \(x_2>x_1\), then the feasible policy set shrinks:

\[
\mathcal F(x_2)\subseteq \mathcal F(x_1),
\]

so

\[
Y^\star(x_2)\le Y^\star(x_1).
\]

However, **non-increasing does not imply strictly decreasing**. A flat frontier is possible.

The current concern is that if the verifier is always saturated and the controller always fills the verification budget, then for every feasible \(x\), system goodput may remain approximately

\[
Y^\star(x)\approx \alpha_{\mathrm{eff}}\,C_{\mathrm{verify}},
\]

or, more generally, a nearly constant useful-token rate determined by verifier capacity.

Then the frontier is approximately rectangular:

\[
Y^\star(x)\approx
\begin{cases}
C, & 0\le x\le \bar x,\\
\text{infeasible}, & x>\bar x.
\end{cases}
\]

and

\[
\mathrm{AUC}\approx C\,\bar x.
\]

If this is what the system truly does, then AUC is mostly a product of maximum throughput and maximum sustainable minimum-user rate, rather than a rich Pareto trade-off.

---

## 2. Why this may happen

### 2.1 Work-conserving saturated verifier

If the server always has enough work and always verifies up to its token/batch budget, then changing the fairness/interactivity requirement may only redistribute service across users while leaving total useful-token production almost unchanged.

Example:

- verifier useful capacity = 100 tokens/s,
- 10 users,
- \(x_{\rm req}=2\): allocate service unevenly, total still 100,
- \(x_{\rm req}=5\): redistribute more evenly, total still 100,
- \(x_{\rm req}=10\): equalize completely, total still 100,
- only above 10 does the requirement become infeasible.

This produces a horizontal frontier until the feasibility boundary.

### 2.2 Constant reward per verification token

If

\[
\frac{\text{useful committed tokens}}{\text{verification tokens}}
\]

is nearly constant across users and operating points, then filling the same verification budget naturally gives nearly constant goodput.

### 2.3 Symmetric compute-bound regime

In a homogeneous saturated compute-bound regime,

\[
T_v\approx \theta_f\Gamma,
\]

and aggregate useful rate may approach a constant server capacity. In that regime, fairness constraints may change the allocation but not total goodput much.

This would be a **structural result**, not a bug.

---

## 3. What can create a non-trivial frontier

A decreasing frontier requires increasing interactivity to reduce system efficiency, not merely redistribute a fixed-capacity resource.

### 3.1 Heterogeneous acceptance

If users have different acceptance efficiencies, e.g.

- user A: \(\alpha=0.95\),
- user B: \(\alpha=0.50\),

then max-throughput scheduling may prefer A. A higher minimum-interactivity requirement may force more verifier budget onto B, reducing useful reward per unit server work.

Expected mechanism:

\[
x_{\rm req}\uparrow
\Rightarrow
\text{more service to low-efficiency users}
\Rightarrow
Y\downarrow.
\]

### 3.2 Heterogeneous draft/network cost

Slow drafting, low uplink rate, high RTT, or other client-specific costs may make some streams expensive to keep interactive. Raising their required rate may consume more network/server opportunity and lower aggregate goodput.

### 3.3 Batching-delay trade-off

This is likely the strongest mechanism to inspect.

Throughput-oriented behavior:

```text
wait -> build large/full batch -> high GPU efficiency
```

Interactivity-oriented behavior:

```text
launch earlier -> smaller/underfilled batch -> lower GPU efficiency
```

If the implementation always waits until the verification token budget is full, then the minimum-interactivity requirement may have almost no effect on server efficiency.

A non-trivial frontier requires the controller to be able to trade batching efficiency for interactivity, for example by launching an underfilled batch when queue/interactivity pressure is high.

---

## 4. Main implementation questions for Codex

Inspect the repository and answer each item with exact file/function references.

### Q1. Does `x_requirement` affect runtime decisions, or is it only checked after the run?

Trace `x_requirement` from config/CLI into the controller.

Determine whether it directly changes any of:

- virtual queue \(Z_i\),
- admission,
- client selection,
- batch launch timing,
- batch composition,
- speculation depth \(\gamma_i\),
- bandwidth allocation,
- token budget.

If `x_requirement` is only used to mark a completed run as feasible/infeasible, report that clearly.

### Q2. Does the server always wait for a full verification budget?

Inspect batching/scheduling logic.

Determine whether a batch can launch when:

- the token budget is not full,
- some client has high interactivity debt / high \(Z_i\),
- waiting longer would violate an interactivity target,
- there are ready requests but fewer than the preferred batch size.

Report the exact launch condition.

### Q3. Is the verifier work-conserving?

Determine whether, whenever ready verification work exists, the server is almost always busy.

Check whether there are explicit idle periods caused by:

- draft/network critical path,
- waiting for batch fill,
- controller admission,
- synchronization/gating,
- empty queues.

### Q4. Is useful reward per verification token approximately constant?

Compute per run:

\[
\eta_{\rm verify}
=
\frac{\text{committed useful tokens}}
{\text{verification token positions}}.
\]

Check whether \(\eta_{\rm verify}\) changes materially with `x_requirement`.

### Q5. Does increasing `x_requirement` alter service allocation across users?

For each `x_requirement`, compare:

- per-client service share,
- per-client committed-token rate,
- per-client selected \(\gamma_i\),
- per-client acceptance efficiency,
- per-client bandwidth allocation,
- per-client wait time.

If allocation changes but aggregate goodput does not, this supports the fixed-capacity redistribution explanation.

### Q6. Does increasing `x_requirement` change average batch size or underfilled-batch frequency?

This is critical.

If both remain unchanged, the implementation currently exposes little or no batching-efficiency/interactivity trade-off.

### Q7. Is admission/concurrency actually adaptive to `x_requirement`?

The framework may already have an admission/batching algorithm that determines how many streams to serve. Verify whether the number of admitted/active streams \(B(t)\) responds to interactivity pressure.

Do **not** assume that manually sweeping \(B\) is the correct main experiment. In the full framework, \(B(t)\) should be treated as a controller output if admission is part of the policy.

---

## 5. Diagnostic experiment to run

Use one fixed deployment and one fixed workload trace. Sweep only the minimum interactivity requirement.

Suggested values:

\[
x_{\rm req}\in\{0,1,2,3,4,5,6,8,10,12\}
\]

or stop after several consecutive infeasible points.

Use the same:

- prompt IDs,
- random seed,
- client profiles,
- network trace,
- model pair,
- controller parameter \(V\),
- run duration.

The purpose is not yet to compare many baselines. First characterize the proposed controller itself.

---

## 6. Mandatory metrics to log for every `x_requirement`

### Aggregate metrics

- `system_goodput`
- `achieved_x_min`
- `p10_interactivity`
- `median_interactivity`
- `constraint_satisfied`
- `target_utilization`
- `verifier_tokens_per_second`
- `useful_tokens_per_verification_token`
- `mean_batch_size`
- `mean_verification_tokens_per_batch`
- `underfilled_batch_fraction`
- `mean_batch_wait_ms`
- `p95_batch_wait_ms`
- `mean_selected_gamma`
- `mean_server_queue`
- `p95_server_queue`
- `mean_admitted_concurrency`
- `mean_active_clients`

### Per-client metrics

- client ID
- achieved interactivity \(x_i\)
- committed useful tokens
- service share
- acceptance rate
- selected \(\gamma_i\) distribution
- draft latency
- uplink latency
- server waiting time
- verification time
- bandwidth allocation
- virtual queue \(Z_i\)
- any local/server price used by the controller

### Per-batch metrics

- batch ID
- launch timestamp
- batch size
- total verification tokens \(\Gamma\)
- configured token budget
- fill ratio = \(\Gamma/\Gamma_{\rm budget}\)
- reason batch launched
- server service time
- number of clients represented
- whether the batch was launched due to queue/interactivity pressure

If the current logger does not have a `launch_reason`, add it before running the final diagnostic.

---

## 7. Figures to generate

### Figure A — Goodput vs requirement

- x-axis: `x_requirement`
- y-axis: `system_goodput`
- mark feasible and infeasible runs separately

This reproduces the observed flat-then-infeasible behavior.

### Figure B — Achieved minimum interactivity vs requirement

- x-axis: `x_requirement`
- y-axis: `achieved_x_min`
- add reference line \(y=x\)

This identifies the feasibility boundary \(\bar x\).

### Figure C — Verifier saturation diagnostics

- x-axis: `x_requirement`
- lines:
  - target utilization,
  - verification tokens/s,
  - useful tokens / verification token.

If all three are nearly constant over the feasible range, a flat goodput frontier is expected.

### Figure D — Batching behavior

- x-axis: `x_requirement`
- lines:
  - mean batch size,
  - mean fill ratio,
  - underfilled-batch fraction,
  - p95 batch wait.

If batching behavior is invariant to `x_requirement`, then the controller currently has no batching-efficiency/interactivity trade-off.

### Figure E — Allocation redistribution

For representative low and high feasible `x_requirement` values, show per-client:

- service share,
- \(x_i\),
- acceptance rate,
- mean \(\gamma_i\).

This distinguishes “same total capacity, different allocation” from “actual efficiency loss”.

---

## 8. Interpretation / decision tree

### Case A — Flat goodput + constant verifier efficiency + constant batching behavior

Evidence:

- target utilization approximately constant and near saturation,
- verification tokens/s approximately constant,
- useful tokens / verification token approximately constant,
- batch fill ratio approximately constant,
- only per-client allocation changes.

Conclusion:

> The flat frontier is structurally consistent with a saturated work-conserving verifier. `x_requirement` mainly redistributes a fixed-capacity resource.

Implication for paper:

- AUC may be close to \(Y_{\max}\bar x\),
- the “rich whole-frontier optimization” thesis becomes weak,
- consider reframing the main problem as maximizing goodput subject to heterogeneous interactivity constraints, with AUC as an evaluation summary rather than the central runtime objective.

### Case B — Flat goodput because `x_requirement` does not affect controller decisions

Evidence:

- `x_requirement` only appears in post-run feasibility checks or weakly in logging,
- no change in admission, batching, \(\gamma\), bandwidth, or priorities.

Conclusion:

> This is primarily an implementation/modeling issue. The controller is not actually solving a different constrained problem as `x_requirement` changes.

Action:

- report exact missing coupling,
- propose the smallest code change needed so interactivity pressure can affect runtime decisions,
- do not implement the change until the diagnosis is reviewed.

### Case C — `x_requirement` changes allocation but not batching efficiency

Evidence:

- per-client shares change,
- \(Z_i\) changes,
- but target utilization, batch size, and reward per verification token stay constant.

Conclusion:

> The controller is enforcing fairness/interactivity, but the modeled system has almost fixed aggregate capacity.

This is a structural warning for the AUC thesis.

### Case D — Higher `x_requirement` causes underfilled/earlier batches or service to inefficient clients, and goodput decreases

Evidence:

- batch size/fill ratio decreases or low-efficiency users receive more verifier work,
- useful reward per server work decreases,
- goodput decreases while achieved minimum interactivity increases.

Conclusion:

> A genuine non-trivial goodput–interactivity frontier exists. AUC remains a meaningful central evaluation object.

The paper should explicitly identify which mechanism generates the slope.

---

## 9. Important theoretical clarification

Do not expect a strictly decreasing curve by definition.

The theory only guarantees

\[
Y^\star(x_2)\le Y^\star(x_1)\quad\text{for }x_2>x_1.
\]

Plateaus are valid.

A policy that already achieves \(x_{\min}=4\) with goodput 80 will produce the same operating point for requirements 0, 1, 2, 3, and 4 unless the controller has a reason to choose a different action.

Thus a “flat then infeasible” graph is not automatically a bug.

---

## 10. Do not manually sweep admitted concurrency in the main diagnostic

The framework may already contain an admission/batching algorithm that chooses how many streams to serve.

For the **full controller diagnostic**:

- sweep `x_requirement`,
- keep external workload fixed,
- let the controller choose admission/concurrency,
- log \(B(t)\) / active/admitted concurrency as an output.

Manual \(B\) sweeps remain useful for the separate symmetric structural experiment that studies \(\gamma^\star(B)\), but they should not replace the full controller's own admission logic when diagnosing the AUC frontier.

---

## 11. Requested Codex deliverable

Please inspect the repository, run the diagnostic, and create a report:

```text
results/AUC_FRONTIER_DIAGNOSTIC_REPORT.md
```

The report must contain:

1. exact code paths/functions where `x_requirement` enters the system;
2. exact batch-launch condition;
3. whether admission/concurrency responds to `x_requirement`;
4. whether batches can launch underfilled;
5. whether the verifier is work-conserving/saturated;
6. the five diagnostic figures above;
7. a compact table for every `x_requirement` with all aggregate metrics;
8. one of the Case A/B/C/D diagnoses above, with evidence;
9. a conclusion answering:

> **Does the current AUC-maximization formulation produce a genuinely non-trivial goodput–interactivity frontier in this implementation, or is AUC effectively collapsing to throughput × feasible interactivity range?**

Do not hide a flat result. The purpose of this task is to determine whether the current central paper thesis is empirically justified before further implementation effort.