# AUC Frontier Diagnostic Report

Diagnostic requested in `AUC_FRONTIER_DIAGNOSTIC.md`. All findings below are from
inspecting `legacy/src/edge_specsim/{controller,simulator,verification_batcher,
verification_queue}.py` and analyzing the 24 already-collected real-GPU runs in
`results/exp1_frontier/raw/x_*.csv` (8 values of `x_requirement` x 3 seeds,
N=21 clients, gsm8k, `verification_batching.scheduler: fcfs`). No new GPU runs
were needed to reach the primary diagnosis; one small confirmatory re-run with a
corrected scheduler is recommended at the end (not yet executed).

---

## 1. Where `x_requirement` enters the system (exact code path)

`configs/*.yaml`'s `controller.min_interactivity_tps` (aliased `simulation.min_interactivity_tps`
as fallback) flows into:

```
simulator.py:  self.controller = OnlineController(..., min_tps=float(controller_cfg.get(
                   "min_interactivity_tps", sim["min_interactivity_tps"])), ...)
controller.py: self.min_tps = min_tps   (constructor, line ~65)
```

It is used in **exactly one place**: `OnlineController.observe_completion()` (eq. 12's Z_i update):

```python
# controller.py, observe_completion()
client.z_queue = max(0.0, slot_base_queue + self.min_tps * slot_seconds - slot_useful_tokens)
```

`client.z_queue` then feeds `_paper_index_cost_ratio()`:

```python
# controller.py, _paper_index_cost_ratio()
queue_weight = client.z_queue if self.use_virtual_queues else 0.0
benefit_scale = max(1e-9, self.V + queue_weight)
```

which sets the cost ratio `c_i` used by `paper_index_gamma()` to pick γ_i (`policy="adaptive_index"`,
our AUC-controller). **This is a real, correct coupling, not a post-run-only check** — Q1's "Case B:
no coupling" does not apply literally. But it is a **single, weak lever**: it only changes γ_i
uniformly upward once Z_i has already accumulated positive backlog, i.e. only after the
requirement has already been missed for a while. Below the system's natural throughput, Z_i stays
at 0 forever and the lever is a complete no-op (confirmed empirically, see Section 3).

`z_queue`/`weight` is also passed into `VerificationBatcher.submit(weight=self.controller.V +
client.z_queue, ...)` (simulator.py `_run_round`) — this is the **second**, batching-side lever
the theory needs (Theorem 3(ii), knapsack by ω_i/(γ_i+1)). See Section 2 for why it is currently
inert.

## 2. Exact batch-launch condition (Q2) — and why it ignores `x_requirement`

`VerificationQueue.pop_batch()` (`verification_queue.py`) selects a batch when:

```python
cutoff_ms = self.waiting[0].arrival_time + batch_wait_ms   # batch_wait_ms = 5.0 in our config
eligible = [r for r in self.waiting if r.arrival_time <= cutoff_ms]
```

then applies `scheduler_name` (our config: **`fcfs`**) to rank/select up to `max_batch_size` items
within `verify_token_budget`. **`fcfs`'s `_select_fcfs()` takes requests in arrival order and does
not look at `request.weight` (the `(V+Z_i)` term) at all.** The `weighted_utility` and `knapsack`
scheduler options exist in the same file and *do* use `weight` — but our experiment config never
selected them. **This means the AUC-controller's server-side batching subproblem (Theorem 3(ii))
was never actually exercised in Exp 1: interactivity pressure only ever reaches γ_i, never batch
composition or batch timing.**

There is also no mechanism anywhere in `_loop()`/`pop_batch()` that launches an *early, underfilled*
batch because some client's `Z_i` is high or an interactivity deadline is close — the only early-exit
condition is "no more requests arrive before `cutoff_ms`" (a fixed 5ms wait), independent of any
per-client urgency signal. Section 3.3's batching-efficiency/interactivity trade-off mechanism does
not exist in the current implementation.

## 3. Is the verifier saturated / work-conserving? (Q3) — empirically, no: it is barely used

| x_target | mean_batch_size | mean_fill_ratio | useful_tok/verify_tok | verifier_tok/s |
|---|---|---|---|---|
| 0.1 | 1.037 | 3.52% | 0.868 | 36.58 |
| 0.3 | 1.032 | 3.51% | 0.869 | 36.28 |
| 0.6 | 1.034 | 3.52% | 0.867 | 36.40 |
| 1.0 | 1.035 | 3.51% | 0.866 | 36.45 |
| 1.3 | 1.034 | 3.51% | 0.867 | 36.63 |
| 1.6 | 1.038 | 3.56% | 0.869 | 36.93 |
| 2.0 | 1.037 | 3.66% | 0.867 | 37.39 |
| 2.5 | 1.031 | 3.74% | 0.866 | 37.48 |

(full per-seed table: `results/AUC_FRONTIER_DIAGNOSTIC_metrics.csv`; per-x_target summary:
`results/AUC_FRONTIER_DIAGNOSTIC_metrics_summary.csv`)

**`mean_batch_size ≈ 1.03` and `mean_fill_ratio ≈ 3.5%` at every single x_target.** The server's
100-token verify budget is used at ~1/28th of capacity; 94%+ of verify calls contain exactly one
client's request. Root cause: `verification_batching.batch_wait_ms = 5.0` (5 milliseconds) is far
shorter than a real client's round-trip time (draft compute + network model + real vLLM verify call,
tens to low-hundreds of ms measured), so almost no two clients' requests ever land inside the same
5ms window regardless of which scheduler is configured. **The verifier is not saturated — it has
massive spare capacity — but batching essentially never happens, so there is no shared capacity
being contended for or redistributed in the first place.** This is a different (and more fixable)
root cause than Section 2.1's "saturated work-conserving verifier" hypothesis: the bottleneck here
is not verifier contention, it is that `batch_wait_ms` is misconfigured for this workload's real
timing, which independently also explains Q2's finding (batching lever is inert because there is
essentially no batching to act on).

## 4. Is useful reward per verification token constant? (Q4) — yes, essentially exactly

`useful_tokens_per_verification_token` ranges **0.866–0.869** across all 8 x_target values (3rd
decimal noise only, no trend). This matches Section 2.2's hypothesis, but note it follows
*mechanically* from γ_max=4 and a roughly-constant achieved acceptance rate (α_hat), not from any
adaptive batching efficiency — since batch_size≈1 always, there is no batching-efficiency channel
for this ratio to vary through in the first place.

## 5. Does `x_requirement` change allocation across users? (Q5) — only through Z_i, weakly

| x_target | mean_z_queue | mean_server_queue | mean_gamma | Y_mean | achieved min_x | constraint met |
|---|---|---|---|---|---|---|
| 0.1 | 0.000 | 45.05 | 2.398 | 35.35 | 1.512 | yes |
| 0.3 | 0.000 | 50.49 | 2.399 | 35.07 | 1.502 | yes |
| 0.6 | 0.044 | 48.40 | 2.399 | 35.15 | 1.504 | yes |
| 1.0 | 0.460 | 70.62 | 2.397 | 35.16 | 1.503 | yes |
| 1.3 | 1.736 | 49.08 | 2.399 | 35.37 | 1.513 | yes |
| 1.6 | 23.24 | 42.13 | 2.431 | 35.75 | 1.529 | **no** |
| 2.0 | 73.70 | 52.11 | 2.524 | 36.35 | 1.544 | **no** |
| 2.5 | 147.74 | 45.01 | 2.622 | 36.68 | 1.546 | **no** |

`mean_z_queue` grows by **>3 orders of magnitude** (0.0 -> 147.7) as x_target goes from 0.1 to 2.5 —
the Z_i coupling from Section 1 is real and responds correctly in direction. But its only effect is
a **9% increase in mean γ** (2.40 -> 2.62) applied roughly uniformly, which raises `Y_mean` by only
**+3.8%** (35.35 -> 36.68) — and critically, **Y increases with x instead of decreasing**, directly
violating Proposition 2(ii)'s Y*(·) nonincreasing property. `mean_server_queue` (Q, the batching-side
price) shows no trend at all with x (45–70, noisy) — consistent with Section 2's batching lever being
disconnected from x_requirement (Section 2's fcfs scheduler finding).

**Conclusion for Q5:** allocation does shift with x (Z_i, γ both move), but not in the direction or
through the mechanism the theory requires (fairness-driven reallocation from strong to weak streams
via batching/pricing) — it is a uniform γ inflation from one runaway backlog term, not a Y*(x)
trade-off.

## 6. Does `x_requirement` change batch size or underfilled-batch frequency? (Q6) — no

`mean_batch_size` (1.031–1.038) and `mean_fill_ratio` (3.51%–3.74%) are flat across the entire
x_target sweep (see Section 3's table) — confirms directly that the implementation currently
exposes **no batching-efficiency/interactivity trade-off**, matching Section 3.3's concern exactly.

## 7. Is admission/concurrency adaptive to `x_requirement`? (Q7) — no, there is no admission control

`simulator.py`'s `_run_client()` runs every one of the `simulation.num_clients` (=21, a static config
value) continuously for the whole experiment; nothing in the codebase ever reduces the number of
active/admitted clients in response to interactivity pressure or server load. `B(t)` (admitted
concurrency) is not a controller output anywhere in this implementation — it is fixed externally by
config. (Manual B-sweeps, per Section 10 of the diagnostic spec, remain appropriate for the separate
symmetric γ*(B) structural experiment, not this diagnostic.)

## 8. Figures

Given the flat/near-degenerate nature of the swept range (`achieved_x_min` only spans
1.50–1.55 while `x_target` was swept 0.1–2.5 — see the previous conversation's diagnosis), the five
requested figures reduce to the tables in Sections 3 and 5 above, which already show every series
(fill ratio, useful/verify-token ratio, batch size, Z_i, server queue, Y, min_x) as a function of
x_target. A plotting pass (matplotlib PNGs) can be added on request; the numbers already settle the
diagnosis without it.

## 9. Diagnosis: which Case?

**None of Case A/B/C/D applies cleanly; this is closest to Case B, with two concrete, fixable root
causes, not a fundamental structural fixed-capacity result:**

1. **The batching-side lever (Theorem 3(ii), the mechanism Section 3.3 identifies as the strongest
   candidate for a real frontier slope) is switched off by configuration**: `verification_batching.
   scheduler: fcfs` discards the `(V+Z_i)`-weighted `expected_useful_tokens` value entirely. Switching
   to `scheduler: weighted_utility` or `scheduler: knapsack` (both already implemented in
   `verification_queue.py`, unused until now) would let batch *composition* respond to interactivity
   pressure, as the theory requires.

2. **Batches essentially never form at all** (`batch_wait_ms=5.0` << real per-client round-trip time,
   giving `mean_batch_size≈1.03`), independent of scheduler choice — there is no batching efficiency
   to trade off because there is no batching. This must also be fixed (raise `batch_wait_ms`, e.g. to
   50–200ms, matched to the real round-trip time this workload/N/GPU-sharing configuration produces)
   before item 1's scheduler fix can have any effect.

The verifier is **not saturated** (mean fill ratio 3.5%) — Case A's "fixed verifier capacity" story
does not fit this deployment as configured; there is no contended shared resource being redistributed
because batching does not happen. The one lever that *is* wired up (Z_i -> γ_i) is real (Q1: not a
pure post-run check) but too weak and too uniform (raises γ for everyone by ~9% rather than
reallocating verifier priority toward the lagging stream), which is why `Y*(x)` came out increasing
rather than the theoretically-required nonincreasing/concave shape.

## Answer to the central question

> Does the current AUC-maximization formulation produce a genuinely non-trivial goodput–interactivity
> frontier in this implementation, or is AUC effectively collapsing to throughput x feasible
> interactivity range?

**Neither, exactly — the implementation is not yet exercising the mechanism the theory relies on to
produce a genuine frontier.** The batching-side control (server-side knapsack pricing by `(V+Z_i)`)
that Theorem 3 needs to reallocate verifier priority toward interactivity-starved streams is present
in the codebase but disabled by two config choices (`scheduler: fcfs`, `batch_wait_ms` far shorter
than real round-trip time). With only the device-side γ_i lever active, and that lever moving all
clients' γ together rather than differentially, the measured `Y(x)` is not a valid empirical proxy
for `Y*(x;ξ)` and should not be used to conclude anything about whether the AUC thesis holds
structurally in this system. **Recommended before any further conclusion:** re-run the same
x_requirement sweep with `scheduler: weighted_utility` (or `knapsack`) and `batch_wait_ms` raised to
match real observed round-trip time (e.g. 100ms, informed by `round_latency_ms`'s p50 in the existing
CSVs), and check whether `mean_batch_size`/`mean_fill_ratio` become x-dependent and whether `Y(x)`
turns nonincreasing. This has not yet been run — it is the concrete next diagnostic step, per Section
6's "do not implement the change until the diagnosis is reviewed."
