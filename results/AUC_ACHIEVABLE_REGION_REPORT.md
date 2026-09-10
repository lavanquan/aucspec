# AUC Achievable-Region Diagnostic — Report

Protocol: `AUC_ACHIEVABLE_REGION_DIAGNOSTIC.md`. Goal: decide whether the near-rectangular
`Y_controller(x)` measured in `AUC_FRONTIER_DIAGNOSTIC_REPORT.md` (Sections 10–11) reflects a
true rectangular *achievable* region `Y*(x)` or merely the current controller failing to move
along that region.

All runs: real GPU inference on Setonix (AMD MI250X / ROCm vLLM 0.9.1), Qwen2.5-7B target /
Qwen2.5-1.5B draft, N=21 clients, 3 equal dataset-defined groups (7 each), identical
latency/network ranges across groups, `scheduler: weighted_utility`, `batch_wait_ms=150`,
`x_requirement` fixed at a benign 0.1 (NOT swept). Independent variable = a diagnostic-only
per-class priority multiplier. 3 seeds per point.

---

## 1. Exact code path of the diagnostic class multiplier

- Config key: `clients.diagnostic_priority_multiplier: {easy_gsm8k, medium_cnn_summarize, hard_math}`
  (`configs/auc_achievable_region_base.yaml`).
- Stored per client: `legacy/src/edge_specsim/models.py` — `ClientProfile.diagnostic_priority_multiplier`
  (default `1.0`, so it is a no-op for every prior experiment).
- Assigned in `legacy/src/edge_specsim/simulator.py` `_make_clients()`:
  `diagnostic_priority_multiplier=float(c.get("diagnostic_priority_multiplier", {}).get(client_class_name, 1.0))`.
- **Applied** in `simulator.py` at the verifier-submit call:
  `weight = client.diagnostic_priority_multiplier * (self.controller.V + (client.z_queue if use_virtual_queues else 0.0))`
  i.e. `w_i = m_{c(i)} * (V + Z_i)` exactly as Section 4 specifies.
- That `weight` flows unchanged into `VerificationRequest.weight`
  (`legacy/src/edge_specsim/verification_batcher.py` `submit()`), and the `weighted_utility`
  scheduler ranks by `batching_utility / verifier_token_cost` where
  `batching_utility = weight * expected_useful_tokens`
  (`legacy/src/edge_specsim/verification_queue.py` `_weighted_utility_density`).
- It does **not** touch acceptance sampling, draft/network latency, dataset sampling, reward
  accounting, `x_requirement`, or the virtual-queue update equations.

Drivers: `scripts/run_auc_achievable_region.py` (`--sweep a|am|b`, `--verify-token-budget`,
`--tag`), `scripts/run_auc_achievable_region.sh`, `scripts/plot_auc_achievable_region.py`.

---

## 2. The bottleneck group is NOT the one the task name suggests

The diagnostic (inherited from Exp5) labelled the groups easy/medium/hard by *task* difficulty.
Measured behaviour is different:

| group | mean measured `alpha_hat` | mean per-class `x_min` |
|---|---|---|
| easy_gsm8k | 0.89 | ~1.74–1.90 |
| **medium_cnn_summarize** | **0.77** | **~1.37–1.56  ← sets the system floor** |
| hard_math | 0.93 | ~2.21–2.40 |

`hard_math` has the **highest** acceptance and the **highest** interactivity — it is never the
constraint on `min_i x_i`. `medium_cnn_summarize` (CNN/DailyMail summarization: the draft model
predicts free-form summary phrasing far worse than structured math reasoning) is the true
bottleneck. This invalidated the first sweep and forced a corrected one.

---

## 3. Batch competition — is the lever even exercisable?

With the Section-11 config (`verify_token_budget=6`), `mean_batch_size ≈ 1.2`: batches almost
always hold a single request, so no scheduler (this multiplier, or the controller's own
`V+Z_i`) has anything to rank. Raising the budget to 20/12 OOMs — the `medium_cnn_summarize`
prompts are ~6× longer (mean ~935, max ~2400 context tokens vs ~160 for gsm8k), so
`prompt_logprobs` peak memory in a multi-request batch is far larger than in the
gsm8k-only Exp1 runs (which tolerated `budget=100`). `verify_token_budget=10` +
`target_gpu_memory_utilization=0.65` + `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` runs
stably and gives `mean_batch_size ≈ 1.8` — real, if modest, within-batch competition. All
headline results below use `budget=10`.

---

## 4. Sweep A — boost `hard_math` (the wrong group), vary `m_hard` ∈ {1..64}

| m_hard | Y | min_x | eta | hard_math vtok-share | mean_batch |
|---|---|---|---|---|---|
| 1  | 38.89 | 1.357 | 0.757 | 0.300 | 1.79 |
| 2  | 39.40 | 1.340 | 0.769 | 0.292 | 1.81 |
| 4  | 38.58 | 1.312 | 0.765 | 0.292 | 1.79 |
| 8  | 38.91 | 1.343 | 0.765 | 0.291 | 1.81 |
| 16 | 39.31 | 1.336 | 0.766 | 0.291 | 1.82 |
| 32 | 39.46 | 1.364 | 0.764 | 0.290 | 1.82 |
| 64 | 39.02 | 1.353 | 0.765 | 0.292 | 1.81 |

A 64× multiplier moves nothing (`min_x` 1.31–1.36, all within ±1σ; `Y`, `eta`, `vtok-share`
flat). Expected: `hard_math`'s own `x_min` is already ~2.2–2.4, far above the system floor, so
giving it more verifier priority cannot lift `min_i x_i`.

---

## 5. Sweep AM — boost `medium_cnn_summarize` (true bottleneck), vary `m_medium` ∈ {1..64}

| m_medium | Y | min_x | eta | cnn service-share | cnn vtok-share | mean_batch |
|---|---|---|---|---|---|---|
| 1  | 39.14 | **1.372** | 0.761 | 0.348 | 0.395 | 1.80 |
| 2  | 42.62 | **1.498** (+9.2%) | 0.754 | 0.359 | 0.394 | 1.83 |
| 4  | 43.20 | 1.506 | 0.763 | 0.358 | 0.393 | 1.82 |
| 8  | 42.93 | 1.514 | 0.756 | 0.361 | 0.390 | 1.82 |
| 16 | 43.55 | 1.544 | 0.755 | 0.360 | 0.393 | 1.83 |
| 32 | 43.26 | 1.519 | 0.754 | 0.359 | 0.391 | 1.83 |
| 64 | 43.76 | **1.557** (+13.5%) | 0.750 | 0.361 | 0.391 | 1.84 |

`min_x` rises ~9% at `m=2` and ~13.5% at `m=64` (per-seed `min_x_sd ≈ 0.06–0.11, n=3`, so the
`m=1 → m≥2` step is roughly 1–1.5σ and reproduces in every seed). Crucially **`Y` rises**
(39.1 → 43.8, +12%) and **`eta` is flat** (0.761 → 0.750). The gain is a single discrete step
at `m=2` and then saturates.

Per-class `x_min` (Fig 5): all three groups' floors rise together with `m_medium`
(gsm8k 1.74→1.90, cnn 1.37→1.56, math 2.21→2.40) — the pipeline as a whole flows better,
not just the boosted group.

Note: `cnn` *verifier-token* share barely moves (0.395→0.391) even as its *service* share and
`x_min` rise. The lever is not "hand cnn more verifier tokens" — it changes batch
ordering/composition so the low-`alpha` cnn requests (which have the lowest
`expected_useful_tokens`, hence lowest `batching_utility`, hence always lose ties at `m=1`)
stop being deferred and stalling the round-robin loop.

---

## 6. Figures

- `results/auc_achievable_region/fig1_point_cloud.png` — achievable `(x_min, Y)` cloud.
- `results/auc_achievable_region/fig2_upper_envelope.png` — empirical `Ŷ*(x)` step envelope.
- `results/auc_achievable_region/fig3_group_response.png` — target-group share vs its multiplier
  (flat for `hard_math`/`m_hard`; small positive step for `cnn`/`m_medium`).
- `results/auc_achievable_region/fig4_mechanism.png` — `min_x ↑`, `Y ↑`, `eta` flat vs `m_medium`.
- `results/auc_achievable_region/fig5_per_class.png` — per-class interactivity floor vs `m_medium`.

---

## 7. Outcome

**Outcome C** (`AUC_ACHIEVABLE_REGION_DIAGNOSTIC.md` §9): prioritising the true bottleneck class
raises `min_x` **without** a goodput cost (Y actually improves) — so the earlier flat
`Y_controller(x)` was, in part, the `adaptive_index` controller sitting on an *inefficient*
allocation inside the same capacity region, not a hard physical wall at ~0.9–1.4 tok/s.

But the reachable improvement is **small and saturating** (~10–13% in `min_x`, one step at
`m=2`), so the achievable region is only *mildly* non-rectangular at this scale
(N=21, `gamma_max=4`, this model pair). Sweep A separately rules out "insufficient verifier
priority for `hard_math`" as any part of the story.

Sweep B (2-D `m_medium × m_hard` grid) and the isolated single-group capacity test (§10) were
not needed: §10 is only triggered by Outcome B, and Sweep A already shows `hard_math` priority
is irrelevant.

---

## 8. Final answer

> **Is the near-rectangular AUC frontier a true property of the achievable region, or a
> failure of the current controller to move along it?**

**Both, with the controller the larger factor at this scale.** There exists a reachable
operating point about **10–13% higher in system `min_x` at equal-or-higher goodput** that the
current virtual-queue / index controller does not find — reached here by a static 2× priority
correction on the one class (`medium_cnn_summarize`) that actually sets the interactivity
floor. The `weighted_utility` ranking key `weight × expected_useful_tokens` structurally
under-weights low-acceptance clients — precisely the floor-setting group — and `Z_i` does not
compensate enough. So the frontier is **not** as rectangular as `Y_controller(x)` implied, and
the flat curve should **not** be read as evidence against the AUC formulation.

However, once that allocation inefficiency is removed the remaining curvature is shallow and
saturates immediately, so the achievable region here is still **close to** rectangular — the
AUC-vs-single-point advantage is real but modest at N=21 / `gamma_max=4`. Getting a
richer frontier needs a structural change (wider `gamma_choices`, larger N, or an operating
regime nearer the roofline knee), consistent with the Section 10 recommendations of the
frontier report.

**Recommended next step:** fix the controller allocation — make the batching weight (or `Z_i`
dynamics) compensate for per-client acceptance so low-`alpha` clients are not perpetually
deferred — then re-measure `Y_controller(x)` against this empirical `Ŷ*(x)` envelope.

Deliverables: `results/auc_achievable_region/{points_raw.csv, points_summary.csv,
frontier_points.csv, mechanism_summary.json, fig1..fig5.png}`;
`configs/auc_achievable_region_base.yaml`; `scripts/run_auc_achievable_region.{py,sh}`;
`scripts/plot_auc_achievable_region.py`.
