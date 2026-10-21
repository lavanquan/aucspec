# AUCspec C4-A — Final CPU falsification of the deficit-aware scheduling benefit

Branch `research/c4-paper-viability` (base `research/c3-1-physical-feasibility` @ `e2e4eee`). CPU only; no new controller feature (the ranks below are diagnostic modes of the existing selector, not a tuned scheduler); no GPU, no pricing, no main merge.
All numbers are **model-derived** (real simulator/queue/batcher code, CPU verifier stand-in), 1,840 runs, 0 ledger violations, 0 invalid, 0 empty batches, 0 cap or token-budget violations. Raw output on scratch; compact results `results/c4/scheduler_falsification_summary.json`; code `scripts/c4/falsification_analysis.py`, `scripts/c3/mechanism_validation.py c4`.

## 1. Question and design

Is the small deficit-first benefit found in C3.1 (+3–5 % x_min at N=24, B_max=8) caused by **waiting age** (a standard FCFS-like effect) or by the **SLO-deficit state** Z_i?
Four ranks on the *identical* work-conserving path, constraints (B_max, 4096 processed tokens, `max_skips = 2`), depth and populations:

| label | rank | what it isolates |
|---|---|---|
| FCFS | arrival order (`c3_fcfs`) | the fair baseline |
| AGE | oldest **decision** first (`age`) | waiting age only |
| ZONLY | the client's genuine stamped Z_i (`z_only`) | the SLO-deficit state only |
| ZEFF | Z_i + x_req · waiting-age (`deficit_fcfs`, the C3 effective deficit) | the C3 treatment (α-free) |

Populations: the **grounded** families of C3.1 only (`measured`, `measured+jitter`; α_i ~ N(0.930, 0.017) clipped to the measured 0.88–0.96; prompts 85–170 tokens; homogeneous hardware) — acceptance distributions were **not** broadened. 20 independent populations per cell (population = sampling unit; one matched paired run per rank), γ fixed.
Cells: negative controls (N=24, 32; B_max=16), constrained (N=24, 32; B_max=8), beyond-region (N=48, B_max=16). Primary γ=2 (the GPU-validated depth), replicate γ=4. **Predeclared** SLO-floor sensitivity x_req ∈ {2, 3, 4} tok/s (γ=2; cells (24,16), (24,8), (32,8), (48,16)), fixed before the run.

## 2. Result 1 — age-only does not reproduce the benefit; it is indistinguishable from FCFS here

Paired differences vs FCFS (γ=2, x_req=3, measured family, 20 populations; mean [95 % t-interval over populations]):

| cell | FCFS x_min | AGE Δx_min | ZONLY Δx_min | ZEFF Δx_min | Δgoodput (ZEFF) |
|---|---|---|---|---|---|
| N=24, B_max=8 *(constrained)* | 2.76 | −0.001 [−0.008, +0.006] | **+0.137 [+0.123, +0.151] (+5.0 %)** | **+0.142 [+0.129, +0.155] (+5.2 %)** | +0.03 % |
| N=32, B_max=8 | 2.03 | 0.000 | +0.001 | 0.000 | −0.06 % |
| N=24, B_max=16 *(negative control)* | 2.91 | +0.003 | −0.002 | +0.007 [+0.001, +0.014] | +0.21 % |
| N=32, B_max=16 *(negative control)* | 2.22 | +0.001 | +0.001 | 0.000 | −0.02 % |
| N=48, B_max=16 *(outside validated region)* | 1.43 | +0.002 | +0.107 (+7.5 %) | +0.113 (+7.9 %) | +0.10 % |

γ=4 replicate: ZONLY +4.6 % / ZEFF +3.4 % at (24,8); +12.8 % / +13.2 % at (48,16); AGE ≈ 0. γ=2 with arrival jitter: ZEFF +5.0 % at (24,8), +7.6 % at (48,16).

**Why AGE ≡ FCFS in these families:** with identical rtt/draft configuration the order of *decision* times equals the order of *arrival* times; in the binding batches of the FCFS runs the AGE selection differs from FCFS in **0.2 %** of them (0.0–1 % across cells). The experiment therefore shows (i) that waiting age cannot be the explanation (it is the same ranking as the baseline), and (ii) that the experiment *cannot* separate "age" from "arrival order" — that would need clients whose decision-to-arrival delay differs (heterogeneous hardware), which is **not** in the grounded families and was deliberately not added.

## 3. Result 2 — attribution of the ranking changes: Z, not age

On the *FCFS runs' own eligible sets* (binding batches only; same sets for all ranks), N=24, B_max=8, x_req=3:

| quantity | value |
|---|---|
| binding batches | ≈ 505 per run (99 % of all batches) |
| ZEFF ≠ FCFS (member set differs) | **97 %** |
| AGE ≠ FCFS | **0.2 %** |
| ZONLY ≠ FCFS | 98.7 % |
| among the batches where ZEFF changed the FCFS choice: reproduced by AGE alone | **0.2 %** |
| … reproduced by ZONLY alone | **65 %** |
| … reproduced by neither (needs the Z+age combination) | 35 % |
| batches with any positive stamped Z | 99 % |

At x_req=2 (where FCFS already satisfies everyone) only 50 % of batches change and 14 % are reproduced by Z alone; at (48,16) 18 % by Z alone, 81 % need the combination. So the fraction of ranking changes attributable to **Z rather than age** is ≈ 100 % (age alone explains 0.2 %); the part that needs Z *and* the waiting-age term is 35 % (N=24, B_max=8) to 81 % (N=48, B_max=16).

## 4. Result 3 — an increase of x_min is not an increase in SLO satisfaction

| cell (γ=2, measured) | x_req | FCFS clients below SLO (of N) | ZEFF Δ(clients below SLO) | populations in which *every* client meets the SLO: FCFS → ZEFF |
|---|---|---|---|---|
| N=24, B_max=8 | 2 | 0.0 | 0.00 | 100 % → 100 % |
| N=24, B_max=8 | **3** | 18.4 | **+1.75 (worse)** [fewer in 15 % of populations, more in 50 %] | 0 % → 5 % |
| N=24, B_max=8 | 4 | 24.0 | 0.00 | 0 % → 0 % |
| N=32, B_max=8 | 3 | 32.0 | 0.00 | 0 % → 0 % |
| N=48, B_max=16 | 3 | 48.0 | 0.00 | 0 % → 0 % |

The +5 % in x_min comes from **flattening the rates around the SLO**, not from lifting clients over it: at x_req=3 the baseline has 5.6 of 24 clients meeting 3 tok/s, ZEFF has 3.8 — it raises the *worst* client from 2.76 to 2.90 at the expense of clients that were at or just above 3.0. At x_req=2 and 4 the *set of satisfied clients* is unchanged. Queueing: mean wait unchanged (±1 ms), **maximum wait +396 ms** (FCFS minimises the tail); skipped-service share −0.18 (ZEFF) / −0.44 (ZONLY) in absolute fraction of served requests; goodput ±0.1 %; Jain +0.0007 (baseline already ≈ 0.998).

## 5. Verdict

1. **Age-only does not account for the benefit** (it is identical to FCFS in the grounded families and yields 0.00 %). The benefit is carried by the deficit state: the stamped Z_i alone reproduces +5.0 % of the +5.2 %.
2. **That does not make it a research contribution.** Ranking by the client's accumulated shortfall is the least-attained-service / deficit-round-robin / virtual-token-counter family; whether the Lindley deficit adds anything beyond those is a literature question (see `AUCSPEC_C4_MINIMUM_PUBLISHABLE_EVIDENCE.md`) not a result here. The effect is +3–5 % at **one** of two constrained cells, **zero** at the other and at both negative controls, +8–13 % only outside the validated region, it does **not** increase the number of clients that meet an SLO (it decreases it at x_req=3), and it worsens the tail wait by ≈ 400 ms.
3. **Recommendation:** do **not** present deficit-aware scheduling as a novel mechanism or a headline contribution. Keep it, if at all, as (a) a documented small rate-equalisation effect of a standard fair-share rule in the constrained cell and (b) the reason the baseline must be work-conserving FCFS. The age explanation is rejected only in the narrow, trivial sense above; the decisive grounds for dropping it are magnitude, regime, absence of SLO-satisfaction gain and standardness.
4. No scheduler was tuned: `max_skips`, V, caps, ranks and sensitivity values were fixed before running; the SLO-floor sweep was predeclared; the acceptance distribution was not widened.

## 6. Limits

Model-derived (affine verifier; acceptance geometric beyond position 2); one catalog family; homogeneous hardware (so age ≡ arrival); 20 populations × one run-seed per cell; the negative-control rows differ from FCFS only through the first batch (one-off ordering) and are within noise.
