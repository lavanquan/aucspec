# AUCspec C3 — Minimal RQ3 GPU plan and GO/NO-GO (nothing authorized, nothing submitted)

## 0. Decision

**GPU GO/NO-GO: NO-GO.** No GPU work is requested, authorized or submitted by C3. Reasons, strictly:
1. **No mechanism has been shown to help in the regime the GPU can currently reach.** In the measured RQ1 region (B_max = 16, N ≤ 32) the batch cap does not bind (0–1 % of batches), so
   deficit-aware ranking is vacuous (CIs span 0); adaptive depth is identical to constant γ=4 at the assumed cost and harmful under mis-specification (−42 % goodput worst case).
2. The only positive CPU result — deficit-first ranking raises the floor (+37 %, capacity at 2 tok/s 19.2 → 25.6 clients) for ≈ −4 % goodput — appears **only** for heterogeneous populations with a
   binding cap (B_max = 8, or N ≥ 48 at B_max = 16), is model-derived, and uses a baseline (work-conserving FCFS) that the emulator's legacy rule does not provide.
3. The remaining budget (1.474 node-h) buys at most ~6 N≥24 runs: **no statistics, no population variation**.
4. Open prerequisites (§5) are CPU-side and not done.

A GPU step becomes **GO** only if all of: (a) the user authorizes budget and scope in writing; (b) prerequisites §5 are done and reviewed; (c) the pre-registered analysis §4 is committed *before* any run;
(d) the decision rule for depth (§3.1) is accepted. Until then the paper's RQ1 stays a descriptive measurement and the controller claim stays unsupported.

## 1. What a GPU pilot could legitimately test (and what not)

| question | testable on the emulator (real verifier, virtual clients) | note |
|---|---|---|
| does the verifier wall time follow `T = a + b·ctx + c_req·B + b_γ·γ`? | **yes**, a micro-benchmark outside the emulator | the only way to settle adaptive depth; **not authorized**: C2 forbids new calibration and C3 keeps b_γ "assumed" |
| does deficit-first ranking raise the floor when the cap binds? | **yes**, scheduler-only comparison with equal information and the same physical constraints | model prediction: x_min +0.6 tok/s (+37 %) at N=24, B_max=8 |
| does adaptive depth help? | no (inert unless b_γ ≳ 1–5 ms/token, unmeasured) | gate on §3.1 |
| does work-conserving formation matter on hardware? | it is an emulator/serving semantic, not a controller claim | report as baseline definition |
| certified capacity / frontier | **no** | needs ≫ budget |

## 2. Arms (all: runtime H, V1 greedy, `batch_wall`, same population/prompts/phys. constraints, oracle α̂ for every arm)

| arm | depth | scheduler | purpose |
|---|---|---|---|
| R | γ=4 | legacy `fcfs` (head+W) | historical reference only |
| **B** | γ=4 | `c3_fcfs` (work-conserving FCFS, constraints C) | **fair baseline** |
| **T** | γ=4 | `c3_deficit` deficit-first | treatment (mechanism A isolated: depth is constant, ranking is the only difference to B) |

γ=2 arms add nothing (γ=4 dominates by 16–20 clients in the CPU model and in the verifier-ms/useful-token analysis); adaptive-depth arms are excluded until §3.1 passes.
Equal information: identical α̂, identical x_req and V, identical caps (B_max=8, M=4096), identical population and prompt block per client, identical early-stop/lifecycle protocol.

## 3. Isolation of effects and decision rules

* **Scheduling vs work conservation:** R→B isolates formation; B→T isolates ranking.
* **Depth:** γ=4 constant in B and T, so no depth effect can leak into the scheduler comparison.
* **3.1 Depth fate (pre-registered, separate, optional micro-benchmark, requires its own authorization):** measure verifier wall for fixed batch membership at γ ∈ {0,1,2,4} (≥ 30 repetitions, batch sizes 4/8/16, contexts 100/200/400).
  Estimate b_γ with a block bootstrap over repetitions. Rule: if the 95 % upper bound of b_γ < 1 ms/token ⇒ **adaptive depth is closed** (remove from the paper); if the lower bound ≥ 1 ms/token ⇒ re-open mechanism B with the *measured* value
  (the CPU study shows a mis-assumed value is the dominant risk). Estimated cost: tens of minutes of one GPU (~0.1 node-h), no emulator.
* Population: heterogeneous, **asynchronous** — per-client start offsets (`clients.start_offset_ms_max`) and non-negative arrival jitter (`clients.arrival_jitter_ms_max`), both implemented opt-in in the real simulator
  (default 0 ⇒ historical results unchanged); never mixed with the historical lock-step runs.

## 4. Pre-registered analysis and statistical interpretation

* Primary observable at fixed (N=24, B_max=8, floor x_req=3): **x_min** and **Jain index**, plus goodput Y as the cost; secondary: queue wait max, skips. No frontier, no sweep.
* CPU-model expectation (for calibration of expectations only, not a threshold to tune): T−B Δx_min ≈ +0.6 tok/s, ΔY ≈ −2.8 tok/s.
* Decision rule: T "supports" mechanism A iff, in **every** run-seed, x_min(T) − x_min(B) ≥ +15 % of x_min(B) **and** Y(T) ≥ 0.9·Y(B) **and** the ledger/lifecycle/validity checks pass. Otherwise "not supported".
* **Interpretation hierarchy.** Sampling units are *populations* (prompt-block assignments) ⊃ run-seeds. With greedy verification, run-seeds mostly repeat the same trajectory (variation = wall-time noise only); the population is the dominant random effect.
  A pilot with one population is a **pilot observation / case study**: it can falsify (no floor gain) but cannot certify capacity or generalize to other populations. Certification would need ≥ 5 independent populations × arms × an N sweep
  (≈ 5×2×3 = 30 runs ≈ 6–7 node-h) with population as the random effect and a paired analysis per population — a separate budget.
* Exploratory validity protocol: N ≥ 24 runs fail the *original* real/virtual gate (3.07–3.28 observed in C1); results must carry the `exploratory_v2` label, exactly as in C1.4, and the original verdict must be quoted.

## 5. Prerequisites before any GO (all CPU)

1. Review of C3 branch; decision on keeping `c3_*` opt-in code in the canonical tree. 2. A config + launcher for the pilot arms (no canonical-policy change). 3. A unit test that the real simulator and the harness agree on offsets/jitter (partly done: knobs exist, ledger clean).
4. The analysis script of §4 committed with a frozen decision rule. 5. Confirmation that `max_batch_size=8` with `max_num_batched_tokens=4096` and chunked prefill off is memory-safe (it uses less memory than the cap-16 runs that completed, but this is untested).
6. The inherited `test_payload_interface` float-precision failure fixed or explicitly waived.

## 6. Budget (estimates from C1.3/C1.4 runs: ≈ 0.18–0.25 node-h per N ≥ 24 run; remaining 1.474 node-h; **not requested**)

| block | runs | node-h | status |
|---|---|---|---|
| P0: verifier γ micro-benchmark (§3.1) | – | ≈ 0.1 | optional, separate authorization |
| P1: arms B, T (+R) at N=24, B_max=8, 1 population, 2 run-seeds | 4 (6 with R) | 0.7–1.0 (1.1–1.5) | pilot observation only |
| certification (5 populations × 3 N × arms B,T) | ≈ 30 | ≈ 6–7 | exceeds budget; future |

Budget is committed only after §0 criteria (a)–(d). The smallest policy-comparison pilot (P1, 4 runs) costs ≈ 0.7–1.0 node-h, i.e. ≈ 50–70 % of the remaining budget, and yields no certified claim.

## 7. What the paper can say today without any GPU

RQ1 as a descriptive, single-seed characterization of interactivity–concurrency capacity (with its gate verdicts); a causal-virtual-time methodology; and, from C3, the negative/conditional findings:
the batch cap does not bind in the measured region, adaptive depth is inert at the assumed cost, work-conserving formation is a 10–15 % baseline effect under asynchrony, and deficit-first scheduling is a conditional,
model-derived interactivity-for-throughput exchange awaiting hardware evidence.
