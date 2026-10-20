# AUCspec RQ1 pilot — first GPU gate (seed 1 only)

Descriptive evidence from **one run-seed**. No significance, no certified capacity, no N\*(x), no ICA, no policy comparison. The
pre-registered thresholds were not touched. Only the first seed job was submitted; nothing further is submitted.

## 1. What ran
| Item | Value |
|---|---|
| Code state | `4a7e86640c954a55e6d3042b1d3fbbc4e1cc1e50` (clean, pushed). Tests at that SHA on Setonix: 55 modules, all pass except the pre-existing `test_payload_interface` float-equality failure (identical at `aba9cec`); `test_capacity_script_e2e` now passes (4). Local Windows: 447 pass, the same 4 pre-existing Windows-only failures |
| Run directory | `RQ1_PILOT_4a7e866` (new, immutable, config hash `57814b92db0713a0`, `--planned-runs 18`); config regenerated identically from the generator; manifest sha256 matches the frozen identity record |
| Job | Slurm **51179457**, `gpu` partition, 1 node (`gpu:8`), `--time=01:04:00`; submitted after `budget_guard check` (0.000 + 1.067 ≤ 3.2) |
| Exact walltime / consumption | **00:31:23 = 0.523 node-h** (1883 s) → `results/rq1_pilot_seed1/budget_ledger.json` (`authorized_jobs: 1`) |
| Settings | runtime H (Triton FA 3.3.1 shim, V0 engine, `batch_wall`, price zero, fixed budget 100), γ = 2, FCFS, `population_seed` 20261020 / `run_seed` 1, identical static hardware (`draft_speed_multiplier` [1,1]), `isolated_profile` 22.2 ms/token, `causal_virtual_time` + ledger, warm-up 30 s, window 120 s, early stop +10 s, 64 questions per client |
| Order | N = 8 (gate), 1, 2, 4, 12, 16: all six `RUN_DONE exit=0`, no gate failure, `PILOT_JOB_DONE` |

Not identical across clients by design: channel phase/fading (stochastic) and the per-client prompts/acceptance process; only the static hardware parameters are identical.

## 2. Per-N results (seed 1, window 120 s, exact denominator)
| N | client rate min / mean / max | x_min | Y(N) | N·x_min | x_min/mean | verifier util. | batch wall mean / p90 / max (ms) | batch-size histogram (window) |
|---|---|---|---|---|---|---|---|---|
| 1 | 7.76 / 7.76 / 7.76 | 7.758 | 7.76 | 7.8 | 1.000 | 0.28 | 98 / 94 / 765 | {1: 342} |
| 2 | 7.66 / 7.70 / 7.74 | 7.658 | 15.40 | 15.3 | 0.995 | 0.29 | 104 / 125 / 440 | {2: 336} |
| 4 | 6.22 / 6.39 / 6.47 | 6.225 | 25.56 | 24.9 | 0.974 | 0.41 | 174 / 189 / 843 | {4: 283} |
| 8 | 3.98 / 4.09 / 4.17 | 3.975 | 32.73 | 31.8 | 0.972 | 0.62 | 419 / 983 / 1337 | {8: 179} |
| 12 | 4.03 / 4.13 / 4.22 | 4.033 | 49.59 | 48.4 | 0.976 | 0.63 | 415 / 490 / 752 | {12: 181} |
| 16 | 3.32 / 3.41 / 3.54 | 3.317 | 54.62 | 53.1 | 0.972 | 0.69 | 552 / 683 / 1226 | {16: 150} |

(All per-client rates are in `results/rq1_pilot_seed1/pilot_seed1_compact.json`.) Tokens per round ≈ 2.72–2.75 at every N; every finished
sample reached the 96-token cap, **0 EOS stops**.

## 3. Draft model, lifecycle, causality, time
| N | virtual draft ms/token | physical GPU draft ms/token (vs 22.2) | physical wall ms/token | min in-window rounds/client | last client virtual time (s) | ledger events / violations | wall / virtual s = ratio | mean round / queue wait (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 22.2 | 23.2 (+5 %) | 23.3 | 342 | 160.1 | 1837 / 0 | 81 / 160 = 0.51 | 351 / 150 |
| 2 | 22.2 | 35.1 (+58 %) | 35.2 | 336 | 160.1 | 2696 / 0 | 101 / 160 = 0.63 | 357 / 149 |
| 4 | 22.2 | 84.9 (+282 %) | 85.1 | 283 | 160.3 | 3794 / 0 | 171 / 160 = 1.06 | 425 / 148 |
| 8 | 22.2 | 136.1 (+513 %) | 147.4 | 179 | 160.4 | 4382 / 0 | 245 / 160 = 1.53 | 670 / 147 |
| 12 | 22.2 | 137.7 (+520 %) | 197.0 | 181 | 160.5 | 6304 / 0 | 320 / 161 = 1.99 | 663 / 145 |
| 16 | 22.2 | 145.7 (+556 %) | 253.5 | 150 | 160.4 | 6782 / 0 | 348 / 160 = 2.17 | 801 / 145 |

Workload lifecycle: 64 samples assigned per client, `client_work_complete` never true, no exhausted or unverifiable client; every client was stopped
by the early stop at ≈ 160 s virtual (a client consumes ≈ 13 of its 64 questions). All per-run gates (`pilot_gate.py`) passed: 0 causality
violations, no INVALID run, ≥ 20 in-window rounds, real/virtual ≤ 3 (max 2.17). Runtime: six V0 engine initialisations (v0.9.1); the runtime guard
(`require_triton_fa`) raises instead of falling back and none raised; the only "sdpa" text in the log is the **HF draft** model's warning, not the verifier.
Not audited separately this time: duplicate/dropped verifier requests beyond the ledger's `check_causality` (which does verify no duplicate or unserved
request), and the token-accounting identity (Y = Σ x_i is built from the same committed-token column).

## 4. Questions
1. **Does the corrected causal simulator run on real GPUs?** Yes: six runs up to N = 16, 0 violations over ≈ 25,800 ledger events, no deadlock,
   no exhaustion. Cost of causal coordination: real/virtual time 0.51 → 2.17.
2. **Does the isolated-profile model behave as intended?** Yes by construction (virtual 22.2 ms/token at every N). The N = 1 physical draft
   measurement is **23.2 ms/token, +4.5 %** from the profile (wall +5 %), so the profile is validated and was not re-tuned. Physical latency grows to
   **+556 %** at N = 16 (single-process contention) — the E1 artifact the profile removes. First-order effect had the confounded mode been used:
   adding ≈ 2 × (145.7 − 22.2) ≈ 250 ms to an 801 ms round at N = 16 would cut the rate by roughly a quarter; this is an estimate, not a measurement.
3. **Are workload and window adequate?** Yes: no exhaustion, ≥ 150 in-window rounds per client, spread across clients ≤ 3 % of the mean (`x_min/mean` ≥ 0.97).
   All samples hit the 96-token cap, so the 24.8-token tolerance was never approached.
4. **Should the pilot proceed to the remaining seed jobs?** Not automatically. Recommendation: yes, after one decision from you.
   * **Range.** Y(16)/Y(8) = **1.67**, x_min(16)/x_min(8) = **0.83** for this seed. Under the pre-registered rules a ratio ≥ 1.5 means "goodput still
     rising; extend N" (INCONCLUSIVE, not SUPPORTS); the script prints `INCONCLUSIVE(partial)` since only seed 1 exists. Verifier utilisation is 0.69 at N = 16:
     the system is not saturated. This neither supports nor falsifies RQ1; it says the tested range is too small. Thresholds unchanged.
   * **N = 8 anomaly (hypothesis, not verified).** 45 of 243 batches at N = 8 took > 600 ms (13–23 s and 40–44 s into the run, i.e. partly in warm-up, partly in the
     window) versus 3 of 242 at N = 12; N = 8 was the first run of the job (cold caches/JIT?). The median batch wall at N = 8 (287 ms) is below N = 12 (418 ms), yet
     x_min(8) ≈ x_min(12), so the N = 8 point looks depressed and the 8 → 16 ratios are biased in a direction I cannot quantify from one seed. Suggest a different
     N order in later seeds (e.g. 1, 8, …) and logging cold-start effects.
   * **Lockstep batching.** Every batch contained all N clients: homogeneous deterministic drafts keep the clients synchronised and the 150 ms wait gathers them.
     This is a property of the homogeneous design, not of real traffic, and must accompany any RQ1 statement.
   Options: (a) seeds 2–3 unchanged (≈ 0.52 node-h each) to complete the registered design, expecting INCONCLUSIVE again; (b) add N = 24 and 32 per seed (extends the
     registered N set — needs your sign-off, not a threshold change); (c) both.
5. **Is the 3.2 node-h forecast credible?** Yes. Actual 0.523 node-h vs forecast 0.52 (31 min at causal factor 1), even though real/virtual ratios at N ≥ 8
   (1.5–2.2) exceeded the assumed 1.0–1.6. Two more identical jobs bring the total to ≈ 1.57 node-h; worst case 0.523 + 2 × 1.067 = 2.66 ≤ 3.2. Adding N = 24 and 32 would
   cost roughly 0.2–0.25 node-h per seed (estimate from the measured ratios).

## 5. Limits of this evidence
One seed; descriptive; homogeneous hardware with lockstep batching; fixed γ = 2; no EOS-shortened answers observed; the isolated draft profile comes from the earlier
N = 1 run and was re-confirmed here (+4.5 %); the N = 8 batch-wall anomaly is unexplained. Nothing here changes the status of the unsupported claims in the C0/C1 audits.

Artifacts: `results/rq1_pilot_seed1/pilot_seed1_compact.json`, `results/rq1_pilot_seed1/budget_ledger.json`; raw CSV/ledger on
`/scratch/pawsey1257/quanla/canonical_raw/RQ1_PILOT_4a7e866/` (not in git).
