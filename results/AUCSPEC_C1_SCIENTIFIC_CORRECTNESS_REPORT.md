# AUCspec Task C1 — scientific correctness repairs before RQ1

Branch `research/c1-scientific-correctness` (from `research/c0-scientific-readiness` @ `aba9cec`). CPU only: no GPU job was
submitted, no recalibration, no pricing, no frontier, no merge. The frozen runtime H (`V1-batchwall-tritonfa33`), V1 greedy
semantics and tie-break, models, calibrated pricing, controller objective/heuristics, baselines and all historical raw
results are unchanged. Every new behaviour is behind a config switch and **off by default**, so historical results stay
readable and keep their labels ("span-heuristic validated", "population redrawn per seed", "real-order batching").

Verdict in one paragraph. The measurement pipeline and the virtual-time core are now repaired and tested on CPU, **and the repair
exposed one more blocker that C0 had not isolated** (E1 below: the emulator inflates draft latency 6.6× at N=8, so per-client
capacity at N>7 measured with `shared_gpu_emulation` is partly an emulator artifact). A mitigation exists behind a flag but needs
your decision. Recommendation: **conditional GO for the descriptive RQ1 pilot only** (section 8); **NO-GO** for any certified
N\*(x), frontier, ICA, RQ3 or RQ4 run.

## 1. What changed (commits, small and separated)

| Commit | Content | Audit items |
|---|---|---|
| measurement | `workload_lifecycle.py`, lifecycle columns in the simulator, `population_seeds.py`, INVALID in `CandidateMeasurement`, cache schema v3, tests | M1, M2, M4 |
| causality | `docs/VIRTUAL_TIME_EVENT_GRAPH.md`, `virtual_causality.py`, causal batcher loop, controller snapshot history, simulator flag, tests | D1, D2 (D3 documented) |
| statistics | `capacity_certified.py` (Student-t classifier, boundary/ICA/dual logic), driver rewired to it, Monte-Carlo and before/after scripts, tests | M3, M5, M6, M7, M8, M9 |
| pilot | draft-timing `isolated_profile`, early stop, pilot config + budget scripts, revised plan | E1, M12 |
| reports | this report and `AUCSPEC_RQ1_PILOT_REVISED.md`, results | – |

## 2. Phase 1 — measurement integrity

### M1/M2 workload exhaustion (was HIGH)
* **Definition** (`workload_lifecycle.py`, virtual time): each round record carries `client_samples_assigned`, `sample_finished`,
  `client_samples_pending`, `client_work_complete`. A client is **EXHAUSTED iff its last assigned sample completed strictly before
  `measurement_end_ms`**; a round straddling the end, completion exactly at the end, sparse rounds, zero in-window tokens with
  work left, and a client stopped by the optional early stop are all ACTIVE. Legacy rows without the columns are UNKNOWN (not
  exhausted) unless an explicit `expected_samples_per_client` is given, in which case the result is labelled "inferred".
* **INVALID ≠ INFEASIBLE.** `CandidateMeasurement.invalid`; both legacy classifiers return `invalid`; the certified classifier,
  search, area and dual functions propagate it (an INVALID probe aborts the search; any INVALID point makes the area `INVALID`
  with no number); the driver stops seed escalation at the first INVALID run; the candidate cache is schema v3 so measurements
  validated with the old heuristic are rejected, not silently reused.
* **Tests** (`test_workload_lifecycle.py`, 16): sparse ongoing rounds (old heuristic mis-fires, new does not), genuine early
  exhaustion, completion at `end−1 ms / end / end+1 ms` (EXHAUSTED / ACTIVE / ACTIVE), in-flight request at the end, zero-token
  client with pending work, client without rows, zero assigned samples, work finished before warm-up, no window, INVALID
  propagation through both legacy classifiers, and the recorded **B6 smoke pattern**.
* **B6 re-check** (real recorded rows, 826 rounds): the span heuristic flagged **5 of 8** clients (the INVALID verdict);
  the lifecycle assessment gives **8/8 ACTIVE** (clients ran 107–128 virtual s, all with a round in flight at the window end).
  The recorded verdict and numbers are left exactly as recorded; only the interpretation is corrected
  (`CANONICAL_QUALIFICATION_ERRATUM.md`). The 20 s-window rates stay non-citable (about 11 rounds per client).

### M4 fixed nested populations (was HIGH)
* `simulation.population_seed` selects a master population built from per-client sub-generators
  `Random(sha256(population_seed, i))`: client *i* depends on nothing but `(clients config, population_seed, i)`; population(N)
  is the exact prefix of the master. `simulation.seed` is now only the **run-seed** (controller randomness, scheduler ties,
  channel block fading, target sampling). Prompt blocks are the existing nested blocks (the dataset shuffle seed is
  `dataset.seed`, independent of the run-seed). Round records carry `population_seed`, `run_seed`, `population_fingerprint`
  (no run-seed, no policy inside).
* Without `population_seed` the old code path is untouched (historical results are *not* reinterpreted).
* **Before/after** (`results/c1_20261020/before_after_traces.json`): legacy client 0 RTT across run-seeds 1–5 =
  79.9 / 19.8 / 46.5 / 48.7 / 82.6 ms; new = 51.7 ms for every run-seed, identical `fingerprint_N8`.
* **Tests** (`test_population_seeds.py` 6, `test_simulator_causal_wiring.py` through the real `_make_clients`): identity across N
  (prefix), run-seeds, and policy; prompt assignment inside the fingerprint; degenerate ranges give a homogeneous population.

## 3. Phase 2 — virtual-time causality

Event graph, the two leaks and the repair are in `docs/VIRTUAL_TIME_EVENT_GRAPH.md`. Summary:

* **D1** a batch formed at cutoff *C = a_head + W* must contain every request with arrival ≤ *C*. Repair: every client carries a
  *lookahead lower bound* on its next arrival (start `RTT/2`; at a decision at *t*: `t+RTT/2`; after a result at *f*: `f+RTT`);
  the batcher forms a batch only when all clients without a queued request have lower bound `> C` (strict, so a request
  arriving exactly at *C* is included), re-reading the head after draining new submissions.
* **D2** a decision at virtual time *t* must not see a server-state update with timestamp > *t*. Repair: the verifier publishes
  each completed batch to the controller *in batch order before waking any client*; the controller keeps a snapshot history and
  a decision reads `server_state_at(t)`; a decision proceeds only when the verifier horizon
  `H = min(start of batch in service, max(server_available, min(head arrival, min lb) + W)) > t`.
* **No sleep, no real-time delay**: waiters are released by state changes only (a test asserts the absence of `sleep` in the
  causal code). **Deadlock**: the client with the smallest decision time is never blocked (`H ≥ t + RTT/2 + W`); `RTT=W=0`
  (no lookahead) is rejected; a detector raises `CausalityDeadlock` if every active client is blocked (tested).
* **Ledger** records virtual timestamp, type, producer, decision timestamp, latest visible state update, queue-state version
  for arrivals, batch formation/completion and decisions; `check_causality()` verifies D1, D2, non-overlapping service, no
  duplicate or dropped request; the simulator writes `causality_ledger.jsonl` and `causality_check.json`.
* **Config**: `simulation.causal_virtual_time` (default false = historical real-order) and `simulation.causality_ledger`
  (record in either mode). Batch selection, `capacity_dpp_score/value`, knapsack, acceptance, V1 greedy and physical
  `batch_wall` accounting are untouched; the only ordering change outside the flag is a deterministic tie-break
  (arrival, client id, request id) in the waiting list.
* **Before/after** (30 random scenarios, random real delays, same virtual input): real-order batching violates D1 in
  **29/30 scenarios (250 violations)** and reproduces the sequential virtual-time reference in **1/30**; causal mode: **0
  violations and 30/30** identical to the reference. D2 targeted scenario (a decision at virtual *t=0* whose real evaluation follows two completed
  batches): legacy decision saw a state update at **200 ms** and state version 2 where 0 batches had finished by then; causal mode: no
  violation, maximum (visible update − decision time) = 0 (`causality_d2_targeted` in `before_after_traces.json`).
* **Tests** (`test_virtual_causality.py` 28): permuted real completion order → identical batches, receive times and ledger;
  arrivals before/at/after the cutoff; batches while other drafts are in flight; tied timestamps; future completions invisible;
  non-overlap; no drop/duplicate; deterministic replay; 12 random scenarios against an independent sequential reference;
  lookahead-violation and deadlock guards; the decision must block until a batch finishing before it has been processed, and the
  ledger flags a stale decision (version completeness). **Mutation checks**: making the cutoff watermark always true fails 18 tests;
  removing the horizon wait fails 5 (random-order property tests, deadlock, blocking test) — the first version of the D2 test did
  *not* catch that mutant, which is why the version-completeness check and the blocking test were added.
* **Not fixed (D3, limitation)**: `_allocate_shared_bandwidth` and `set_active_client_signal` read other clients' current state
  in real order. In the qualified regime the payload is ~100 B (upload 0.03 ms of a ~1.2 s round) and the pilot uses the
  `proportional` allocator and a fixed policy, so D3 cannot move a pilot number by more than ~0.03 ms/round; it must be fixed
  before any RQ4 bandwidth ablation. `roofline_knee` budgets are rejected in causal mode.
* **Residual risk**: causal mode throttles real-time parallelism (clients wait for the horizon); the real/virtual time ratio on
  GPU is unmeasured. The pilot budget carries a factor for it (section 5).

## 4. Phase 3 — statistical certification and boundary accounting

### M3 classifier (was HIGH)
`classify_certified`: per client *K* run-seed rates → Bonferroni Student-t bounds with `alpha_c = delta_eff / N`; one-sided
quantile `t_{1−alpha_c, K−1}` (FEASIBLE uses only the lower bounds, INFEASIBLE only the upper bounds, each at total error
`delta_eff`; `sided="two"` is the more conservative variant); **minimum certification size K_min = 5**, UNCERTAIN below it (the
screening hint is logged, never used as a verdict); a per-client sd floor of 1 % of x prevents zero-width intervals from
near-deterministic runs; optional alpha-spending over sequential looks (`n_looks`, = 3 for K=5…7) and boundary probes
(`n_probes`). The percentile bootstrap remains only as `bootstrap_sensitivity` (explicitly `certificate: False`).
**Assumptions stated in the module**: A1 independent run-seeds *given a fixed population*, A2 approximately normal per-client
means, A3 windows long enough for edge-round noise to be small. No unconditional 95 % claim is made.

Operating characteristics (N=10, weak client exactly at *x*, sd = 10 % of *x*, 500 trials; `classifier_operating_characteristics.json`):

| K | normal: false-FEASIBLE t-bound / bootstrap | t(3) heavy tail t / bootstrap | lognormal t / bootstrap | power to certify FEASIBLE at +1.67 σ (t-bound / bootstrap, normal) |
|---|---|---|---|---|
| 3 | 0.2 % / 12.0 % | 0.2 % / 10.2 % | 0.0 % / 6.0 % | 4 % / 88 % |
| 5 | 0.2 % / 7.2 % | 0.2 % / 5.4 % | 0.0 % / 3.0 % | 41 % / 92 % |
| 7 | 0.6 % / 3.0 % | 0.4 % / 3.6 % | 0.0 % / 0.0 % | 73 % / 95 % |
| 12 | 0.6 % / 1.2 % | 0.2 % / 0.6 % | 0.0 % / 0.0 % | 98 % / 99 % |

Reading: the t-bound keeps false-FEASIBLE ≤ 0.6 % in every tested distribution (well inside δ = 5 %), the bootstrap does not for
K ≤ 7, and the price is power — **certifying a boundary point needs K ≈ 8–12 seeds unless the between-seed spread is much
smaller than 10 % of *x*** (the pilot measures that spread). Sequential 5→7 looks: 0.6 % (normal) / 0.2 % (t3) without spending,
0.4 % / 0.4 % with it; the family-wise error over *M* independent probes is `1−(1−p)^M` (1.0 % for M=5, 2.0 % for M=10 at the
normal K=5 rate). The driver sets `n_looks`/`n_probes` from the realised plan. False-INFEASIBLE (weak client 1.67σ above *x*) was 0 in all tested cells for both procedures.

### M5/M7 boundaries and what a bound means (was HIGH)
`certified_n_search` / `summarize_probes`: only **certified** points are endpoints; UNCERTAIN/INVALID are never bounds.
* `lower_bound_n` = largest certified-feasible N — an **unconditional** lower bound on controller-achieved capacity.
* `upper_bound_unconditional` exists only if every N in (lower, cap] is certified infeasible.
* `upper_bound_monotone` = (smallest certified-infeasible N) − 1 is valid **only under assumption M** (this controller's
  feasibility is down-closed in N — existential regions are monotone, controller-achieved feasibility need not be). The
  probes can falsify M: a certified-feasible N above a certified-infeasible N withdraws the conditional bound
  (`nonmonotone_unresolved`, tested with a counterexample), and the driver adds falsification probes (N\*+2, N\*+3, 1.5 N\*).
* Gray zones: at an UNCERTAIN midpoint the neighbours inside the open bracket are probed; if the whole bracket is gray the
  result is `unresolved_upper`; an exact boundary is only `exact_conditional` (adjacent certified points **and** assumption M).
* Before/after on the same table (N=1–3 feasible, 4–6 uncertain, ≥7 infeasible): legacy returns upper = 4 (the uncertain
  point); new returns lower 3, conditional upper 6, gray {4,5,6}, status `bracketed`.

### M6 ICA staircase (was HIGH)
`area_from_boundaries`: lower integral uses the value at the *right* end of each grid cell, upper the *left* end (with
`min`/`max` against the neighbour if monotonicity in *x* is violated, reported); points with an unresolved upper bound
contribute `n_ref`; INVALID → no number; the grid must cover the window; the gap is split into discretisation and boundary
uncertainty. Before/after for N\*(x)=⌊24/x⌋ on 7 grid points: true area 54.6; legacy bounds [76, 76] (**lower bound above the
truth**); new [48, 76] with discretisation gap 28 and boundary-uncertainty gap 0.

### M8 dual consistency
`dual_consistency` checks pairwise interval compatibility of N\*(x) and x\*(N) and overlap of the two area intervals; exact
identities are asserted **only** when every point is exact, no gray/invalid/violations exist and the x\*(N) grid is complete
(otherwise `exact_identities_asserted: false`). Tested on a synthetic monotone region (consistent, exact asserted), on an injected
contradiction (detected) and on a gray point (identities not asserted).

### M9 monotonicity and sequential testing
Falsification probes are in the driver; repeated sequential testing is handled by alpha-spending options and reported in the
output (`delta`, `delta_effective`, quantile); both default to the realised number of looks, `n_probes_budget` defaults to 1 and
must be set for a family-wise statement over a whole search.

## 5. New blocker found while re-costing the pilot — E1: draft-emulator contention

Measured on the recorded smokes (`before_after_traces.json` inputs): isolated single-client drafting costs **22 ms/token**
(N=1: γ=1…4 → 22.0, 43.8, 66.0, ≈90 ms, p50≈p90). In the N=8 tfa33 smoke the same quantity is **146 ms/token** (γ=2: 207 ms,
γ=4: 509 ms, p90 845 ms) because all clients' HF drafts share one Python process (GIL and launch contention), plus worker-lock
queueing for clients sharing a GPU (N>7). In `shared_gpu_emulation` this **measured** latency becomes the virtual edge draft
latency, so mean draft time per round is 299 ms of a 1178 ms round at N=8 and grows with N. Consequence: a per-client rate
curve x(N) measured this way mixes verifier contention (the physics we want) with emulator contention (an artifact), exactly
the quantity RQ1 interprets. Draft cost on a real edge device does not depend on how many other devices exist.

Mitigation implemented behind a flag (default off): `simulation.draft_execution_mode: isolated_profile` with
`simulation.draft_profile: {overhead_ms, per_token_ms}`; the virtual draft latency is `fixed + multiplier · profile(γ)` and ignores
the measured contention (measured values are still logged). This changes what is *virtual*, not the frozen runtime, models or
verifier. It needs your explicit decision (section 8) because it is a modelling choice; the profile (22.2 ms/token) comes from the
N=1 recorded run and is re-measured by the pilot's N=1 run. Tests: independence from N and from injected real delays.

## 6. Test evidence

| Suite | Result |
|---|---|
| local (Windows, no torch): `legacy/tests` + `tests` excluding the torch-only modules | see section 9 |
| Setonix CPU job (`work` partition, pytorch module, no GPU): full `legacy/tests` + `tests` incl. simulator wiring | see section 9 |
| pre-existing failures unrelated to C1 (also present at `aba9cec`): 3 × `test_runtime_guard` (Windows paths/env), `test_batch_wall_fallback_when_target_does_not_stamp` | inherited, not touched |

New tests (106): workload lifecycle 16, populations 6, causality 28, certified capacity 32, driver end-to-end 4, simulator wiring 8
(real `_make_clients/_run_client/_run_round` code with stub target/draft workers, no torch needed), pilot config 5, pilot analysis 7, draft timing +2. Intent checks beyond "passes": a mutations of the cutoff watermark (18 failures) and of the horizon wait (5 failures);
legacy code is shown to violate each invariant on the same inputs (section 3, 4 before/after); property tests compare the
asyncio implementation with an independent sequential reference over random real orders.

## 7. Remaining scientific blockers and unsupported claims

| ID | Blocker / claim | Status after C1 | Blocks |
|---|---|---|---|
| E1 | Draft-emulator contention confounds x(N) | mitigation implemented, **needs decision**; profile re-measured by pilot N=1 | RQ1 interpretation, everything downstream |
| B1 | No valid capacity evidence on the corrected runtime | open — needs the pilot (GPU) | RQ1–RQ5 |
| B2 | Causal mode never run with real GPU concurrency (real/virtual ratio, throttling) | CPU-proved; **first GPU job is the test** | pilot cost |
| B3 | Certification power: K≈8–12 seeds per boundary point at 10 % between-seed spread | quantified; true spread unknown until pilot | cost of any certified N\*(x) |
| B4 | Monotonicity of controller-achieved feasibility in N and x | assumption M explicit, falsification probes added; never proven | upper bounds, ICA upper |
| B5 | D3 (bandwidth/active-client state read in real order) | documented limitation | RQ4 bandwidth ablation |
| B6 | Server price 0; knapsack/bandwidth inactive; depth driven by an ill-scaled device-queue price | unchanged (out of C1 scope) | RQ3/RQ4 |
| B7 | Verifier calibration on the corrected runtime absent; loaded-vs-isolated gap (+86 ms/batch) unexplained | unchanged | RQ2, RQ3 |
| B8 | EOS-shortened answers can still exhaust a client | detected (INVALID), budgeted (64 questions), not prevented | pilot validity per point |
| U1 | "measured roofline knee", "physical verifier-token efficiency", "asynchronous DPP guarantees", "existential optimal capacity from one controller", "certified boundaries from invalid workloads" | remain unsupported (C0 list), nothing in C1 supports them | manuscript claims |

## 8. GPU GO / NO-GO

**NO-GO**: certified N\*(x), x\*(N), ICA, frontier, RQ3/RQ4 runs, calibration, anything with pricing — blockers B3–B7.

**Conditional GO — descriptive RQ1 pilot** (`AUCSPEC_RQ1_PILOT_REVISED.md`), only if all hold:
1. you approve the draft-timing choice for E1 (`isolated_profile`, recommended) — or explicitly accept the confounded
   `shared_gpu_emulation` and restrict the pilot's claim to the verifier-attributable part by decomposition;
2. the Setonix CPU test job in section 9 is green;
3. you accept the budget cap in the pilot plan (expected 1.6 node-h, hard cap 3.2, first job is a gate with abort criteria);
4. the run directory is created by the existing canonical launcher with `causal_virtual_time: true` and ledger violations = 0
   as a per-run validity condition.
Nothing has been submitted.

## 9. Final CPU test run

**Setonix** (`work` partition CPU job 51177828, pytorch module, no GPU, each module in its own process with a 150 s timeout, code state
`a55239a`): **53 of 54 modules pass completely, all new/changed C1 modules pass** (lifecycle 17, populations 6, causality 28, certified
capacity 32, simulator wiring 8, pilot config 5, pilot analysis 7, draft timing 6, certify_x3 8, capacity 43, x3 harness etc.).
Exceptions, none caused by C1:
* `test_capacity_script_e2e.py` timed out at 150 s there: the driver's `_environment_info()` does `import vllm`, which hangs on a node
  without a GPU (probe job: `import vllm` > 150 s; torch 8 s, transformers 7 s). Fixed afterwards by stubbing `_environment_info` in the test
  fixture; verified locally (4 passed); not re-run on Setonix.
* `test_payload_interface.py::test_full_vocab_payload_exposes_dense_distribution_support` fails on a float-precision equality
  (`-0.2` vs `-0.2000000298…`); **identical failure at the parent commit `aba9cec`** (pre-existing).
* Three modules were previously reported failing locally on Windows (`test_runtime_guard` ×3, `test_batch_wall_fallback…`); on Setonix they pass
  (platform-specific paths/timer resolution).

**Local (Windows, no torch)**: 432 passed, 4 failed (the 4 Windows-only failures above, identical at `aba9cec`); the 3 torch-only modules are
covered by the Setonix run.

Code after `a55239a` differs only by the e2e-fixture stub and the before/after script (extra D2 scenario); no library code changed.
