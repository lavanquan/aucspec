# Virtual-time event dependency graph (as implemented before C1) and the causal repair

Scope: `EdgeSpecSimulator._run_round`, `VerificationBatcher._loop/_process`, `OnlineController.observe_server_batch`.
Time `t` is *virtual* milliseconds. Real execution (asyncio + GPU threads) decides **when code runs**, virtual
timestamps decide **what should have been visible**. A correct hybrid simulator must make the second independent
of the first.

## 1. Events and their producers (before C1)

| Event | Virtual timestamp | Producer | Consumer |
|---|---|---|---|
| `decision(i, k)`: depth γ_i(k) | `t = client.virtual_time_ms` (= receive time of round k-1) | client coroutine i | controller (`choose_gamma`), reads Z_i, Q_i (own) and **Q_s, price, active-client count (global)** |
| `draft(i, k)` | `[t, t + d_i]` (measured GPU latency, virtual) | draft worker (real GPU, real queueing) | – |
| `arrival(i, k)` | `a = t + d_i + RTT_i/2 + upload_i` | client coroutine, **submitted in real completion order** | `VerificationBatcher.queue` |
| `batch_formed(b)` | cutoff `C = a_head + W` (W = `batch_wait_ms`) | batcher loop | `pop_batch(eligible = {a <= C})` |
| `batch_service(b)` | `[s, f]`, `s = max(C, f_{b-1})`, `f = s + wall_b` (physical wall time) | batcher (`_process`, real GPU) | – |
| `result(i, k)` | `f` | batcher, via `future.set_result` | client i |
| `server_state_update(b)` | `f` | **the first client coroutine that resumes** (`observe_server_batch`, dedup by batch id) | controller `server_queue`, `server_price`, `server_state_max_finish_ms` |
| `receive(i, k)` | `r = f + RTT_i/2 + download_i` | client coroutine | Z_i, Q_i updates, next `decision(i,k+1)` at `t = r` |

Dependency edges: `receive(i,k) → decision(i,k+1) → draft → arrival(i,k+1) → batch_formed → batch_service → result/receive`,
plus the global edges `server_state_update(b) → decision(j, ·)` for every client `j`, and the *shared* edge
`arrival(·) → batch_formed` for every client.

## 2. Where real order leaks into virtual semantics

**D1 (batch formation).** `batch_formed` with cutoff `C` must contain every request with `a <= C`. The batcher
forms it from whatever has *really* been submitted. A client that is still drafting in real time (GPU queueing,
scheduler jitter) but whose virtual arrival is earlier than `C` is silently excluded. A "second chance"
(`asyncio.sleep(0)` + drain) hid this in the common case. Measured on the tfa33 smoke (N=8): 31.3 % of batches omit an
already-arrived request (mean 162–185 ms early); defective-runtime run N=10: 25.3 %.

**D2 (shared server state).** `observe_server_batch` runs when a *client* resumes, in real order, and overwrites a
single mutable `server_queue`. A decision at virtual time `t` can therefore observe (i) a later batch finishing at
`f > t` (6.7 % of depth decisions in the smoke, mean 212 ms in the future), and (ii) the stamped price on a request
includes it for 47 % of requests. The server price is 0 in the qualified fixed-budget configuration, so D2 is
presently *inert* but would corrupt any priced policy, and any baseline reading `server_queue`.

**D3 (other global reads, documented, not changed).** `_allocate_shared_bandwidth` and
`set_active_client_signal(len(_active_clients()))` read other clients' *current* `gamma`, `z_queue`, `device_queue`,
`remaining_tokens` in real order. In the qualified regime (greedy proposals, payload ~100 B, upload 0.027 ms,
shared allocator `proportional`) the effect on `upload_ms` is bounded by the payload term and is ≈ 0.03 ms of a
round of ≈ 1.8 s; `active_clients` is used only by budget-per-client policies that are off in the pilot. D3 is a
**limitation** until a priced bandwidth ablation (RQ4) is run.

## 3. Repair: conservative virtual-event coordination (no sleeps, no real-time delays)

A `VirtualCoordinator` keeps, for every client, a **lower bound** on the virtual timestamp of its next request
arrival, and the batcher/decisions only proceed when the lower bounds make the outcome independent of real order
(conservative parallel discrete-event simulation with lookahead).

Lookahead (all hold by construction because draft, upload, download ≥ 0 and RTT_i > 0):

* client starts: `lb_i = RTT_i/2`;
* client resumed at decision time `t`: `lb_i = t + RTT_i/2`;
* result delivered for a batch finishing at `f`: `lb_i = f + RTT_i` (down + up half-RTTs);
* client has a request queued or in service: not counted (its next arrival is after its current result);
* client retired (no work): `lb_i = +inf`.

**Batch formation (D1).** Head `h` = earliest queued request, cutoff `C = a_h + W`. A batch is formed only when every
client that is *not* in the queue has `lb_i > C` (strict: a request arriving exactly at `C` is eligible). Newly
submitted items are drained and the head recomputed until stable. The selection function, budget, scheduling
formulas and `batch_wall` accounting are untouched.

**Server-state visibility (D2).** The batcher publishes each completed batch to the controller *itself, in batch
order*, before waking any client, and the controller stores a monotone **snapshot history**
`(finish_ms, Q_s, price, version)`. A decision (or request stamp) at virtual time `t` reads the snapshot with
`finish_ms <= t`, and may only proceed once the **verifier horizon** `H > t`:

`H = min( start of a batch currently in service,  max(server_available_at, min(head arrival, min_i lb_i) + W) )`

(every batch not yet processed finishes at or after `H`). A snapshot with `finish_ms > t` is never returned.

**Deadlock freedom.** The client with the smallest decision time `t*` is never blocked: its own bound gives
`H >= t* + RTT/2 + W > t*`, and any batch waiting for a watermark is waiting for clients that are *drafting* (not
waiting on the coordinator). Degenerate `RTT = 0 and W = 0` has no lookahead and is rejected at construction; a
runtime detector raises `CausalityDeadlock` if every active client is blocked on the coordinator.

**Ledger.** Every arrival, batch formation, batch completion and decision is recorded with: virtual timestamp,
type, producer, decision timestamp, latest server-state update visible to the decision, and virtual queue-state
version. `EventLedger.check_causality()` asserts: visible update ≤ decision timestamp; every request with arrival ≤
cutoff is in the batch or an earlier batch; verifier service intervals do not overlap; no request dropped/duplicated.

## 4. What is *not* changed
Acceptance sampling, V1 greedy verification and tie-break, physical `batch_wall` accounting, `capacity_dpp_score`,
`capacity_dpp_value` and the knapsack, baseline semantics. Causal mode is a config flag
(`simulation.causal_virtual_time: true`); the historical mode remains available and its results stay labelled
"real-order batching".
