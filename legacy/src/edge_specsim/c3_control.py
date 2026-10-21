"""C3 minimal capacity-aware control (opt-in, experimental; does NOT modify `capacity_dpp` or any canonical policy).

Two separable mechanisms, specified in results/AUCSPEC_C3_CONTROLLER_DESIGN.md:

  A. `c3_deficit` batch scheduler (+ work-conserving batch formation in VerificationBatcher): deficit-aware selection of at most
     B_max requests whose processed tokens fit the physical token budget and whose PREDICTED batch time fits an optional cap.
  B. `c3_depth` speculative-depth policy (C3Controller): per-client argmax of a score whose terms are all in (weighted) tokens.

Every parameter that is not measured is marked ASSUMED; no constant fitted on the C1 traces is presented as independent validation.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from .controller import OnlineController
from .expectations import expected_useful_tokens_from_profile


# ----------------------------------------------------------------------------------------------------------------------
# physical cost model (used for PREDICTION by the controller; the simulator/GPU supplies the true service time)
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CostModel:
    """T_batch(S) = a + sum_{r in S} d_r,   d_r = c_req + b*ctx_r + b_gamma*gamma_r        [ms]          (design eq. (S1))

    a            fixed per-batch overhead (ms)                      preliminary, loaded-run fit (results/c2): 52 ms
    b            ms per processed context token                     preliminary, loaded-run fit: 0.19 ms/token
    b_gamma      ms per processed CANDIDATE token                   ASSUMED = b: the C1 traces hold gamma constant, so it is not identified
    c_req        ms fixed cost per admitted request                 ASSUMED 0 (confounded with b_gamma*gamma when gamma is constant)
    """
    a_ms: float = 52.0
    b_ms_per_token: float = 0.19
    b_gamma_ms_per_token: float | None = None
    c_req_ms: float = 0.0

    @property
    def b_gamma(self) -> float:
        return self.b_ms_per_token if self.b_gamma_ms_per_token is None else self.b_gamma_ms_per_token

    def demand_ms(self, ctx_tokens: int, gamma: int) -> float:
        return self.c_req_ms + self.b_ms_per_token * max(0, ctx_tokens) + self.b_gamma * max(0, gamma)

    def batch_ms(self, members: list[tuple[int, int]]) -> float:
        """members: (ctx_tokens, gamma) pairs; empty batch costs nothing."""
        return (self.a_ms + sum(self.demand_ms(c, g) for c, g in members)) if members else 0.0


@dataclass(frozen=True)
class C3Params:
    # scheduling constraints (physical; never relaxed by the controller)
    max_batch_size: int = 16
    processed_token_budget: int = 4096         # frozen runtime H: max_num_batched_tokens (chunked prefill off)
    service_time_cap_ms: float = math.inf      # optional cap on the PREDICTED batch time
    max_skips: int = 2                          # starvation guard: eligible-but-unselected K times => served next (deterministic bound)
    # weights
    V: float = 50.0                             # tokens (same V as the canonical controller)
    x_req_tps: float = 3.0                      # SLO floor (tokens/s) driving the deficit
    cost: CostModel = field(default_factory=CostModel)
    # depth policy
    window_ms: float = 5000.0                   # trailing window for the goodput price
    t_ref_ms: float = 52.0                      # congestion normaliser: waiting work equal to one fixed batch overhead => gate 1/2
    use_price: bool = True                      # False = device-latency-only depth ablation (no verifier term)
    use_deficit_weight: bool = True
    read_lag_ms: float = 0.0                    # fault injection for the delayed-state test: decisions read the snapshot of time t - lag (never > 0 in a real run)
    # mechanism A ranking, declared in advance (both are evaluated, neither is tuned):
    #   "density": rank by (V + Z_eff) * phi_hat / (a/B_max + d_r)           max-weight / throughput-weighted form
    #   "deficit": rank by Z_eff first (neediest client first), ties by density   floor-oriented form
    #   "deficit_fcfs": rank by Z_eff first, ties by ARRIVAL order (diagnostic ablation: removes the density tie-break)
    #   "age": oldest DECISION first (age-only falsification rank, C4); "z_only": genuine stamped client Z_i only (C4 falsification rank)
    priority_mode: str = "density"


# ----------------------------------------------------------------------------------------------------------------------
# mechanism A: deficit-aware selection
# ----------------------------------------------------------------------------------------------------------------------
def effective_deficit(req, start_ms: float, x_req_tps: float, v: float) -> float:
    """Z_i^eff(S) = Z_i(stamp) + x_req * (S - t_stamp)/1000   [tokens]                                         (design eq. (D2))
    The client delivers no token between its last completion (stamp time) and this request's completion, so its deficit grows
    linearly while it waits; both operands are known at S (decision time <= arrival <= S)."""
    z_stamp = max(0.0, float(req.weight) - float(v))
    t0 = req.decision_time_ms if req.decision_time_ms is not None else req.arrival_time
    return z_stamp + x_req_tps * max(0.0, start_ms - t0) / 1000.0


def request_priority(req, start_ms: float, p: C3Params) -> float:
    """density = (V + Z_eff) * phi_hat / (a/B_max + d_r)   [tokens per ms]                                      (design eq. (D3))"""
    z = effective_deficit(req, start_ms, p.x_req_tps, p.V) if p.use_deficit_weight else 0.0
    ctx = len(req.context_token_ids)
    d = p.cost.demand_ms(ctx, req.gamma) + p.cost.a_ms / max(1, p.max_batch_size)
    return (p.V + z) * max(1e-9, req.expected_useful_tokens) / max(1e-9, d)


class ConstraintInfeasible(ValueError):
    """A request that cannot be served within a HARD limit even alone (processed-token budget M, or predicted service-time cap).  Raised instead of
    serving it anyway: the run is explicitly invalid (fail-closed).  This is a property of the request, not of the schedule."""


def preflight_request(req, p: C3Params) -> None:
    """Admission preflight (also called by VerificationBatcher.submit): a singleton must satisfy the hard limits on its own.
      * processed tokens  p_r = ctx_r + gamma_r <= M                  hard (engine max_num_batched_tokens, chunked prefill off)
      * predicted time    T_hat({r}) <= T_max                           model-based: a prediction, NOT a memory-safety guarantee
    Actual GPU memory safety (sampler logprob workspace ~ processed tokens x vocab) cannot be established by this check or by any CPU test."""
    ptoks = len(req.context_token_ids) + len(req.candidate_token_ids)
    if ptoks > p.processed_token_budget:
        raise ConstraintInfeasible(f"request (client {req.client_id}) has {ptoks} processed tokens > budget {p.processed_token_budget}")
    t_hat = p.cost.batch_ms([(len(req.context_token_ids), req.gamma)])
    if t_hat > p.service_time_cap_ms:
        raise ConstraintInfeasible(f"request (client {req.client_id}) predicted singleton service {t_hat:.1f} ms > cap {p.service_time_cap_ms:.1f} ms")


def _greedy_fill(ranked: list, eligible: list, p: C3Params) -> list:
    """Admit in rank order while (count, processed tokens, predicted batch time) constraints hold.  Fail-closed: every eligible request must be admissible
    alone (preflight), so the first ranked request can always be admitted without bypassing a hard limit (liveness for admissible workloads) and a
    non-admissible request is never silently served.  Requests eligible but not admitted get one more skip."""
    for r in eligible:
        preflight_request(r, p)
    selected, tokens, members = [], 0, []
    for r in ranked:
        ptoks = len(r.context_token_ids) + len(r.candidate_token_ids)
        m = members + [(len(r.context_token_ids), r.gamma)]
        fits = (len(selected) < p.max_batch_size and tokens + ptoks <= p.processed_token_budget
                and p.cost.batch_ms(m) <= p.service_time_cap_ms)
        if fits:
            selected.append(r)
            tokens += ptoks
            members = m
    chosen = {id(r) for r in selected}
    for r in eligible:
        if id(r) not in chosen:
            r.c3_skips = getattr(r, "c3_skips", 0) + 1
    return selected


def select_c3_deficit(eligible: list, start_ms: float, p: C3Params) -> list:
    """Mechanism A: greedy max-density selection (design eq. (D3)).  Never returns an empty list for a non-empty eligible set (work
    conserving, no forced-service escape hatch).  A request skipped `max_skips` times is ranked first (deterministic wait bound)."""
    def key(r):
        forced = getattr(r, "c3_skips", 0) >= p.max_skips
        if forced:
            return (0, 0.0, 0.0, r.arrival_time, r.client_id, r.request_id)
        dens = request_priority(r, start_ms, p)
        z = effective_deficit(r, start_ms, p.x_req_tps, p.V) if p.priority_mode in ("deficit", "deficit_fcfs") else 0.0
        if p.priority_mode == "deficit_fcfs":
            return (1, -z, 0.0, r.arrival_time, r.client_id, r.request_id)
        if p.priority_mode == "age":
            t0 = r.decision_time_ms if r.decision_time_ms is not None else r.arrival_time
            return (1, t0, 0.0, r.arrival_time, r.client_id, r.request_id)
        if p.priority_mode == "z_only":
            return (1, -max(0.0, float(r.weight) - float(p.V)), 0.0, r.arrival_time, r.client_id, r.request_id)
        return (1, -z, -dens, r.arrival_time, r.client_id, r.request_id)

    return _greedy_fill(sorted(eligible, key=key), eligible, p)


def select_c3_fcfs(eligible: list, start_ms: float, p: C3Params) -> list:
    """Fair baseline: arrival order under EXACTLY the same physical constraints as select_c3_deficit (only the ranking differs)."""
    del start_ms
    return _greedy_fill(sorted(eligible, key=lambda r: (r.arrival_time, r.client_id, r.request_id)), eligible, p)


# ----------------------------------------------------------------------------------------------------------------------
# state: physical/virtual workload ledger (Stage A) -- pure function of events <= t
# ----------------------------------------------------------------------------------------------------------------------
class WorkLedger:
    """Unfinished predicted verifier work, reconstructable at any time t from the events with timestamp <= t.

      U(t) = sum_{r waiting at t} d_r  +  (remaining service of the batch in service at t)             [ms]   (design eq. (S2))
      A(t) = sum of predicted demand of requests that arrived by t;   D(t) = sum of predicted demand of requests whose batch finished by t.
      Each request enters A exactly once (at its arrival) and D exactly once (at its batch's completion).  The per-batch overhead a is
      charged once, to the batch (service interval), never to the requests -> no double counting on submission and completion.
    """

    def __init__(self) -> None:
        self.arrivals: list[tuple[float, int, float]] = []        # (t, request key, d_r)
        self.batches: list[tuple[float, float, float, list[int]]] = []  # (start, finish, overhead_charged, member keys)

    def arrive(self, t: float, key: int, demand_ms: float) -> None:
        self.arrivals.append((float(t), int(key), float(demand_ms)))

    def serve(self, start: float, finish: float, overhead_ms: float, keys: list[int]) -> None:
        if self.batches and start + 1e-9 < self.batches[-1][1]:
            raise ValueError("overlapping verifier service")
        self.batches.append((float(start), float(finish), float(overhead_ms), list(keys)))

    def A(self, t: float) -> float:
        return sum(d for ts, _, d in self.arrivals if ts <= t + 1e-9)

    def D(self, t: float) -> float:
        done = 0.0
        dem = {k: d for _, k, d in self.arrivals}
        for _, fin, _, keys in self.batches:
            if fin <= t + 1e-9:
                done += sum(dem[k] for k in keys)
        return done

    def unfinished_work(self, t: float) -> float:
        """Waiting demand + in-service remaining time (overhead included for the batch in service)."""
        dem = {k: (ts, d) for ts, k, d in self.arrivals}
        served_started = {k for s, _, _, ks in self.batches if s <= t + 1e-9 for k in ks}
        waiting = sum(d for ts, k, d in ((a[0], a[1], a[2]) for a in self.arrivals) if ts <= t + 1e-9 and k not in served_started)
        remaining = 0.0
        for s, fin, _, _ in self.batches:
            if s <= t + 1e-9 < fin:
                remaining = fin - t
        return waiting + remaining


# ----------------------------------------------------------------------------------------------------------------------
# mechanism B: context-aware depth control
# ----------------------------------------------------------------------------------------------------------------------
def depth_score(omega: float, phi: float, gamma: int, mu_tok_per_ms: float, b_gamma_ms: float, xhat_tok_per_ms: float, c_dev_ms: float) -> float:
    """Design eq. (B1), a pure function so that dimensional consistency can be tested:

        S = omega*phi - (mu*b_gamma + omega*xhat*c_dev) * gamma           [tokens]

    mu [tok/ms] * b_gamma [ms/tok_drafted] -> tokens per drafted token;  xhat [tok/ms] * c_dev [ms/tok_drafted] -> tokens per drafted token.
    Rescaling time (ms -> s) multiplies every rate by k and divides every duration by k, leaving S unchanged."""
    return omega * phi - (mu_tok_per_ms * b_gamma_ms + omega * xhat_tok_per_ms * c_dev_ms) * gamma


class C3Controller(OnlineController):
    """policy == "c3_depth":  gamma_i = argmax_g S_i(g),

        S_i(g) = omega_i * phi_hat_i(g)  -  [ mu(t) * b_gamma  +  omega_i * xhat_i * c_dev,i ] * g                       [tokens]     (design eq. (B1))

      omega_i  = (V + Z_i)/V                         dimensionless weight (tokens/tokens)
      phi_hat  = expected useful tokens per round (positional acceptance profile, no UCB bonus), phi_hat(0) = 1
      xhat_i   = own delivered rate over the last rounds [tokens/ms] (own completions: causal)
      c_dev,i  = tau_d + kappa/r                     ms per drafted token on the device (existing definition)
      mu(t)    = G(t) * U_hat(t)/(U_hat(t) + T_ref)  tokens/ms: verifier shadow price = congestion gate x trailing goodput of the snapshot
      b_gamma  = ASSUMED verifier ms per candidate token (CostModel.b_gamma)
    All other policies are inherited unchanged."""

    def __init__(self, *args, c3: C3Params | None = None, **kwargs) -> None:
        requested = str(kwargs.get("policy", "fixed_gamma")).lower()
        if requested == "c3_depth":                      # the base class validates against its own (unchanged) policy list
            kwargs = dict(kwargs, policy="fixed_gamma")
        super().__init__(*args, **kwargs)
        self.policy = requested
        self.c3 = c3 or C3Params()
        # snapshot history, appended in verifier-completion order (D2 publication order); batches are sequential so start and finish are both monotone
        self._c3_hist: list[tuple[float, float, int, float, float, float]] = []   # (finish, wall_ms, weighted_useful, leftover_work_ms, start, predicted_ms)
        self._c3_finishes: list[float] = []
        self._c3_starts: list[float] = []
        self._own: dict[int, list[tuple[float, int]]] = {}              # client -> [(receive_ms, useful)] own completions
        self.c3_log: list[dict] = []
        self.shadow = False                      # True: another policy ACTS, the depth rule only logs what it would have chosen (open-loop replay)
        self.shadow_log: list[dict] = []

    # --- publication (called by the simulator in batch order, before any client wakes)
    def observe_batch_meta(self, meta) -> None:
        finish = float(meta.verify_finish_ms)
        if self._c3_finishes and finish + 1e-9 < self._c3_finishes[-1]:
            raise ValueError("C3 snapshots must be published in finish order")
        start = float(meta.batch_start_ms)
        wu = float(getattr(meta, "batch_weighted_useful", 0.0)) or float(meta.batch_useful_tokens)     # no C3 batcher params: unweighted tokens
        self._c3_hist.append((finish, float(meta.batch_wall_ms), wu, float(meta.waiting_work_after_ms), start,
                              float(meta.predicted_service_ms)))
        self._c3_finishes.append(finish)
        self._c3_starts.append(start)

    def observe_completion(self, client, client_receive_ms, useful_tokens):
        self._own.setdefault(client.client_id, []).append((float(client_receive_ms), int(useful_tokens)))
        return super().observe_completion(client, client_receive_ms, useful_tokens)

    # --- reads (pure functions of the history <= t)
    def snapshot_at(self, t_ms: float) -> dict:
        """State as of virtual time t, a pure function of the batches published for the event trace:
          * goodput G(t): deficit-weighted realized useful tokens, sum_j (w_j/V) u_j, of the batches that FINISHED by t over the trailing window
            (D2 snapshot) [weighted tokens/ms] - the same unit as the omega_i*phi_i terms it is traded against;
          * demand U_hat(t): predicted remaining service of the batch that STARTED by t (start + predicted - t, never the realized finish) plus the
            predicted work of the requests that had arrived by that batch's start and were not in it (they are served after it).
        The coordinator's horizon rule (decision at t waits until horizon > t) guarantees every batch with start <= t is already published."""
        t = float(t_ms)
        k = bisect.bisect_right(self._c3_finishes, t + 1e-9)
        ks = bisect.bisect_right(self._c3_starts, t + 1e-9)
        goodput, finish = 0.0, 0.0
        if k:
            finish = self._c3_hist[k - 1][0]
            lo = bisect.bisect_right(self._c3_finishes, finish - self.c3.window_ms)
            goodput = sum(h[2] for h in self._c3_hist[lo:k]) / max(1.0, min(self.c3.window_ms, finish))
        u_hat, start_used = 0.0, None
        if ks:
            _, _, _, leftover, start_used, predicted = self._c3_hist[ks - 1]
            u_hat = max(0.0, start_used + predicted - t) + leftover
        return {"version": k, "started": ks, "finish": finish, "start_used": start_used, "goodput_tok_per_ms": goodput, "u_hat_ms": u_hat}

    def own_rate_tok_per_ms(self, client_id: int, t_ms: float) -> float:
        hist = [h for h in self._own.get(client_id, []) if h[0] <= t_ms + 1e-9][-8:]
        if len(hist) < 2:
            return self.min_tps / 1000.0                                         # prior: the floor
        span = hist[-1][0] - hist[0][0]
        return (sum(u for _, u in hist[1:]) / span) if span > 0 else self.min_tps / 1000.0

    def c3_score(self, client, gamma: int, profile: list[float], snap: dict, xhat: float) -> float:
        p = self.c3
        omega = (p.V + (client.z_queue if (self.use_virtual_queues and p.use_deficit_weight) else 0.0)) / p.V
        phi = expected_useful_tokens_from_profile(profile) if gamma > 0 else 1.0
        gate = snap["u_hat_ms"] / (snap["u_hat_ms"] + p.t_ref_ms) if snap["u_hat_ms"] > 0 else 0.0
        mu = (gate * snap["goodput_tok_per_ms"]) if p.use_price else 0.0
        c_dev = self._capacity_dpp_device_cost_per_token_ms(client)
        return depth_score(omega, phi, gamma, mu, p.cost.b_gamma, xhat, c_dev)

    def _c3_argmax(self, client, snap: dict, xhat: float) -> int:
        best, best_s = 0, self.c3_score(client, 0, [], snap, xhat)
        for g in self.gamma_choices:
            if g == 0:
                continue
            s = self.c3_score(client, g, client.positional_acceptance_profile(g, False), snap, xhat)
            if s > best_s + 1e-12:                                                  # ties -> smaller gamma
                best, best_s = g, s
        return best

    def choose_gamma(self, client, total_rounds: int) -> int:
        if self.policy != "c3_depth":
            gamma = super().choose_gamma(client, total_rounds)
            if self.shadow:
                t = client.virtual_time_ms
                snap = self.snapshot_at(t)
                xhat = self.own_rate_tok_per_ms(client.client_id, t)
                self.shadow_log.append({"client": client.client_id, "t": t, "acted": gamma, "would": self._c3_argmax(client, snap, xhat),
                                        "u_hat": snap["u_hat_ms"], "goodput": snap["goodput_tok_per_ms"], "xhat": xhat})
            return gamma
        self._tick += 1
        client.refresh_learning_stats(total_rounds)
        t = client.virtual_time_ms
        snap = self.snapshot_at(t - self.c3.read_lag_ms)
        xhat = self.own_rate_tok_per_ms(client.client_id, t)
        best = self._c3_argmax(client, snap, xhat)
        self.c3_log.append({"client": client.client_id, "t": t, "gamma": best, "version": snap["version"], "started": snap["started"], "start_used": snap["start_used"], "u_hat": snap["u_hat_ms"],
                            "goodput": snap["goodput_tok_per_ms"], "xhat": xhat, "z": client.z_queue})
        client.last_selected_gamma = best
        return best


# ----------------------------------------------------------------------------------------------------------------------
# online estimate of the physical service time (specification component; NOT wired into the controller by default)
# ----------------------------------------------------------------------------------------------------------------------
class OnlineCostEstimator:
    """Ridge-regularised recursive least squares for  wall = a + b*ctx_sum + c_req*B + b_gamma*gamma_sum   [ms]  (forgetting factor lambda).

    The prior (mean, precision) is the ASSUMED cost model.  A direction in coefficient space that the data never excites keeps its prior
    variance, so the estimator REPORTS the identifiability gap instead of hiding it: while gamma is constant, gamma_sum = gamma*B is collinear
    with B, only the combination c_req + gamma*b_gamma is learned and the two coefficients individually stay uncertain.
    Prediction uses only features known when the batch is formed (no look-ahead); update() receives the measured wall time afterwards."""

    def __init__(self, prior: CostModel | None = None, prior_precision=(1e-4, 1e2, 1e-1, 1e-1), forgetting: float = 1.0) -> None:
        import numpy as np

        self._np = np
        prior = prior or CostModel()
        self.theta = np.array([prior.a_ms, prior.b_ms_per_token, prior.c_req_ms, prior.b_gamma], float)
        self.P = np.linalg.inv(np.diag(np.asarray(prior_precision, float)))
        self.lam = float(forgetting)
        self.n = 0

    @staticmethod
    def features(members: list[tuple[int, int]]):
        return [1.0, float(sum(c for c, _ in members)), float(len(members)), float(sum(g for _, g in members))]

    def predict_ms(self, members: list[tuple[int, int]]) -> float:
        return float(self._np.dot(self.theta, self.features(members))) if members else 0.0

    def update(self, members: list[tuple[int, int]], wall_ms: float, noise_var: float = 25.0 ** 2) -> None:
        np = self._np
        x = np.asarray(self.features(members), float)
        Px = self.P @ x
        gain = Px / (self.lam * noise_var + float(x @ Px))
        self.theta = self.theta + gain * (float(wall_ms) - float(self.theta @ x))
        self.P = (self.P - np.outer(gain, Px)) / self.lam
        self.n += 1

    def coef_std(self):
        return self._np.sqrt(self._np.clip(self._np.diag(self.P), 0.0, None))
