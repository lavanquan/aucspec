"""C5-A: a small, INDEPENDENT closed-system discrete-event model used only to test mathematical claims (it shares no code with the project simulator).

Model (all times in ms):  N clients; client i alternates  [wait for service]  ->  batch finishes at f  ->  returns after D_i  ->  next request arrives at f + D_i
(+ optional non-negative jitter).  A single server processes batches ATOMICALLY; a batch S takes  T(S) = a + sum_{i in S} d_i  (deterministic, affine; noise optional).
Batch formation rules (the server is idle at t_free):
  hold_head      start s = max(t_free, a_head + W); batch = the first min(B, #arrived by s) requests in arrival order (ties: client id)     [the C3 work-conserving rule]
  full_or_timeout s = max(t_free, min(a_head + W, arrival time of the B-th unserved request)); same fill rule                                [launch immediately when B are available]
  wait_k         s = max(t_free, arrival time of the k-th unserved request), batch = first min(B, #arrived by s)                             [Kar et al.-style threshold]
Every client is, when the server is idle, either waiting or scheduled to arrive (service is atomic), so the decision is exact.
Outputs: completions per client in a window, batch-size sequence, busy fraction.  Rates are phi_i * completions / window  (phi_i = expected useful tokens per round)."""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field


@dataclass
class Result:
    t_end: float
    completions: dict
    batches: list = field(default_factory=list)   # (start, finish, [client ids])
    deadlocked: bool = False

    def rate(self, i, phi, t0, t1):
        n = sum(1 for (s, f, ids) in self.batches if t0 <= f <= t1 for j in ids if j == i)
        return phi * n / ((t1 - t0) / 1000.0)


def simulate(N, B, a, d, D, phi=None, W=0.0, rule="hold_head", k=None, start=None, t_max=60000.0, jitter=0.0, noise=0.0, seed=0):
    """d, D: scalar or list of per-client values. start: per-client first-arrival times (default all 0)."""
    rng = random.Random(seed)
    dl = d if isinstance(d, (list, tuple)) else [d] * N
    Dl = D if isinstance(D, (list, tuple)) else [D] * N
    start = start or [0.0] * N
    pend = [(float(start[i]), i) for i in range(N)]            # (arrival time, client) scheduled arrivals
    heapq.heapify(pend)
    waiting: list[tuple[float, int]] = []                       # arrived, unserved (sorted by (time, id))
    t_free = 0.0
    batches = []
    comp = {i: 0 for i in range(N)}
    while True:
        unserved = len(waiting) + len(pend)
        if unserved == 0:
            break
        # arrival times of the unserved requests in order
        allq = sorted(waiting + list(pend))
        a_head = allq[0][0]
        if rule == "hold_head":
            s = max(t_free, a_head + W)
        elif rule == "full_or_timeout":
            tB = allq[B - 1][0] if len(allq) >= B else float("inf")
            s = max(t_free, min(a_head + W, tB))
        elif rule == "wait_k":
            if len(allq) < k:
                return Result(t_free, comp, batches, deadlocked=True)
            s = max(t_free, allq[k - 1][0])
        else:
            raise ValueError(rule)
        if s > t_max:
            break
        while pend and pend[0][0] <= s:
            waiting.append(heapq.heappop(pend))
        waiting.sort()
        members = waiting[:B]
        waiting = waiting[B:]
        ids = [i for (_, i) in members]
        T = a + sum(dl[i] for i in ids)
        if noise:
            T *= 1.0 + noise * (2 * rng.random() - 1)
        f = s + T
        batches.append((s, f, ids))
        for i in ids:
            comp[i] += 1
            heapq.heappush(pend, (f + Dl[i] + (jitter * rng.random() if jitter else 0.0), i))
        t_free = f
    return Result(t_free, comp, batches)


def cycle_stats(res: Result, t0: float, t1: float, N: int):
    """Batch-size sequence and mean batch size of the batches that finish in [t0, t1]; busy fraction of the server in the window."""
    bs = [(s, f, ids) for (s, f, ids) in res.batches if t0 <= f <= t1]
    busy = sum(min(f, t1) - max(s, t0) for (s, f, _) in res.batches if f > t0 and s < t1) / (t1 - t0)
    return {"n_batches": len(bs), "sizes": [len(ids) for (_, _, ids) in bs], "mean_size": sum(len(ids) for (_, _, ids) in bs) / max(1, len(bs)), "busy": busy}


# ----------------------------------------------------------------------------------------------------------------------------------------
# closed forms (the objects whose validity conditions are being tested)
# ----------------------------------------------------------------------------------------------------------------------------------------
def greedy_partition(N, B):
    q, r = divmod(N, B)
    return [B] * q + ([r] if r else [])


def saturated_cycle(N, B, a, d_list):
    """K*a + sum d  with K = ceil(N/B)."""
    K = -(-N // B)
    return K * a + sum(d_list)


def maxplus_cycle(N, B, a, d_list, D, W):
    """R = max{ K a + sum d ,  D + W + T_max },  T_max = a + (sum of the B largest... of the actual largest batch) for the greedy partition in client-id order.
    Valid ONLY under the assumptions of Theorem A (see the report): identical D, hold_head rule, fixed greedy partition in id order, deterministic affine batches."""
    part = greedy_partition(N, B)
    T = []
    p = 0
    for sz in part:
        T.append(a + sum(d_list[p:p + sz]))
        p += sz
    return max(sum(T), D + W + max(T))
