"""C3 scenario factory (CPU closed loop on the real simulator/queue/batcher code; CPU physical-verifier stand-in).

Everything is fixed in advance and independent of controller outcomes.  Populations are STATIC and matched across arms: the same (N, population,
seed) gives the same alphas, hardware, prompt lengths, start offsets and arrival jitter to every arm; the acceptance draws are coupled
(a token position has the same accept/reject outcome under every policy that proposes it)."""
from __future__ import annotations

import contextlib
import io
from dataclasses import replace

import closed_loop_harness as ch
from edge_specsim.c3_control import C3Params, CostModel

POPULATIONS = ("homog_sync", "hetero_sync", "hetero_async")


def _znorm(pop_seed: int, tag: str, i: int) -> float:
    """Hash-based approximately standard normal (Irwin-Hall n=12), independent across (pop_seed, tag, i)."""
    return sum(ch._u("pz", pop_seed, tag, i, j) for j in range(12)) - 6.0


# Measured in the valid C1 GPU traces (results/c3_1/trace_conditions.json): per-client alpha_hat 0.930 (sd 0.017, range 0.88-0.96, gamma=2, GSM8K-train);
# per-prompt alpha dispersion ~0.02 (mean of the 7 per-run between-prompt estimates); identical rtt/draft latency for all clients; per-request context
# mean 158 (batch means p5-p95 121-196, ctx_max up to 263 tokens).  Start offsets and jitter are NOT measured properties: they are the pilot's
# controllable de-synchronisation knobs.
MEASURED_ALPHA_MEAN, MEASURED_ALPHA_SD, MEASURED_PROMPT_ALPHA_SD = 0.930, 0.017, 0.020


def population(kind: str, n: int, *, ctx_max: int = 300, pop_seed: int = 0, alpha_sd: float | None = None, alpha_mean: float = MEASURED_ALPHA_MEAN) -> dict:
    """Return Spec kwargs for a static population.  pop_seed = 0 reproduces the C3 populations exactly; pop_seed > 0 draws an independent population
    (kinds 'measured', 'measured_jitter', 'wide_async' only)."""
    if kind in ("measured", "measured_jitter"):
        sd = MEASURED_ALPHA_SD if alpha_sd is None else alpha_sd
        lo, hi = (0.88, 0.96) if alpha_sd is None else (0.30, 0.995)           # measured clip range; wider spreads are clipped only to a valid probability
        alphas = [min(hi, max(lo, alpha_mean + sd * _znorm(pop_seed, "alpha", i))) for i in range(n)]
        plens = [int(85 + 85 * ch._u("pz", pop_seed, "pl", i)) for i in range(n)]             # prompt 85-170 tokens => ctx up to ~270 with 96 generated (moment-matched)
        out = dict(alphas=alphas, hetero_hw=False, prompt_lens=plens, alpha_prompt_sd=MEASURED_PROMPT_ALPHA_SD,
                   start_offsets_ms=[800.0 * ch._u("pz", pop_seed, "off", i) for i in range(n)])
        if kind == "measured_jitter":
            out["jitter_ms"] = 120.0
        return out
    if kind == "wide_async" and pop_seed:
        return dict(alphas=[0.60 + 0.35 * ch._u("pz", pop_seed, "alpha", i) for i in range(n)], hetero_hw=True,
                    prompt_lens=[int(80 + (ctx_max - 80) * ch._u("pz", pop_seed, "pl", i)) for i in range(n)],
                    start_offsets_ms=[800.0 * ch._u("pz", pop_seed, "off", i) for i in range(n)], jitter_ms=120.0)
    if kind == "wide_async":
        kind = "hetero_async"
    if kind == "homog_sync":
        return {}
    if kind == "weak_client":                       # one low-acceptance, long-context client among strong short-context ones
        al, pl = [0.92] * n, [112] * n
        al[0], pl[0] = 0.35, 700
        return dict(alphas=al, hetero_hw=False, prompt_lens=pl, start_offsets_ms=[800.0 * ch._u("off", i) for i in range(n)], jitter_ms=120.0)
    alphas = [0.60 + 0.35 * ch._u("alpha", i) for i in range(n)]
    plens = [int(80 + (ctx_max - 80) * ch._u("pl", i)) for i in range(n)]
    out = dict(alphas=alphas, hetero_hw=True, prompt_lens=plens)
    if kind == "hetero_async":
        out["start_offsets_ms"] = [800.0 * ch._u("off", i) for i in range(n)]
        out["jitter_ms"] = 120.0
    elif kind != "hetero_sync":
        raise ValueError(kind)
    return out


ARMS = {
    # name: (policy, fixed_gamma, scheduler, work_conserving)
    # legacy references (historical head+W windowing, logical token budget only)
    "fix2+legacy": ("fixed_gamma", 2, "fcfs", False),
    "fix4+legacy": ("fixed_gamma", 4, "fcfs", False),
    # fair baselines: work-conserving FCFS under the SAME physical constraints as the C3 scheduler
    "fix2+PF": ("fixed_gamma", 2, "c3_fcfs", True),
    "fix4+PF": ("fixed_gamma", 4, "c3_fcfs", True),
    # mechanism A only
    "fix2+DS": ("fixed_gamma", 2, "c3_deficit", True),
    "fix4+DS": ("fixed_gamma", 4, "c3_deficit", True),
    "fix2+DSz": ("fixed_gamma", 2, "c3_deficit", True),      # deficit-first ranking
    "fix4+DSz": ("fixed_gamma", 4, "c3_deficit", True),
    "fix2+DSa": ("fixed_gamma", 2, "c3_deficit", True),
    "fix4+DSa": ("fixed_gamma", 4, "c3_deficit", True),      # diagnostic ablation: deficit-first with ARRIVAL tie-break (no density term)
    # mechanism B only (and its ablation without the verifier price)
    "depth+PF": ("c3_depth", 0, "c3_fcfs", True),
    "depthNP+PF": ("c3_depth", 0, "c3_fcfs", True),
    # combined
    "depth+DS": ("c3_depth", 0, "c3_deficit", True),
    "depth+DSz": ("c3_depth", 0, "c3_deficit", True),
}


def make_spec(arm: str, n: int, pop: str, seed: int = 1, *, cap: int = 16, c3: C3Params | None = None, window_s: float = 120.0,
              warmup_s: float = 30.0, margin_s: float = 10.0, x_req: float = 3.0, V: float = 50.0, ctx_max: int = 300, **overrides) -> ch.Spec:
    policy, g, sched, wc = ARMS[arm]
    base = c3 or C3Params(max_batch_size=cap, V=V, x_req_tps=x_req)
    if arm.endswith("DSz"):
        base = replace(base, priority_mode="deficit")
    if arm.endswith("DSa"):
        base = replace(base, priority_mode="deficit_fcfs")
    if arm.startswith("depthNP"):
        base = replace(base, use_price=False)
    kw = dict(n=n, policy=policy, fixed_gamma=g if policy == "fixed_gamma" else 2, scheduler=sched, work_conserving=wc, max_batch=cap, c3=base,
              seed=seed, window_s=window_s, warmup_s=warmup_s, margin_s=margin_s, x_req=x_req, V=V, token_budget=100)
    kw["oracle_alpha"] = True
    pop_seed = overrides.pop("pop_seed", 0)
    alpha_sd = overrides.pop("alpha_sd", None)
    kw.update(population(pop, n, ctx_max=ctx_max, pop_seed=pop_seed, alpha_sd=alpha_sd))
    spread = overrides.pop("start_spread_ms", None)
    kw.update(overrides)
    if spread:
        kw["start_offsets_ms"] = [float(spread) * ch._u("off", i) for i in range(n)]
    return ch.Spec(**kw)


def run(spec: ch.Spec):
    with contextlib.redirect_stdout(io.StringIO()):
        return ch.run_sim(spec)


def metrics(sim) -> dict:
    """Aggregate measures for one run: rates, deficits, batches, waits, starvation, constraint enforcement, ledger."""
    import numpy as np

    s = ch.summarize(sim)
    recs = sim.records
    waits_by_client: dict[int, list[float]] = {}
    for r in recs:
        waits_by_client.setdefault(r["client_id"], []).append(r["batch_start_ms"] - r["server_arrival_ms"])
    mean_w = {c: float(np.mean(v)) for c, v in waits_by_client.items()}
    max_w = max(max(v) for v in waits_by_client.values())
    log = getattr(sim, "batch_log", [])
    sizes = [len(r["chosen"]) for r in log]
    cap = sim.verifier.max_batch_size
    bl = {}
    for r in recs:
        bl[r["verification_batch_id"]] = (r["verification_batch_size"], r["verification_batch_processed_tokens"])
    z = np.array([r["z_queue"] for r in recs])
    gam = [r["gamma"] for r in recs]
    rates = sorted(s["per_client_rates"].values()) if isinstance(s["per_client_rates"], dict) else sorted(s["per_client_rates"])
    out = {"x_min": s["x_min"], "x_mean": s["x_mean"], "Y": s["Y"], "jain": float(np.sum(rates) ** 2 / (len(rates) * np.sum(np.square(rates)))) if rates else None,
           "util": s["util"], "batch_size_mean": s["batch_size_mean"], "queue_wait_ms_mean": s["queue_wait_ms_mean"],
           "queue_wait_ms_max": float(max_w), "wait_spread_ms": float(max(mean_w.values()) - min(mean_w.values())),
           "gamma_mean": s["gamma_mean"], "gamma_dist": s["gamma_dist"], "z_mean": float(z.mean()), "z_max": float(z.max()),
           "round_ms_mean": s["round_ms_mean"], "ledger_violations": s["ledger_violations"], "n_batches": len(set(bl)),
           "max_batch_seen": max((v[0] for v in bl.values()), default=0), "cap": cap,
           "max_processed_tokens_seen": max((v[1] for v in bl.values()), default=0),
           "batches_eligible_gt_cap": float(np.mean([r["n_eligible"] > r["cap"] for r in log])) if log else None,
           "forced_service_fraction": float(np.mean([bool(r.get("forced_service")) for r in recs])),
           "invalid": s["invalid"], "rates_sorted": rates,
           "x_client0": (float(s["per_client_rates"][0]) if isinstance(s["per_client_rates"], dict) and 0 in s["per_client_rates"] else None),
           "x_p10": float(np.quantile(rates, 0.1)) if rates else None,
           "slo_deficit_sum": float(sum(max(0.0, sim.spec.x_req - x) for x in rates)),            # sum_i max(0, x_req - x_i)  [tokens/s]
           "frac_clients_below_slo": float(np.mean([x < sim.spec.x_req for x in rates])) if rates else None,
           "skipped_service_fraction": float(np.mean([sk > 0 for r_ in log for sk in r_.get("skips_before_service", [])])) if log else None}
    return out


def task_to_spec(t: dict) -> ch.Spec:
    """Serialisable task -> Spec (so multiprocessing workers rebuild everything locally).  Keys beyond arm/n/pop/seed/cap are optional."""
    t = dict(t)
    arm, n, pop, seed, cap = t.pop("arm"), t.pop("n"), t.pop("pop"), t.pop("seed", 1), t.pop("cap", 16)
    V, x_req = t.pop("V", 50.0), t.pop("x_req", 3.0)
    cost = CostModel(a_ms=t.pop("cost_a", 52.0), b_ms_per_token=t.pop("cost_b", 0.19), b_gamma_ms_per_token=t.pop("cost_bg", None))
    c3 = C3Params(max_batch_size=cap, V=V, x_req_tps=x_req, cost=cost, t_ref_ms=t.pop("t_ref_ms", 52.0), max_skips=t.pop("max_skips", 2),
                  service_time_cap_ms=t.pop("service_time_cap_ms", float("inf")), processed_token_budget=t.pop("token_budget_processed", 4096),
                  window_ms=t.pop("window_ms", 5000.0), use_deficit_weight=t.pop("use_deficit_weight", True), read_lag_ms=t.pop("read_lag_ms", 0.0))
    return make_spec(arm, n, pop, seed, cap=cap, c3=c3, V=V, x_req=x_req, **t)
