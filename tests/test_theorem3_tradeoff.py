"""T2.4: sanity-check Theorem 3's V-tradeoff with a simple closed-loop run --
not a tight numeric bound on the constant C (TASKS.md explicitly allows a
"rough" C), but the two qualitative, individually falsifiable claims eq. (14)
makes: backlog grows with V, and the interactivity-constraint shortfall
shrinks (or at least does not worsen) as V grows. Uses the oracle (known
alpha, no UCB) so this isolates the controller's own behaviour from Theorem
4's learning gap, matching Theorem 3's "known {alpha_i}" hypothesis.
"""

from __future__ import annotations

import random

from sim.controllers.auc_controller import StreamLinkState
from sim.controllers.auc_controller_oracle import build_oracle_controller
from sim.core.acceptance import sample_round
from sim.core.server import batch_token_count


def run_controller_simulation(V: float, num_slots: int, seed: int) -> dict:
    # Round/slot magnitudes are chosen so per-round server compute time
    # (theta_f * Gamma(B)) is comparable to slot_seconds at gamma_max -- a
    # near-saturated system where Q(t) actually builds up -- and x_target
    # sits comfortably below the system's max achievable rate (empirically
    # ~152 tokens/s here) so the interactivity constraint is always met and
    # Q(t) alone (not constraint violation) drives the V-backlog signal.
    # Near x_target's feasibility boundary the transient dynamics are more
    # tangled (a real V-tradeoff curve is Exp 3's job, Phase 5); this test
    # stays in the clean, comfortably-feasible regime on purpose.
    n_streams = 3
    alpha = 0.75
    x_target = 148.0  # tokens/s, below the ~152 tokens/s max at gamma_max=4
    tau_d = 0.003
    kappa = 500.0
    uplink_rate = 2.0e6
    spectral_eff = 4.0
    slot_seconds = 0.02

    controller = build_oracle_controller(
        alpha_hat=[alpha] * n_streams,
        n_streams=n_streams,
        gamma_max=4,
        V=V,
        min_interactivity_x=x_target,
        kappa_bits_per_token=kappa,
        theta_f_seconds_per_token=0.005,
        total_bandwidth_hz=1.0e7,
        gamma_budget=1000.0,  # generous: server never the bottleneck
    )

    links = [
        StreamLinkState(tau_d_seconds=tau_d, uplink_rate_bps=uplink_rate, spectral_efficiency=spectral_eff)
        for _ in range(n_streams)
    ]
    rng = random.Random(seed)

    cumulative_tokens = [0.0] * n_streams
    backlog_samples: list[float] = []
    warmup = num_slots // 5

    for t in range(num_slots):
        decision = controller.decide(t=t, links=links)
        tokens_delivered = [0.0] * n_streams
        for i in decision.selected_batch:
            tokens, _ = sample_round(decision.gammas[i], alpha, rng)
            tokens_delivered[i] = float(tokens)
            cumulative_tokens[i] += tokens
        verification_tokens = batch_token_count([decision.gammas[i] for i in decision.selected_batch])
        controller.update_queues(decision, tokens_delivered, links, verification_tokens, slot_seconds)

        if t >= warmup:
            backlog_samples.append(
                sum(controller.queues.Z) + controller.queues.Q + sum(controller.queues.Qi)
            )

    elapsed_seconds = num_slots * slot_seconds
    empirical_x = [tokens / elapsed_seconds for tokens in cumulative_tokens]
    shortfall = [max(0.0, x_target - x) for x in empirical_x]

    return {
        "avg_backlog": sum(backlog_samples) / len(backlog_samples),
        "avg_shortfall": sum(shortfall) / len(shortfall),
        "empirical_x": empirical_x,
    }


def test_backlog_grows_with_v():
    small = run_controller_simulation(V=0.05, num_slots=6000, seed=1)
    large = run_controller_simulation(V=5.0, num_slots=6000, seed=1)
    assert large["avg_backlog"] > small["avg_backlog"]


def test_larger_v_does_not_worsen_constraint_tracking():
    small = run_controller_simulation(V=0.05, num_slots=6000, seed=2)
    large = run_controller_simulation(V=5.0, num_slots=6000, seed=2)
    # eq. (14): time-average interactivity meets the x_target constraint
    # within O(V/t). In this comfortably-feasible operating point both V
    # values satisfy the constraint with zero shortfall; check larger V
    # does not introduce one (small tolerance for finite-T Monte Carlo noise).
    assert large["avg_shortfall"] <= small["avg_shortfall"] + 0.05
