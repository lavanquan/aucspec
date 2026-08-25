"""AUC-frontier controller: our proposed inner-problem (Px) runtime policy.

Drift-plus-penalty controller (Theorem 3) decomposed into three subproblems:
  (i)   device-side speculation index gamma_i(t), closed form (eq. 13),
        priced by broadcast queues lambda(t) = Q(t), mu_i(t) = Q_i(t);
  (ii)  server-side batching: knapsack on token budget, greedy by
        omega_i / (gamma_i + 1);
  (iii) radio allocation: square-root waterfilling.
Virtual queues Z_i(t), Q(t), Q_i(t) per eq. (12). V is a config parameter
(needed for Exp 3), never hardcoded. Uses UCB/Bernstein censored learning
for alpha_hat_i(t) (Theorem 4) unless disabled -- see auc_controller_oracle.py.
See TASKS.md T2.1, T2.2.
"""
