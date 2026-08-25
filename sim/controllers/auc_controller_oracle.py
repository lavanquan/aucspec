"""auc_controller.py with UCB/Bernstein learning replaced by a static offline
acceptance-rate estimate alpha_hat_i, profiled once before the run.

Not a direct Px solver -- same controller architecture as auc_controller.py,
just alpha_hat_i plugged in as a constant instead of learned online. Profile
alpha_hat_i from the first half of the trace only and evaluate on the second
half to avoid data leakage. See TASKS.md T2.3.
"""
