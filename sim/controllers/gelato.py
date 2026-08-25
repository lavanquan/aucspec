"""GELATO baseline: Lyapunov drift-plus-penalty device-edge token offloading.

Closest competitor to auc_controller.py -- also drift-plus-penalty, but
optimizes a single goodput point rather than sweeping the whole frontier.
Read the GELATO paper before modifying; document every divergence from
auc_controller.py inline. See TASKS.md T3.5.
"""
