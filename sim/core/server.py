"""Shared verification server: roofline latency model and batch verification.

Roofline T_v(B) = theta0 + max(theta_m(B), theta_f * Gamma(B)), knee Gamma_dagger
= theta_m(B) / theta_f (eq. 3). See TASKS.md T1.3.
"""
