"""Fixed multi-SLO baseline (simplified AdaServe-style, EuroSys'26).

Each request has its own latency target; gamma chosen to meet that target.
Not a full hardware-aware speculation-tree construction. See TASKS.md T3.4.
"""
