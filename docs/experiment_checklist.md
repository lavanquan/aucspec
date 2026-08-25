# Experiment Checklist

This checklist is the shortest practical path from "is the project alive?" to
"do we trust the reported frontier/AUC numbers?".

## Recommended Order

| Step | Command | Why run it | Typical time | Read this output |
|---|---|---|---|---|
| 1 | `python scripts/quick_check.py` | Fast plumbing check for core logic, configs, and bilevel wiring. | A few seconds | Terminal output |
| 2 | `python scripts/validate_correctness.py --config configs/default.yaml` | Confirms greedy speculative decoding matches greedy target-only output token-for-token. | Minutes to longer, depending on model and hardware | Terminal output; any mismatch is a blocker |
| 3 | `python scripts/run_simulation.py --config configs/experiments/smoke_simulation_one_client_per_gpu.yaml` | Smoke test for `simulation` mode with one virtual client per draft GPU. | A few minutes | `results/smoke_simulation_one_client_per_gpu_rounds.csv` |
| 4 | `python scripts/run_simulation.py --config configs/experiments/smoke_shared_gpu_emulation.yaml` | Smoke test for `shared_gpu_emulation` mode with visible GPU activity and wall-clock contention. | A few minutes to longer | `results/smoke_shared_gpu_emulation_rounds.csv` |
| 5 | `python scripts/summarize.py results/...rounds.csv` | Quick readout of goodput, TTFT, fairness, utilization, and acceptance. | A few seconds | Terminal summary |
| 6 | `python scripts/run_operating_point.py --config configs/default.yaml --output-dir results/operating_point` | Measures one fixed operating point across one or more seeds. | Minutes to tens of minutes | `results/operating_point/operating_point_summary.csv` |
| 7 | `python scripts/sweep_schedulers.py --config configs/default.yaml` | Compares verification batching schedulers on the same setup. | Tens of minutes | `results/scheduler_sweep/scheduler_summary.csv` |
| 8 | `python scripts/sweep_controller_v.py --config configs/default.yaml` | Sweeps Lyapunov parameter `V` to expose throughput vs fairness/backlog tradeoffs. | Tens of minutes | `results/controller_v_summary/controller_v_summary.csv` |
| 9 | `python scripts/run_baselines.py --config configs/default.yaml` | Runs model/controller/network/learning baseline families. | Long | `results/baselines/baseline_summary.csv` |
| 10 | `python scripts/run_ablations.py --config configs/default.yaml` | Runs paper-style ablations. | Long | `results/ablations/ablation_summary.csv`, `results/ablations/paper_table.csv` |
| 11 | `python scripts/sweep_min_interactivity.py --config configs/default.yaml` | Builds the interactivity-goodput frontier and computes AUC. | Very long | `frontier.csv`, `auc.json`, `paper_table.csv`, `frontier.png` |
| 12 | `python scripts/optimize_configuration.py --config configs/default.yaml` | Solves the outer bilevel configuration search over `xi`. | Longest | `results/configuration_search/.../outer_problem.json`, `evaluations.csv`, best solution JSON |

## What Each Output Is For

| Output | What to use it for |
|---|---|
| `rounds.csv` | Deep debugging. Per-round trace with timing, queue, batching, network, and token-accounting columns. |
| `operating_point_summary.csv` | Stable comparison of one chosen configuration across seeds. |
| `scheduler_summary.csv` | Scheduler comparison. |
| `controller_v_summary.csv` | Throughput/fairness/backlog tradeoff over `V`. |
| `baseline_summary.csv` | Head-to-head baseline comparison across model/controller/network/learning families. |
| `ablation_summary.csv` | Ablation deltas. |
| `paper_table.csv` | Headline table for reporting goodput, AUC, fairness, and TTFT. |
| `frontier.csv` | Final interactivity-goodput frontier after upper concave hull processing. |
| `auc.json` | AUC value and frontier metadata. |
| `frontier.png` | Quick visual check of the frontier shape. |

## Minimal Safe Workflow

Use this order when you want to avoid wasting GPU time on broken runs:

1. `quick_check.py`
2. `validate_correctness.py`
3. `smoke_simulation_one_client_per_gpu.yaml`
4. `summarize.py`
5. `run_operating_point.py`
6. `sweep_schedulers.py` or `sweep_controller_v.py`
7. `run_baselines.py`
8. `run_ablations.py`
9. `sweep_min_interactivity.py`
10. `optimize_configuration.py`

## Useful Monitoring Commands

For live GPU monitoring during smoke tests or operating-point runs:

```bash
nvidia-smi dmon -s pucm -d 1
```

For a quick summary of a completed trace:

```bash
python scripts/summarize.py results/rounds.csv
```

## Notes

- Treat any `validate_correctness.py` mismatch as a hard blocker before running
  frontier, AUC, or bilevel experiments.
- `simulation` mode is the right mode for many independent edge devices with
  profiled latency.
- `shared_gpu_emulation` is useful when you intentionally want real wall-clock
  contention across the configured draft GPUs.
- If a run seems slow while GPU memory is occupied but `volatile gpu-util`
  stays near zero, inspect verifier overhead and CPU-side postprocessing in
  addition to GPU kernels.
