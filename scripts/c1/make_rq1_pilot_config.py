"""Derive the RQ1 pilot config from the frozen canonical tfa33 config (CPU only).  Only the listed keys change; the
runtime block, verifier semantics (batch_wall, price zero, fixed budget 100), models and decoding are inherited unchanged.
Usage: python scripts/c1/make_rq1_pilot_config.py [out.yaml]
"""
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs" / "canonical_batchwall_tritonfa33_v1.yaml"
OUT = ROOT / "configs" / "rq1_pilot_fixed_gamma2_tritonfa33.yaml"


def build() -> dict:
    cfg = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    deg = lambda v: [float(v), float(v)]
    c = cfg["clients"]
    # homogeneous symmetric clients (degenerate ranges): the pilot varies N only
    c["rtt_ms_range"] = deg(50.0)
    c["uplink_mbps_range"] = deg(20.0)
    c["downlink_mbps_range"] = deg(40.0)
    c["draft_tokens_per_second_range"] = deg(40.0)
    c["draft_fixed_latency_ms_range"] = deg(10.0)
    c["uplink_base_snr_db_range"] = deg(14.0)
    c["downlink_base_snr_db_range"] = deg(16.0)
    c["uplink_snr_jitter_db_range"] = deg(2.0)
    c["downlink_snr_jitter_db_range"] = deg(2.0)
    c["snr_period_ms_range"] = deg(1000.0)
    c["packet_loss_range"] = deg(0.0)
    c["draft_speed_multiplier_range"] = deg(1.0)       # A1: no per-device draft slowdown (base config had 0.7-1.5)
    c["snr_block_duration_ms_range"] = deg(100.0)
    c["shared_wireless"] = {"downlink_allocator": "proportional", "uplink_allocator": "proportional",
                            "total_downlink_bandwidth_mhz": 80.0, "total_uplink_bandwidth_mhz": 40.0}
    # one simple fixed policy: fixed depth gamma=2, FCFS batching, no learning in the loop
    cfg["controller"]["policy"] = "fixed_gamma"
    cfg["controller"]["fixed_gamma"] = 2
    cfg["controller"]["exploration_epsilon"] = 0.0
    cfg["controller"]["min_exploration_rounds"] = 0
    cfg["verification_batching"]["scheduler"] = "fcfs"
    # statistics / population / causality / draft emulation (C1)
    s = cfg["simulation"]
    s["population_seed"] = 20261020
    s["seed"] = 1                                   # = run_seed; overridden per run by the launcher
    s["population_n_max"] = 16
    s["questions_per_client"] = 64
    s["causal_virtual_time"] = True
    s["causality_ledger"] = True
    s["draft_execution_mode"] = "isolated_profile"
    s["draft_profile"] = {"overhead_ms": 0.0, "per_token_ms": 22.2}
    cfg["experiment"]["warmup_seconds"] = 30.0
    cfg["experiment"]["measurement_seconds"] = 120.0
    cfg["experiment"]["early_stop_margin_seconds"] = 10.0
    cfg["dataset"]["num_questions"] = 16 * 64
    cfg["capacity_search"]["measurement_seconds_final"] = 120.0
    cfg["capacity_search"]["warmup_seconds"] = 30.0
    return cfg


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    header = ("# RQ1 minimal pilot config (C1): derived from canonical_batchwall_tritonfa33_v1.yaml by\n"
              "# scripts/c1/make_rq1_pilot_config.py.  NOT submitted; descriptive pilot, no certified boundaries.\n")
    out.write_text(header + yaml.safe_dump(build(), sort_keys=True), encoding="utf-8")
    print("wrote", out)
