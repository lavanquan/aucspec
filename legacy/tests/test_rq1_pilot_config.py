"""The RQ1 pilot config must change ONLY what the plan says and inherit the frozen runtime/verifier settings."""
from pathlib import Path

import yaml

from edge_specsim.population_seeds import MasterPopulation

ROOT = Path(__file__).resolve().parents[2]
BASE = yaml.safe_load((ROOT / "configs" / "canonical_batchwall_tritonfa33_v1.yaml").read_text(encoding="utf-8"))
PILOT = yaml.safe_load((ROOT / "configs" / "rq1_pilot_fixed_gamma2_tritonfa33.yaml").read_text(encoding="utf-8"))


def test_generated_file_is_up_to_date_with_the_generator():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mk", ROOT / "scripts" / "c1" / "make_rq1_pilot_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.build() == PILOT


def test_frozen_runtime_and_verifier_semantics_are_inherited_unchanged():
    assert PILOT["runtime"] == BASE["runtime"]
    assert PILOT["models"] == BASE["models"]
    assert PILOT["decoding"] == BASE["decoding"]
    vb_b, vb_p = BASE["verification_batching"], PILOT["verification_batching"]
    for k in ("batch_wait_ms", "budget_mode", "max_batch_size", "service_time_model", "verify_token_budget",
              "target_replication_factor"):
        assert vb_p[k] == vb_b[k], k
    assert PILOT["controller"]["verifier_price_mode"] == "zero"
    assert PILOT["controller"]["server_price_scale"] == BASE["controller"]["server_price_scale"]
    assert PILOT["dataset"]["max_new_tokens"] == BASE["dataset"]["max_new_tokens"] == 96


def test_pilot_is_a_single_simple_fixed_policy_without_pricing():
    assert PILOT["controller"]["policy"] == "fixed_gamma" and PILOT["controller"]["fixed_gamma"] == 2
    assert PILOT["controller"]["exploration_epsilon"] == 0.0
    assert PILOT["verification_batching"]["scheduler"] == "fcfs"


def test_statistical_and_causal_settings():
    s = PILOT["simulation"]
    assert s["population_seed"] is not None and s["nested_population"] and s["causal_virtual_time"]
    assert s["draft_execution_mode"] == "isolated_profile" and s["draft_profile"]["per_token_ms"] > 0
    assert PILOT["experiment"]["measurement_seconds"] == 120.0 and PILOT["experiment"]["early_stop_margin_seconds"] == 10.0
    assert s["questions_per_client"] * s["population_n_max"] <= 1319        # GSM8K test split


def test_population_is_homogeneous_and_independent_of_run_seed():
    pop = MasterPopulation(PILOT["clients"], PILOT["simulation"]["population_seed"], 16)
    for f in ("rtt_ms", "uplink_mbps", "downlink_mbps", "draft_tokens_per_second_base", "draft_fixed_latency_ms",
              "uplink_base_snr_db", "snr_period_ms", "packet_loss"):
        assert len({getattr(c, f) for c in pop.clients}) == 1, f
    assert pop.fingerprint(8) == MasterPopulation(PILOT["clients"], PILOT["simulation"]["population_seed"], 8).fingerprint(8)
