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


def test_runtime_h_manifest_matches_the_frozen_identity_record():
    import hashlib
    import json
    ident = json.loads((ROOT / "results" / "runtime_identity" / "V1-batchwall-tritonfa33.json").read_text())
    man = ROOT / PILOT["runtime"]["manifest"]
    assert hashlib.sha256(man.read_bytes()).hexdigest() == ident["qualified_execution"]["manifest_sha256"] or \
        hashlib.sha256(man.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == ident["qualified_execution"]["manifest_sha256"]
    assert PILOT["runtime"]["identity"] == ident["identity"] and PILOT["runtime"]["require_triton_fa"] is True


def test_live_pilot_switches_are_on():
    s = PILOT["simulation"]
    assert s["causal_virtual_time"] is True and s["causality_ledger"] is True
    assert s["draft_execution_mode"] == "isolated_profile" and s["draft_profile"]["per_token_ms"] == 22.2
    assert PILOT["clients"]["draft_speed_multiplier_range"] == [1.0, 1.0]
    assert PILOT["controller"]["verifier_price_mode"] == "zero"


# ---------------- C1.2 extension config (exploratory; the original pilot config stays immutable) ----------------
EXT = yaml.safe_load((ROOT / "configs" / "rq1_extension_fixed_gamma2_tritonfa33.yaml").read_text(encoding="utf-8"))
CATS = __import__("json").loads((ROOT / "results" / "rq1_extension" / "prompt_catalogs.json").read_text())


def _flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def test_extension_differs_from_the_pilot_only_in_catalog_and_population_size():
    a, b = _flat(PILOT), _flat(EXT)
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert changed == {"dataset.split", "dataset.num_questions", "simulation.population_n_max", "simulation.expected_catalog_hash"}, changed
    assert EXT["dataset"]["split"] == "train" and EXT["dataset"]["num_questions"] == 32 * 64
    assert EXT["simulation"]["population_n_max"] * EXT["simulation"]["questions_per_client"] == EXT["dataset"]["num_questions"]


def test_extension_catalog_is_frozen_unique_and_documented_as_incompatible_with_the_pilot():
    ext, old = CATS["extension_catalog"], CATS["original_pilot_catalog"]
    assert EXT["simulation"]["expected_catalog_hash"] == ext["catalog_hash"] != old["catalog_hash"]
    assert ext["duplicates"]["duplicate_slots"] == 0 and ext["duplicates"]["dataset_names"] == ["gsm8k"] and ext["duplicates"]["id_prefixes"] == ["gsm8k-train"]
    assert old["duplicates"]["id_prefixes"] == ["gsm8k-test"] and CATS["prompt_overlap_between_catalogs"] == 0
    assert "NOT directly comparable" in CATS["compatibility"]
    assert ext["n_slots"] == 2048 and len(ext["client_block_hashes"]) == 32


def test_extension_keeps_every_frozen_scientific_setting():
    assert EXT["runtime"] == BASE["runtime"] and EXT["controller"]["policy"] == "fixed_gamma" and EXT["controller"]["fixed_gamma"] == 2
    assert EXT["verification_batching"]["scheduler"] == "fcfs" and EXT["verification_batching"]["service_time_model"] == "batch_wall"
    assert EXT["controller"]["verifier_price_mode"] == "zero" and EXT["simulation"]["causal_virtual_time"] is True
    assert EXT["simulation"]["draft_execution_mode"] == "isolated_profile" and EXT["simulation"]["draft_profile"]["per_token_ms"] == 22.2
    assert (EXT["experiment"]["warmup_seconds"], EXT["experiment"]["measurement_seconds"]) == (30.0, 120.0)


def test_extension_job_script_has_the_registered_order_and_walltime():
    txt = (ROOT / "scripts" / "c1" / "run_rq1_extension_job.sh").read_text()
    assert "--time=01:15:00" in txt and "12 8 16 24 32" in txt
    pop = MasterPopulation(EXT["clients"], EXT["simulation"]["population_seed"], 32)
    assert pop.fingerprint(16) == MasterPopulation(PILOT["clients"], PILOT["simulation"]["population_seed"], 16).fingerprint(16)   # static profiles unchanged
