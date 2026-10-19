"""Population identity tests (audit item M4)."""
import copy
import json

import pytest

from edge_specsim.population_seeds import MasterPopulation, client_rng, draw_static_client

CLIENTS_CFG = {
    "rtt_ms_range": [15.0, 100.0],
    "uplink_mbps_range": [5.0, 50.0],
    "draft_tokens_per_second_range": [20.0, 60.0],
    "client_classes": {"fast": {"probability": 0.5, "draft_slowdown": 1.0},
                       "slow": {"probability": 0.5, "draft_slowdown": 2.0, "rtt_ms": [80.0, 120.0]}},
}


def test_prefix_identity_across_n():
    m16 = MasterPopulation(CLIENTS_CFG, 7, 16)
    m8 = MasterPopulation(CLIENTS_CFG, 7, 8)
    assert m16.prefix(8) == m8.prefix(8) == m8.clients
    assert m16.fingerprint(8) == m8.fingerprint(8) == m8.fingerprint()
    assert m16.fingerprint(8) != m16.fingerprint(9)
    for n in (1, 4, 8, 12):
        assert m16.prefix(n) == [draw_static_client(CLIENTS_CFG, 7, i) for i in range(n)]


def test_independent_of_run_seed_and_policy():
    """run_seed / policy are not inputs of the master population, so the fingerprint cannot depend on them."""
    cfg_a = copy.deepcopy(CLIENTS_CFG)
    cfg_b = copy.deepcopy(CLIENTS_CFG)
    cfg_b["controller_policy_irrelevant"] = "goodspeed"  # extra key unrelated to static properties
    fa = MasterPopulation(cfg_a, 3, 8).fingerprint(8)
    fb = MasterPopulation(cfg_b, 3, 8).fingerprint(8)
    assert fa == fb
    import inspect
    sig = inspect.signature(MasterPopulation.__init__)
    assert "run_seed" not in sig.parameters and "seed" not in sig.parameters and "policy" not in sig.parameters


def test_population_seed_changes_population():
    assert MasterPopulation(CLIENTS_CFG, 1, 8).fingerprint() != MasterPopulation(CLIENTS_CFG, 2, 8).fingerprint()


def test_client_i_does_not_depend_on_other_clients():
    assert draw_static_client(CLIENTS_CFG, 5, 3) == MasterPopulation(CLIENTS_CFG, 5, 40).clients[3]
    assert client_rng(5, 3).random() == client_rng(5, 3).random()
    assert client_rng(5, 3).random() != client_rng(5, 4).random()


def test_degenerate_ranges_give_homogeneous_clients():
    cfg = {"rtt_ms_range": [50.0, 50.0], "uplink_mbps_range": [20.0, 20.0],
           "draft_tokens_per_second_range": [40.0, 40.0], "draft_fixed_latency_ms_range": [10.0, 10.0],
           "uplink_base_snr_db_range": [12.0, 12.0], "downlink_base_snr_db_range": [14.0, 14.0],
           "uplink_snr_jitter_db_range": [2.0, 2.0], "downlink_snr_jitter_db_range": [2.0, 2.0],
           "snr_period_ms_range": [1000.0, 1000.0]}
    pop = MasterPopulation(cfg, 11, 6)
    fields = ["rtt_ms", "uplink_mbps", "draft_tokens_per_second_base", "uplink_base_snr_db", "snr_period_ms"]
    for f in fields:
        assert len({getattr(c, f) for c in pop.clients}) == 1
    # the SNR phase is still per-client random (channel phase, not a capacity-relevant property)
    assert len({c.snr_phase_rad for c in pop.clients}) == 6


def test_prompt_assignment_is_part_of_the_fingerprint():
    ids = [[f"p{i}_{j}" for j in range(3)] for i in range(8)]
    a = MasterPopulation(CLIENTS_CFG, 1, 8, ids)
    other = [row[:] for row in ids]
    other[2][1] = "different"
    b = MasterPopulation(CLIENTS_CFG, 1, 8, other)
    assert a.fingerprint(2) == b.fingerprint(2)        # prefix before the change is identical
    assert a.fingerprint(3) != b.fingerprint(3)        # client 2's prompt block changed
    assert json.dumps(a.fingerprint())                 # serialisable
