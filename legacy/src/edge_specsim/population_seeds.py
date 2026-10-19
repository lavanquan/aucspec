"""Fixed nested master population with separate population_seed / run_seed (audit item M4).

Problem fixed
-------------
`EdgeSpecSimulator._make_clients` drew every static client property (RTT, uplink, SNR, draft speed, ...)
from the global `random` module seeded with `simulation.seed`.  Different run-seeds therefore produced
different DEVICES, and the draws for client i also depended on N (class assignment for all N clients was
drawn first).  "Client i at seed 1" and "client i at seed 2" were different clients, so per-client
confidence intervals mixed hardware variation with execution noise, and nesting in N only held for a
single seed.

Design
------
* `population_seed` fixes the master population: for client index i the static properties are drawn from
  `random.Random(sha256(population_seed, i))`.  Client i therefore does not depend on N, N_max, the
  run-seed, the policy, or any other client.  Exact prefix restriction: population(N) = master[:N].
* `run_seed` (the existing `simulation.seed`) only drives stochastic EXECUTION: controller exploration,
  scheduler tie randomness, channel block-fading noise (`channel_seed`), target sampling.
* The fingerprint covers population_seed, the static properties and the prompt assignment (never the
  run-seed, never the policy).  `prefix_fingerprint(n)` is the fingerprint of the first n clients.

Legacy behaviour is untouched: when `simulation.population_seed` is absent the simulator keeps the old
code path and its results must still be read as "population redrawn per run-seed".
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass

STATIC_RANGE_KEYS = (
    # (record field, config key, default range)
    ("draft_tokens_per_second_base", "draft_tokens_per_second_range", [20.0, 60.0]),
    ("draft_fixed_latency_ms", "draft_fixed_latency_ms_range", [5.0, 20.0]),
    ("rtt_ms", "rtt_ms", None),
    ("uplink_mbps", "uplink_mbps", None),
    ("downlink_mbps", "downlink_mbps", None),
    ("packet_loss", "packet_loss", [0.0, 0.0]),
    ("uplink_base_snr_db", "uplink_base_snr_db", [8.0, 18.0]),
    ("downlink_base_snr_db", "downlink_base_snr_db", [10.0, 20.0]),
    ("uplink_snr_jitter_db", "uplink_snr_jitter_db", [1.0, 4.0]),
    ("downlink_snr_jitter_db", "downlink_snr_jitter_db", [1.0, 4.0]),
    ("snr_period_ms", "snr_period_ms", [500.0, 3000.0]),
    ("snr_block_duration_ms", "snr_block_duration_ms", [100.0, 100.0]),
)


@dataclass(frozen=True)
class StaticClient:
    """Everything about client i that must not change across run-seeds, N or policies."""

    index: int
    class_name: str
    draft_slowdown: float
    draft_tokens_per_second_base: float
    draft_fixed_latency_ms: float
    rtt_ms: float
    uplink_mbps: float
    downlink_mbps: float
    packet_loss: float
    uplink_base_snr_db: float
    downlink_base_snr_db: float
    uplink_snr_jitter_db: float
    downlink_snr_jitter_db: float
    snr_period_ms: float
    snr_block_duration_ms: float
    snr_phase_rad: float


def client_rng(population_seed: int, index: int) -> random.Random:
    digest = hashlib.sha256(f"aucspec-population:{int(population_seed)}:{int(index)}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _uniform(rng: random.Random, values) -> float:
    lo, hi = values
    return rng.uniform(lo, hi)


def draw_static_client(clients_cfg: dict, population_seed: int, index: int) -> StaticClient:
    """Draw client `index` of the master population.  Depends only on (clients_cfg, population_seed, index)."""
    rng = client_rng(population_seed, index)
    class_cfgs = clients_cfg.get("client_classes")
    if class_cfgs:
        names = list(class_cfgs.keys())
        probs = [float(class_cfgs[n].get("probability", 0.0)) for n in names]
        total = sum(probs)
        if total <= 0.0:
            raise ValueError("clients.client_classes probabilities must sum to a positive value")
        name = rng.choices(names, weights=[p / total for p in probs], k=1)[0]
        class_cfg = class_cfgs[name]
    else:
        name, class_cfg = "default", {}

    slowdown = float(class_cfg.get("draft_slowdown", 1.0))
    values: dict[str, float] = {}
    for fld, key, default in STATIC_RANGE_KEYS:
        if fld == "draft_tokens_per_second_base":
            rng_vals = clients_cfg.get(key, default)
        elif fld == "draft_fixed_latency_ms":
            rng_vals = class_cfg.get(key, clients_cfg.get(key, default))
        elif fld == "rtt_ms":
            rng_vals = class_cfg.get("rtt_ms", clients_cfg["rtt_ms_range"])
        elif fld == "uplink_mbps":
            rng_vals = class_cfg.get("uplink_mbps", clients_cfg["uplink_mbps_range"])
        elif fld == "downlink_mbps":
            rng_vals = class_cfg.get("downlink_mbps", clients_cfg.get("downlink_mbps_range", clients_cfg["uplink_mbps_range"]))
        elif fld == "packet_loss":
            rng_vals = class_cfg.get("packet_loss", clients_cfg.get("packet_loss_range", default))
        else:
            rng_vals = class_cfg.get(key, clients_cfg.get(f"{key}_range", default))
        values[fld] = _uniform(rng, rng_vals)
    values["snr_phase_rad"] = rng.uniform(0.0, 2.0 * math.pi)
    return StaticClient(index=int(index), class_name=name, draft_slowdown=slowdown, **values)


class MasterPopulation:
    """Master population of `n_max` static clients plus a fixed prompt block per client."""

    def __init__(self, clients_cfg: dict, population_seed: int, n_max: int,
                 prompt_ids_by_client: list[list[str]] | None = None) -> None:
        if n_max <= 0:
            raise ValueError("n_max must be positive")
        self.population_seed = int(population_seed)
        self.n_max = int(n_max)
        self.clients = [draw_static_client(clients_cfg, self.population_seed, i) for i in range(self.n_max)]
        self.prompt_ids = prompt_ids_by_client

    def prefix(self, n: int) -> list[StaticClient]:
        if n < 0 or n > self.n_max:
            raise ValueError(f"requested prefix {n} outside [0, {self.n_max}]")
        return self.clients[:n]

    def fingerprint(self, n: int | None = None) -> str:
        """Hash of the first n clients (default: all).  Excludes run_seed and policy by construction."""
        n = self.n_max if n is None else int(n)
        payload = {
            "population_seed": self.population_seed,
            "clients": [asdict(c) for c in self.prefix(n)],
            "prompts": None if self.prompt_ids is None else [list(map(str, p)) for p in self.prompt_ids[:n]],
        }
        # NOTE: population_seed itself is part of the identity; two masters with the same static values but
        # different seeds are still different experiments.
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    prefix_fingerprint = fingerprint
