"""Deterministic nested population catalog for the N* capacity search.

CAPACITY_AUC_NSTAR_IMPLEMENTATION.md Section 4: a naive `_make_clients(N)`
per candidate re-samples the random client composition, so `N=18` and
`N=21` are not nested systems and feasibility need not be monotone. This
module builds ONE frozen catalog of up to `n_max` client specs per seed;
every candidate concurrency uses a prefix (homogeneous) or an
`m`-scaled block (fixed class mix) of that SAME catalog, so increasing
concurrency only ADDS clients and never resamples an existing one.

The catalog holds *specs* (plain dataclasses), not live `ClientProfile`
objects — the simulator turns a spec list into clients. This keeps the
module import-light and CPU-only so it is unit-testable without vLLM.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ClientSpec:
    """Everything needed to reconstruct one client deterministically. The
    simulator reads these fields when materialising a `ClientProfile`; any
    field the simulator samples randomly must instead be frozen here."""

    stable_id: int              # 0-based index into the catalog; never reused
    client_seed: int            # per-client seed for any residual RNG the sim needs
    class_name: str             # "default" for homogeneous, else the class label
    dataset_name: str
    dataset_index: int          # which prompt shard / offset this client draws
    draft_slowdown: float = 1.0
    rtt_ms: float = 40.0
    uplink_mbps: float = 20.0
    downlink_mbps: float = 40.0
    draft_fixed_latency_ms: float = 10.0
    draft_tokens_per_second: float = 40.0
    packet_loss: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PopulationTemplate:
    """Fixed class mix `q`. Homogeneous default is a single unnamed class.

    Section 4.2: candidates are `N(m) = m * sum_c q_c`; searching integer
    `m` keeps class proportions exactly fixed for every candidate.
    """

    counts: tuple[tuple[str, int], ...] = (("default", 1),)

    @classmethod
    def homogeneous(cls) -> "PopulationTemplate":
        return cls(counts=(("default", 1),))

    @classmethod
    def from_config(cls, spec) -> "PopulationTemplate":
        """`spec` is a list like `[1]` or `[1,1,1]` (counts only, class
        names default to c0,c1,...), or a dict `{class_name: count}`, or a
        list of `[class_name, count]` pairs."""
        if spec is None:
            return cls.homogeneous()
        if isinstance(spec, dict):
            return cls(counts=tuple((str(k), int(v)) for k, v in spec.items()))
        if isinstance(spec, (list, tuple)):
            if spec and isinstance(spec[0], (list, tuple)):
                return cls(counts=tuple((str(k), int(v)) for k, v in spec))
            return cls(counts=tuple((f"c{i}", int(v)) for i, v in enumerate(spec)))
        raise TypeError(f"unsupported population_template spec: {spec!r}")

    @property
    def block_size(self) -> int:
        return sum(c for _, c in self.counts)

    def class_sequence(self) -> list[str]:
        """One block of class labels in catalog order, e.g.
        ('easy',2),('hard',1) -> ['easy','easy','hard']."""
        seq: list[str] = []
        for name, count in self.counts:
            seq.extend([name] * int(count))
        return seq


class PopulationCatalog:
    """Frozen list of `ClientSpec` for one capacity-search seed.

    Build once with `n_max` (>= the largest candidate you will ever test,
    typically `N_ref`), then read a prefix with `prefix(n)` or an
    `m`-scaled block with `scale(m)`. Both return a *slice* of the same
    underlying list, so the nesting property holds by construction.
    """

    def __init__(
        self,
        seed: int,
        n_max: int,
        template: PopulationTemplate | None = None,
        *,
        class_datasets: dict[str, str] | None = None,
        class_params: dict[str, dict] | None = None,
        default_dataset: str = "gsm8k",
    ) -> None:
        if n_max <= 0:
            raise ValueError("n_max must be positive")
        self.seed = int(seed)
        self.n_max = int(n_max)
        self.template = template or PopulationTemplate.homogeneous()
        self.class_datasets = dict(class_datasets or {})
        self.class_params = dict(class_params or {})
        self.default_dataset = default_dataset
        self._specs: list[ClientSpec] = self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> list[ClientSpec]:
        rng = random.Random(self.seed)
        class_seq = self.template.class_sequence()
        block = len(class_seq)
        # round n_max up to a whole number of template blocks so every
        # scale(m) that fits is a clean prefix
        n_blocks = (self.n_max + block - 1) // block
        total = n_blocks * block

        # per-class running prompt-shard index, so client k of class c gets
        # shard k (not a random one) -> deterministic and nested
        per_class_index: dict[str, int] = {}
        specs: list[ClientSpec] = []
        for stable_id in range(total):
            cls = class_seq[stable_id % block]
            idx = per_class_index.get(cls, 0)
            per_class_index[cls] = idx + 1
            params = self.class_params.get(cls, {})
            dataset = self.class_datasets.get(cls, params.get("dataset", self.default_dataset))
            specs.append(
                ClientSpec(
                    stable_id=stable_id,
                    client_seed=rng.randrange(2**31),
                    class_name=cls,
                    dataset_name=str(dataset),
                    dataset_index=idx,
                    draft_slowdown=float(params.get("draft_slowdown", 1.0)),
                    rtt_ms=float(params.get("rtt_ms", 40.0)),
                    uplink_mbps=float(params.get("uplink_mbps", 20.0)),
                    downlink_mbps=float(params.get("downlink_mbps", 40.0)),
                    draft_fixed_latency_ms=float(params.get("draft_fixed_latency_ms", 10.0)),
                    draft_tokens_per_second=float(params.get("draft_tokens_per_second", 40.0)),
                    packet_loss=float(params.get("packet_loss", 0.0)),
                    extra=dict(params.get("extra", {})),
                )
            )
        return specs

    # ------------------------------------------------------------------ reads
    def __len__(self) -> int:
        return len(self._specs)

    @property
    def block_size(self) -> int:
        return self.template.block_size

    def prefix(self, n: int) -> list[ClientSpec]:
        """First `n` specs (homogeneous / free-N search)."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n > len(self._specs):
            raise ValueError(f"catalog holds {len(self._specs)} specs, asked for {n}")
        return self._specs[:n]

    def scale(self, m: int) -> list[ClientSpec]:
        """First `m` template blocks, i.e. `m * block_size` specs, keeping
        class proportions exactly fixed (Section 4.2)."""
        if m < 0:
            raise ValueError("m must be non-negative")
        n = m * self.block_size
        return self.prefix(n)

    def max_scale(self) -> int:
        return len(self._specs) // self.block_size

    # ------------------------------------------------------------ provenance
    def fingerprint(self, n: int | None = None) -> str:
        """Stable hash of the spec slice actually used by a run. Two runs
        with the same fingerprint provably used nested (prefix-equal)
        populations. Log this on every capacity run (Section 4.3)."""
        specs = self._specs if n is None else self.prefix(n)
        payload = json.dumps(
            [asdict(s) for s in specs], sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def is_nested_with(self, other: "PopulationCatalog", n_small: int) -> bool:
        """True iff `self.prefix(n_small)` is byte-identical to
        `other.prefix(n_small)` — the check the tests assert."""
        return self.fingerprint(n_small) == other.fingerprint(n_small)
