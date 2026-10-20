"""Frozen prompt catalog for nested populations (C1.2).

The nested population gives client i the block samples[i*qpc:(i+1)*qpc] of ONE ordered catalog, so a smaller N is an exact prefix of a
larger one.  This module makes that assignment auditable: a catalog hash over (sample_id, prompt, max_new_tokens) of every assigned slot,
per-client block hashes, duplicate statistics and an explicit refusal when the catalog is shorter than n_max * qpc (no silent truncation).
"""
from __future__ import annotations

import hashlib
import json


class CatalogError(RuntimeError):
    pass


def _h(items) -> str:
    return hashlib.sha256(json.dumps(items, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def catalog_record(samples, n_max: int, qpc: int) -> dict:
    need = int(n_max) * int(qpc)
    if len(samples) < need:
        raise CatalogError(f"catalog has {len(samples)} samples but n_max*qpc = {need} are required; refusing to truncate or reuse")
    used = list(samples[:need])
    slots = [[s.sample_id, s.prompt, int(s.max_new_tokens)] for s in used]
    blocks = [_h(slots[i * qpc:(i + 1) * qpc]) for i in range(n_max)]
    ids = [s.sample_id for s in used]
    prompts = [s.prompt for s in used]
    return {
        "n_max": int(n_max), "questions_per_client": int(qpc), "n_slots": need, "catalog_available": len(samples),
        "catalog_hash": _h(slots),
        "prefix_hashes": {str(n): _h(slots[: n * qpc]) for n in sorted({1, 2, 4, 8, 12, 16, 24, 32, n_max} & set(range(1, n_max + 1)))},
        "client_block_hashes": blocks,
        "duplicates": {"distinct_sample_ids": len(set(ids)), "distinct_prompts": len(set(prompts)),
                       "duplicate_slots": need - len(set(prompts)), "dataset_names": sorted({s.dataset_name for s in used}),
                       "id_prefixes": sorted({i.rsplit("-", 1)[0] for i in ids})},
    }


def assert_catalog(samples, n_max: int, qpc: int, expected_hash: str | None) -> dict:
    rec = catalog_record(samples, n_max, qpc)
    if expected_hash is not None and rec["catalog_hash"] != expected_hash:
        raise CatalogError(f"catalog hash {rec['catalog_hash']} != frozen {expected_hash}: the prompt assignment changed")
    return rec
