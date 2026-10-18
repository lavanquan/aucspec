"""CPU-only tests for the canonical launcher summary (synthetic CSV, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def launch():
    sys.path.insert(0, str(REPO / "scripts"))
    sys.modules.pop("canonical_launch", None)
    import canonical_launch

    yield canonical_launch
    sys.path.remove(str(REPO / "scripts"))


def test_summarize_csv_dedupes_batches(launch, tmp_path):
    # two batches (sizes 2 and 1); batch rows are repeated once per request
    rows = [
        dict(verification_batch_id=0, verification_batch_size=2, verification_batch_wall_ms=200.0,
             verification_batch_service_ms=200.0, verification_amortized_request_ms=100.0,
             verification_service_time_model="batch_wall", verification_batch_token_cost=8,
             verification_batch_token_budget=100, verification_batch_processed_tokens=50,
             verification_batch_context_tokens_sum=44, batch_start_ms=0.0, verify_finish_ms=200.0),
        dict(verification_batch_id=0, verification_batch_size=2, verification_batch_wall_ms=200.0,
             verification_batch_service_ms=200.0, verification_amortized_request_ms=100.0,
             verification_service_time_model="batch_wall", verification_batch_token_cost=8,
             verification_batch_token_budget=100, verification_batch_processed_tokens=50,
             verification_batch_context_tokens_sum=44, batch_start_ms=0.0, verify_finish_ms=200.0),
        dict(verification_batch_id=1, verification_batch_size=1, verification_batch_wall_ms=170.0,
             verification_batch_service_ms=170.0, verification_amortized_request_ms=170.0,
             verification_service_time_model="batch_wall", verification_batch_token_cost=4,
             verification_batch_token_budget=100, verification_batch_processed_tokens=30,
             verification_batch_context_tokens_sum=26, batch_start_ms=200.0, verify_finish_ms=370.0),
    ]
    path = tmp_path / "r.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    s = launch.summarize_csv(path)
    assert s["n_batches"] == 2
    assert s["service_time_model"] == ["batch_wall"]
    assert s["batch_size"]["max"] == 2
    assert s["wall_over_modeled_mean"] == pytest.approx(1.0)
    assert s["verifier_utilisation"] == pytest.approx(1.0)  # busy 370 ms over a 370 ms span
    assert s["budget_binding_fraction"] == 0.0
