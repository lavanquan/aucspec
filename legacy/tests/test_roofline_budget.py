from __future__ import annotations

from edge_specsim.roofline_budget import RooflineBudgetModel, RooflineKneeEntry


def test_roofline_budget_uses_nearest_profiled_entry() -> None:
    model = RooflineBudgetModel(
        [
            RooflineKneeEntry(batch_size=4, context_length=512, knee_verifier_tokens=20),
            RooflineKneeEntry(batch_size=8, context_length=2048, knee_verifier_tokens=36),
        ]
    )

    assert model.budget_for(batch_size=3, context_length=600) == 20
    assert model.budget_for(batch_size=7, context_length=1800) == 36


def test_roofline_budget_signal_reports_regime() -> None:
    model = RooflineBudgetModel(
        [
            RooflineKneeEntry(batch_size=4, context_length=512, knee_verifier_tokens=20),
        ]
    )

    assert model.signal_for(4, 512, queued_verifier_tokens=10).regime == "memory_bound"
    assert model.signal_for(4, 512, queued_verifier_tokens=20).regime == "roofline_knee"
    assert model.signal_for(4, 512, queued_verifier_tokens=30).regime == "compute_bound"


def test_roofline_service_time_matches_paper_formulation() -> None:
    model = RooflineBudgetModel(
        [
            RooflineKneeEntry(
                batch_size=4,
                context_length=512,
                knee_verifier_tokens=20,
                theta_0_ms=5.0,
                theta_m_ms=40.0,
                theta_f_ms_per_token=2.0,
            ),
        ]
    )

    assert model.service_time_ms(4, 512, total_verifier_tokens=10) == 45.0
    assert model.service_time_ms(4, 512, total_verifier_tokens=30) == 65.0
