from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RooflineKneeEntry:
    batch_size: int
    context_length: int
    knee_verifier_tokens: int
    theta_0_ms: float = 0.0
    theta_m_ms: float = 0.0
    theta_f_ms_per_token: float = 0.0


@dataclass(frozen=True)
class RooflineBudgetSignal:
    budget_tokens: int
    regime: str
    theta_0_ms: float = 0.0
    theta_m_ms: float = 0.0
    theta_f_ms_per_token: float = 0.0
    knee_verifier_tokens: int = 0

    def service_time_ms(self, total_verifier_tokens: int) -> float:
        verifier_tokens = max(0.0, float(total_verifier_tokens))
        return float(self.theta_0_ms) + max(
            float(self.theta_m_ms),
            float(self.theta_f_ms_per_token) * verifier_tokens,
        )


class RooflineBudgetModel:
    def __init__(self, entries: list[RooflineKneeEntry]) -> None:
        if not entries:
            raise ValueError("RooflineBudgetModel requires at least one knee entry")
        self.entries = list(entries)

    @classmethod
    def from_summary_json(cls, path: str | Path) -> "RooflineBudgetModel":
        summary_path = Path(path)
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        knee_rows = payload.get("roofline_knees", [])
        entries = [
            RooflineKneeEntry(
                batch_size=int(row["batch_size"]),
                context_length=int(row["context_length"]),
                knee_verifier_tokens=max(1, int(row["knee_verifier_tokens"])),
                theta_0_ms=float(row.get("theta_0_ms", payload.get("theta_0_ms", 0.0))),
                theta_m_ms=float(row.get("theta_m_ms", 0.0)),
                theta_f_ms_per_token=float(row.get("theta_f_ms_per_token", 0.0)),
            )
            for row in knee_rows
        ]
        if not entries:
            raise ValueError(
                f"No roofline_knees entries found in summary file {summary_path}"
            )
        return cls(entries)

    def _lookup_entry(
        self,
        batch_size: int,
        context_length: int,
    ) -> RooflineKneeEntry:
        requested_batch_size = max(1, int(batch_size))
        requested_context_length = max(1, int(context_length))
        return min(
            self.entries,
            key=lambda entry: (
                abs(entry.batch_size - requested_batch_size),
                abs(entry.context_length - requested_context_length),
                entry.batch_size,
                entry.context_length,
            ),
        )

    def budget_for(
        self,
        batch_size: int,
        context_length: int,
    ) -> int:
        return self.signal_for(
            batch_size=batch_size,
            context_length=context_length,
            queued_verifier_tokens=0,
        ).budget_tokens

    def signal_for(
        self,
        batch_size: int,
        context_length: int,
        queued_verifier_tokens: int,
    ) -> RooflineBudgetSignal:
        best_entry = self._lookup_entry(batch_size, context_length)
        knee_tokens = max(1, int(best_entry.knee_verifier_tokens))
        queued_tokens = max(0, int(queued_verifier_tokens))
        if queued_tokens < knee_tokens:
            regime = "memory_bound"
        elif queued_tokens == knee_tokens:
            regime = "roofline_knee"
        else:
            regime = "compute_bound"
        return RooflineBudgetSignal(
            budget_tokens=knee_tokens,
            regime=regime,
            theta_0_ms=float(best_entry.theta_0_ms),
            theta_m_ms=float(best_entry.theta_m_ms),
            theta_f_ms_per_token=float(best_entry.theta_f_ms_per_token),
            knee_verifier_tokens=knee_tokens,
        )

    def service_time_ms(
        self,
        batch_size: int,
        context_length: int,
        total_verifier_tokens: int,
    ) -> float:
        signal = self.signal_for(
            batch_size=batch_size,
            context_length=context_length,
            queued_verifier_tokens=total_verifier_tokens,
        )
        return signal.service_time_ms(total_verifier_tokens)
