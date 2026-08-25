from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edge_specsim.simulator import EdgeSpecSimulator


@dataclass
class SimulationEventLoop:
    """Compatibility wrapper around the current monolithic simulator runtime."""

    simulator: EdgeSpecSimulator

    async def run(self) -> None:
        await self.simulator.run()


__all__ = ["SimulationEventLoop"]
