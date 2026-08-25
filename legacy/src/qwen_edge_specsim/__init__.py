"""Compatibility package for the staged qwen_edge_specsim project layout."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edge_specsim.simulator import EdgeSpecSimulator

__all__ = ["EdgeSpecSimulator"]


def __getattr__(name: str):
    if name == "EdgeSpecSimulator":
        from edge_specsim.simulator import EdgeSpecSimulator

        return EdgeSpecSimulator
    raise AttributeError(name)
