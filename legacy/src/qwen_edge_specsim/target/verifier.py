from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edge_specsim.target_client import VLLMCandidateVerifier

__all__ = ["VLLMCandidateVerifier"]


def __getattr__(name: str):
    if name == "VLLMCandidateVerifier":
        from edge_specsim.target_client import VLLMCandidateVerifier

        return VLLMCandidateVerifier
    raise AttributeError(name)
