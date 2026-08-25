from .cache import confirmed_prefix_hash

__all__ = [
    "VLLMCandidateVerifier",
    "VerificationBatcher",
    "VerificationBatchMetadata",
    "VerificationQueue",
    "VerificationRequest",
    "confirmed_prefix_hash",
]


def __getattr__(name: str):
    if name == "VLLMCandidateVerifier":
        from .verifier import VLLMCandidateVerifier

        return VLLMCandidateVerifier
    if name in {
        "VerificationBatcher",
        "VerificationBatchMetadata",
        "VerificationQueue",
        "VerificationRequest",
    }:
        from .batcher import (
            VerificationBatchMetadata,
            VerificationBatcher,
            VerificationQueue,
            VerificationRequest,
        )

        return {
            "VerificationBatcher": VerificationBatcher,
            "VerificationBatchMetadata": VerificationBatchMetadata,
            "VerificationQueue": VerificationQueue,
            "VerificationRequest": VerificationRequest,
        }[name]
    raise AttributeError(name)
