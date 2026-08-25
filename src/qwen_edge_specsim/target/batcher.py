from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edge_specsim.verification_batcher import VerificationBatcher
    from edge_specsim.verification_queue import (
        VerificationBatchMetadata,
        VerificationQueue,
        VerificationRequest,
    )

__all__ = [
    "VerificationBatcher",
    "VerificationBatchMetadata",
    "VerificationQueue",
    "VerificationRequest",
]


def __getattr__(name: str):
    if name == "VerificationBatcher":
        from edge_specsim.verification_batcher import VerificationBatcher

        return VerificationBatcher
    if name in {"VerificationBatchMetadata", "VerificationQueue", "VerificationRequest"}:
        from edge_specsim.verification_queue import (
            VerificationBatchMetadata,
            VerificationQueue,
            VerificationRequest,
        )

        return {
            "VerificationBatchMetadata": VerificationBatchMetadata,
            "VerificationQueue": VerificationQueue,
            "VerificationRequest": VerificationRequest,
        }[name]
    raise AttributeError(name)
