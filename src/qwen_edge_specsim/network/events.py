from dataclasses import dataclass


@dataclass(frozen=True)
class NetworkEventTimestamps:
    draft_start_ms: float
    draft_finish_ms: float
    uplink_start_ms: float
    server_arrival_ms: float
    batch_ready_ms: float
    batch_start_ms: float
    verify_finish_ms: float
    client_receive_ms: float


__all__ = ["NetworkEventTimestamps"]
