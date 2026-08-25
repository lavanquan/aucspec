from __future__ import annotations

import os


def visible_cuda_device_count(cuda_visible_devices: str | None = None) -> int:
    raw = cuda_visible_devices
    if raw is None:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return 0
    entries = [part.strip() for part in str(raw).split(",")]
    visible = [entry for entry in entries if entry]
    return len(visible)


def infer_draft_devices(
    total_visible_cuda_devices: int,
    reserve_first_for_target: bool = True,
) -> list[str]:
    visible_count = max(0, int(total_visible_cuda_devices))
    start_index = 1 if reserve_first_for_target else 0
    devices = [f"cuda:{index}" for index in range(start_index, visible_count)]
    if devices:
        return devices
    if visible_count <= 0:
        raise ValueError("No visible CUDA devices were detected")
    if reserve_first_for_target:
        raise ValueError(
            "Only one visible CUDA device was detected. Reserve-free draft placement "
            "would leave no draft GPU. Re-run with --allow-target-sharing."
        )
    raise ValueError("No draft CUDA devices could be inferred")
