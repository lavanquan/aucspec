from __future__ import annotations

from edge_specsim.device_config import infer_draft_devices, visible_cuda_device_count


def test_visible_cuda_device_count_parses_csv() -> None:
    assert visible_cuda_device_count("0,2,4,7") == 4
    assert visible_cuda_device_count("0, 1, 2") == 3
    assert visible_cuda_device_count("") == 0


def test_infer_draft_devices_reserves_first_visible_gpu_for_target() -> None:
    assert infer_draft_devices(4, reserve_first_for_target=True) == [
        "cuda:1",
        "cuda:2",
        "cuda:3",
    ]


def test_infer_draft_devices_can_include_cuda_zero_when_sharing_target() -> None:
    assert infer_draft_devices(4, reserve_first_for_target=False) == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
    ]


def test_infer_draft_devices_rejects_single_gpu_when_target_is_reserved() -> None:
    try:
        infer_draft_devices(1, reserve_first_for_target=True)
    except ValueError as exc:
        assert "Only one visible CUDA device was detected" in str(exc)
    else:
        raise AssertionError("expected ValueError for single reserved target GPU")
