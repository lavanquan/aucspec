from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_specsim.device_config import infer_draft_devices, visible_cuda_device_count


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_config(config_path: Path, cfg: dict) -> None:
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def _infer_visible_count(args: argparse.Namespace) -> int:
    if args.visible_count is not None:
        return max(0, int(args.visible_count))
    env_count = visible_cuda_device_count()
    if env_count > 0:
        return env_count
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Unable to infer CUDA device count: CUDA_VISIBLE_DEVICES is unset and "
            "PyTorch is unavailable. Pass --visible-count explicitly."
        ) from exc
    return int(torch.cuda.device_count())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a runtime config with models.draft_devices inferred from "
            "CUDA_VISIBLE_DEVICES or torch.cuda.device_count()."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--output-config",
        default="configs/runtime_devices.yaml",
        help="Path for the generated YAML config.",
    )
    parser.add_argument(
        "--visible-count",
        type=int,
        help="Override detected visible CUDA device count.",
    )
    parser.add_argument(
        "--allow-target-sharing",
        action="store_true",
        help="Allow draft_devices to include cuda:0 instead of reserving it for the target.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print inferred draft_devices without writing a config.",
    )
    args = parser.parse_args()

    visible_count = _infer_visible_count(args)
    draft_devices = infer_draft_devices(
        total_visible_cuda_devices=visible_count,
        reserve_first_for_target=not args.allow_target_sharing,
    )

    if args.print_only:
        print(",".join(draft_devices))
        return

    config_path = Path(args.config)
    output_path = Path(args.output_config)
    cfg = _load_config(config_path)
    cfg.setdefault("models", {})["draft_devices"] = draft_devices
    _write_config(output_path, cfg)

    print(f"Visible CUDA devices: {visible_count}")
    print(f"Configured draft_devices: {draft_devices}")
    print(f"Wrote runtime config to {output_path}")


if __name__ == "__main__":
    main()
