"""Job-side preflight for the corrected runtime (runs on the compute node, before any simulator work).

Aborts (exit 65) unless: the environment selects Triton FlashAttention, the Triton shim is first on PYTHONPATH,
the imported torch/triton are exactly the manifest versions at the manifest paths, the key file checksums match,
and the frozen dependencies are importable at their frozen versions. It installs nothing.
"""

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "legacy" / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO / "runtime" / "tritonfa33" / "MANIFEST.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from edge_specsim import runtime_guard as rg

    manifest = rg.load_manifest(a.manifest)
    problems = rg.check_environment(dict(os.environ), manifest)
    ev = {}
    try:
        ev = rg.check_imports(manifest)
    except Exception as exc:  # noqa: BLE001
        problems.append(repr(exc))
    for rel, want in manifest["sha256"].items():
        base = {"triton": manifest["triton_realpath_prefix"], "vllm": manifest["vllm_dir"],
                "torch": manifest["torch_dir_prefix"]}[rel.split("/")[0]]
        p = Path(base) / rel.split("/", 1)[1]
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        if got != want:
            problems.append(f"checksum mismatch for {rel}: {got}")
    for dep, want in manifest["frozen_dependencies"].items():
        name = {"pyyaml": "PyYAML"}.get(dep, dep)
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            problems.append(f"dependency {dep} missing (no runtime installs are performed)")
            continue
        if have != want:
            problems.append(f"dependency {dep} {have} != frozen {want}")
    rec = {"ok": not problems, "problems": problems, "imports": ev, "manifest": a.manifest}
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    return 0 if not problems else 65


if __name__ == "__main__":
    sys.exit(main())
