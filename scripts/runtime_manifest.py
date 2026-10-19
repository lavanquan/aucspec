"""Generate runtime/tritonfa33/MANIFEST.json from the live (module) environment.

Run inside the job environment WITH the Triton shim on PYTHONPATH (see runtime/tritonfa33/README.md).
Records versions, import paths and sha256 of the files that decide which attention kernel runs. Read-only.
"""

import hashlib
import json
import os
import sys
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    shim = sys.argv[1]
    out = Path(sys.argv[2])
    import importlib.metadata as md

    import pandas
    import torch
    import triton
    import importlib.util as iu
    import yaml

    link = Path(shim) / "triton"
    tri = os.path.realpath(triton.__file__)
    tdir = os.path.dirname(tri)
    vdir = os.path.realpath(list(iu.find_spec("vllm").submodule_search_locations)[0])
    m = {
        "identity": "V1-batchwall-tritonfa33",
        "shim_dir": shim,
        "shim_symlink": str(link), "shim_symlink_target": os.readlink(link) if link.is_symlink() else None,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__, "torch_dir_prefix": os.path.dirname(os.path.realpath(torch.__file__)),
        "triton_version": triton.__version__, "triton_realpath_prefix": tdir,
        "hip_version": getattr(torch.version, "hip", None),
        "vllm_version": md.version("vllm"), "vllm_dir": vdir,
        "frozen_dependencies": {"pandas": pandas.__version__, "pyyaml": yaml.__version__,
                                "transformers": md.version("transformers"), "numpy": md.version("numpy")},
        "dependency_files": {"pandas": os.path.dirname(os.path.realpath(pandas.__file__)),
                             "yaml": os.path.dirname(os.path.realpath(yaml.__file__))},
        "sha256": {
            "triton/__init__.py": sha(tri),
            "triton/compiler/compiler.py": sha(os.path.join(tdir, "compiler", "compiler.py")),
            "vllm/attention/backends/rocm_flash_attn.py": sha(os.path.join(vdir, "attention", "backends", "rocm_flash_attn.py")),
            "vllm/attention/ops/triton_flash_attention.py": sha(os.path.join(vdir, "attention", "ops", "triton_flash_attention.py")),
            "torch/version.py": sha(os.path.join(os.path.dirname(os.path.realpath(torch.__file__)), "version.py")),
        },
        "expected_attention_layers_qwen2_5_7b": 28,
    }
    out.write_text(json.dumps(m, indent=1, sort_keys=True))
    print(json.dumps({k: m[k] for k in ("torch_version", "triton_version", "vllm_version", "shim_symlink_target", "frozen_dependencies")}))


if __name__ == "__main__":
    main()
