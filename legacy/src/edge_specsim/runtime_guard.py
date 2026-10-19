"""Runtime identity and fallback guards for the corrected ROCm verifier runtime "V1-batchwall-tritonfa33".

Behaviour-neutral: nothing here changes decoding, sampling, batching or the controller. It only
  * checks (and records) which torch/Triton/vLLM files were really imported in the process that runs inference,
  * checks that every target attention layer runs Triton FlashAttention (zero naive-SDPA layers),
  * counts Triton-FA invocations so a run can prove the kernel actually executed.
Any violation raises RuntimeGuardError: there is no silent fallback to the known-defective naive-SDPA path.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

IDENTITY = "V1-batchwall-tritonfa33"
TIE_BREAK_V1 = "first_max_in_vllm_logprob_dict_order"


class RuntimeGuardError(RuntimeError):
    pass


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_environment(env: dict[str, str], manifest: dict[str, Any], strict_first: bool = True) -> list[str]:
    """Pure check of the process environment against the manifest; returns violations."""
    bad = []
    if env.get("VLLM_USE_TRITON_FLASH_ATTN") != "1":
        bad.append(f"VLLM_USE_TRITON_FLASH_ATTN must be '1' (got {env.get('VLLM_USE_TRITON_FLASH_ATTN')!r}); '0' selects the defective naive SDPA")
    if env.get("VLLM_ATTENTION_BACKEND"):
        bad.append("VLLM_ATTENTION_BACKEND must be unset (other values silently fall back on ROCm)")
    if env.get("VLLM_USE_V1", "0") not in ("0", ""):
        bad.append("VLLM_USE_V1 must not request the V1 engine (qualified runtime is the V0 engine)")
    shim = manifest["shim_dir"]
    entries = env.get("PYTHONPATH", "").split(os.pathsep)
    if strict_first:
        # launch-time rule. (Importing edge_specsim.target_client later prepends legacy/src to the env var, which is why
        # the in-process check below uses strict_first=False and relies on sys.path order and the real triton path.)
        if entries[0] != shim:
            bad.append(f"PYTHONPATH must start with the Triton shim {shim!r} (starts with {entries[0]!r})")
    elif shim not in entries:
        bad.append(f"the Triton shim {shim!r} is not on PYTHONPATH")
    if env.get("ALLOW_DEFECTIVE_LEGACY_RUNTIME") == "1":
        bad.append("ALLOW_DEFECTIVE_LEGACY_RUNTIME=1 is set: refusing to run a canonical job on the legacy runtime")
    return bad


def check_imports(manifest: dict[str, Any]) -> dict[str, Any]:
    """Import torch/triton in THIS process and compare with the manifest. Returns the evidence record."""
    import torch  # noqa: WPS433
    import triton  # noqa: WPS433

    ev = {
        "python_executable": sys.executable,
        "torch_version": torch.__version__, "torch_file": os.path.realpath(torch.__file__),
        "triton_version": triton.__version__, "triton_file": os.path.realpath(triton.__file__),
        "triton_import_path": triton.__file__,
        "hip_version": getattr(torch.version, "hip", None),
    }
    bad = []
    if ev["torch_version"] != manifest["torch_version"]:
        bad.append(f"torch {ev['torch_version']} != {manifest['torch_version']}")
    if not ev["torch_file"].startswith(manifest["torch_dir_prefix"]):
        bad.append(f"torch imported from {ev['torch_file']}")
    if ev["triton_version"] != manifest["triton_version"]:
        bad.append(f"triton {ev['triton_version']} != {manifest['triton_version']}")
    if not ev["triton_file"].startswith(manifest["triton_realpath_prefix"]):
        bad.append(f"triton resolved to {ev['triton_file']}, expected under {manifest['triton_realpath_prefix']}")
    if "pythonuserbase" in ev["triton_file"]:
        bad.append("triton was imported from the user-site (3.8.0) package")
    shim = manifest["shim_dir"]
    if shim not in sys.path:
        bad.append("the Triton shim is not on sys.path of this process")
    else:
        user = [i for i, e in enumerate(sys.path) if "pythonuserbase" in e]
        if user and sys.path.index(shim) > min(user):
            bad.append("sys.path lists the user-site before the Triton shim")
    ev["shim_on_sys_path_index"] = sys.path.index(shim) if shim in sys.path else None
    if bad:
        raise RuntimeGuardError("; ".join(bad))
    return ev


def attention_impls(llm) -> list[Any]:
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    out = []
    for m in runner.model.modules():
        impl = getattr(m, "impl", None)
        if impl is not None and hasattr(impl, "use_naive_attn"):
            out.append(impl)
    return out


class FaCounter:
    """Counts Triton-FA invocations per process (wraps each layer's `triton_attn_func`)."""

    def __init__(self):
        self.calls = 0
        self.pids: set[int] = set()

    def install(self, impls):
        for impl in impls:
            orig = impl.triton_attn_func

            def counted(*a, _o=orig, _s=self, **k):
                _s.calls += 1
                _s.pids.add(os.getpid())
                return _o(*a, **k)

            impl.triton_attn_func = counted

    def snapshot(self):
        return {"triton_fa_calls": self.calls, "pids": sorted(self.pids), "pid_now": os.getpid()}


def verify_live_engine(llm, target, rt_cfg: dict[str, Any]) -> dict[str, Any]:
    """Prove from inside the inference process which attention implementation is live."""
    impls = attention_impls(llm)
    cfg = llm.llm_engine.model_config
    n_expected = int(cfg.hf_config.num_hidden_layers)
    ev = {
        "engine_class": f"{type(llm.llm_engine).__module__}.{type(llm.llm_engine).__name__}",
        "executor_class": type(llm.llm_engine.model_executor).__name__,
        "n_attention_layers": len(impls), "n_expected_layers": n_expected,
        "layers_triton_flash_attn": sum(bool(getattr(i, "use_triton_flash_attn", False)) for i in impls),
        "layers_naive_sdpa": sum(bool(i.use_naive_attn) for i in impls),
        "layers_with_naive_func_attribute": sum(hasattr(i, "sdpa_attn_func") for i in impls),
        "attention_impl_class": type(impls[0]).__name__ if impls else None,
        "dtype": str(cfg.dtype), "enforce_eager": bool(cfg.enforce_eager),
        "prefix_caching": bool(llm.llm_engine.cache_config.enable_prefix_caching),
        "chunked_prefill": bool(llm.llm_engine.scheduler_config.chunked_prefill_enabled),
        "max_num_batched_tokens": int(llm.llm_engine.scheduler_config.max_num_batched_tokens),
        "max_num_seqs": int(llm.llm_engine.scheduler_config.max_num_seqs),
        "effective_max_logprobs": getattr(target, "effective_max_logprobs", None),
        "tie_break_rule": getattr(target, "TIE_BREAK_RULE", None),
        "greedy_only": getattr(target, "greedy_only", None),
    }
    bad = []
    if not ev["engine_class"].startswith("vllm.engine.llm_engine."):
        bad.append(f"engine is {ev['engine_class']}, expected the V0 LLMEngine")
    if ev["n_attention_layers"] != n_expected or n_expected <= 0:
        bad.append(f"{ev['n_attention_layers']} attention layers found, expected {n_expected}")
    if ev["layers_triton_flash_attn"] != n_expected:
        bad.append(f"only {ev['layers_triton_flash_attn']}/{n_expected} layers use Triton FlashAttention")
    if ev["layers_naive_sdpa"] != 0 or ev["layers_with_naive_func_attribute"] != 0:
        bad.append(f"{ev['layers_naive_sdpa']} layers use naive SDPA (known defective fallback)")
    if ev["effective_max_logprobs"] != 1:
        bad.append(f"effective max_logprobs is {ev['effective_max_logprobs']}, canonical runtime requires 1")
    if ev["prefix_caching"]:
        bad.append("prefix caching must be off")
    if ev["tie_break_rule"] != TIE_BREAK_V1:
        bad.append(f"tie-break rule {ev['tie_break_rule']!r} is not the original V1 rule")
    if ev["greedy_only"] is not True:
        bad.append("verifier is not greedy-only")
    if bad:
        raise RuntimeGuardError("; ".join(bad))
    counter = FaCounter()
    counter.install(impls)
    ev["_counter"] = counter
    return ev


def check_semantics_config(full_cfg: dict[str, Any]) -> dict[str, Any]:
    """The frozen experimental semantics of the runtime identity (config level)."""
    vb, ctl = full_cfg.get("verification_batching") or {}, full_cfg.get("controller") or {}
    rec = {"service_time_model": vb.get("service_time_model"), "verifier_price_mode": ctl.get("verifier_price_mode"),
           "budget_mode": vb.get("budget_mode")}
    bad = []
    if rec["service_time_model"] != "batch_wall":
        bad.append(f"service_time_model must be batch_wall (got {rec['service_time_model']!r})")
    if rec["verifier_price_mode"] != "zero":
        bad.append(f"verifier_price_mode must be zero (got {rec['verifier_price_mode']!r})")
    if bad:
        raise RuntimeGuardError("; ".join(bad))
    return rec


def enforce_runtime(target, rt_cfg: dict[str, Any], repo_root: Path, full_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full check used by the simulator right after the target engine exists. Returns the identity record."""
    manifest = load_manifest(Path(repo_root) / rt_cfg.get("manifest", "runtime/tritonfa33/MANIFEST.json"))
    if rt_cfg.get("identity") != IDENTITY:
        raise RuntimeGuardError(f"runtime.identity must be {IDENTITY!r}")
    bad = check_environment(dict(os.environ), manifest, strict_first=False)
    if bad:
        raise RuntimeGuardError("; ".join(bad))
    imports = check_imports(manifest)
    live = verify_live_engine(target.llm, target, rt_cfg)
    semantics = check_semantics_config(full_cfg) if full_cfg is not None else None
    import vllm

    return {"identity": IDENTITY, "imports": imports, "live_engine": live, "semantics": semantics,
            "vllm_version": vllm.__version__, "vllm_file": os.path.realpath(vllm.__file__),
            "manifest_sha256": __import__("hashlib").sha256((Path(repo_root) / rt_cfg.get("manifest", "runtime/tritonfa33/MANIFEST.json")).read_bytes()).hexdigest(),
            "env": {k: os.environ.get(k) for k in ("VLLM_USE_TRITON_FLASH_ATTN", "PYTHONPATH", "VLLM_ATTENTION_BACKEND", "VLLM_USE_V1")}}


def serialisable(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop the live counter object but keep its current snapshot."""
    if record is None:
        return None
    rec = dict(record)
    live = dict(rec["live_engine"])
    counter = live.pop("_counter", None)
    if counter is not None:
        live.update(counter.snapshot())
    rec["live_engine"] = live
    return rec
