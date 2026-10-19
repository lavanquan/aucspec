"""CPU-only tests: corrected-runtime guards, launcher content, legacy-launcher refusal."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from edge_specsim import runtime_guard as rg  # noqa: E402

SHIM = "/software/x/shim"
MAN = {"shim_dir": SHIM}
GOOD_ENV = {"VLLM_USE_TRITON_FLASH_ATTN": "1", "PYTHONPATH": f"{SHIM}:/other"}


def test_environment_good_and_each_violation():
    assert rg.check_environment(GOOD_ENV, MAN) == []
    assert rg.check_environment({**GOOD_ENV, "VLLM_USE_TRITON_FLASH_ATTN": "0"}, MAN)
    e = dict(GOOD_ENV)
    del e["VLLM_USE_TRITON_FLASH_ATTN"]
    assert rg.check_environment(e, MAN)
    assert rg.check_environment({**GOOD_ENV, "PYTHONPATH": "/other:" + SHIM}, MAN)   # shim must be FIRST
    assert rg.check_environment({k: v for k, v in GOOD_ENV.items() if k != "PYTHONPATH"}, MAN)
    assert rg.check_environment({**GOOD_ENV, "VLLM_ATTENTION_BACKEND": "TORCH_SDPA"}, MAN)
    assert rg.check_environment({**GOOD_ENV, "VLLM_USE_V1": "1"}, MAN)
    assert rg.check_environment({**GOOD_ENV, "ALLOW_DEFECTIVE_LEGACY_RUNTIME": "1"}, MAN)


def _fake_llm(n_triton=28, n_naive=0, engine_cls="LLMEngine", engine_mod="vllm.engine.llm_engine", prefix=False):
    impls = [SimpleNamespace(use_naive_attn=False, use_triton_flash_attn=True, triton_attn_func=lambda *a, **k: 1) for _ in range(n_triton)]
    impls += [SimpleNamespace(use_naive_attn=True, use_triton_flash_attn=False, sdpa_attn_func=lambda *a: 1, triton_attn_func=None) for _ in range(n_naive)]
    mods = [SimpleNamespace(impl=i) for i in impls]
    eng = type(engine_cls, (), {})()
    eng.__class__.__module__ = engine_mod
    eng.model_executor = SimpleNamespace(driver_worker=SimpleNamespace(model_runner=SimpleNamespace(model=SimpleNamespace(modules=lambda: mods))))
    eng.model_config = SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=28), dtype="bf16", enforce_eager=False)
    eng.cache_config = SimpleNamespace(enable_prefix_caching=prefix)
    eng.scheduler_config = SimpleNamespace(chunked_prefill_enabled=False, max_num_batched_tokens=4096, max_num_seqs=256)
    return SimpleNamespace(llm_engine=eng)


def _target(max_lp=1, tie=rg.TIE_BREAK_V1):
    return SimpleNamespace(effective_max_logprobs=max_lp, TIE_BREAK_RULE=tie, greedy_only=True)


def test_live_engine_accepts_the_corrected_runtime_and_counts_calls():
    llm = _fake_llm()
    ev = rg.verify_live_engine(llm, _target(), {})
    assert ev["layers_triton_flash_attn"] == 28 and ev["layers_naive_sdpa"] == 0
    impls = rg.attention_impls(llm)
    for _ in range(3):
        impls[0].triton_attn_func()
    assert ev["_counter"].snapshot()["triton_fa_calls"] == 3
    assert "_counter" not in rg.serialisable({"live_engine": ev})["live_engine"]


@pytest.mark.parametrize("kw,tgt", [
    ({"n_triton": 27, "n_naive": 1}, {}),            # one naive layer
    ({"n_triton": 0, "n_naive": 28}, {}),            # the legacy defective runtime
    ({"n_triton": 20}, {}),                          # missing layers
    ({"engine_mod": "vllm.v1.engine.llm_engine"}, {}),  # V1 engine
    ({"prefix": True}, {}),
    ({}, {"max_lp": 2}),                             # diagnostic top-2 engine is not canonical
    ({}, {"tie": "smallest_token_id"}),              # V2 tie-break
])
def test_live_engine_refuses_every_fallback_or_noncanonical_setting(kw, tgt):
    with pytest.raises(rg.RuntimeGuardError):
        rg.verify_live_engine(_fake_llm(**kw), _target(**tgt), {})


def test_canonical_guard_requires_runtime_section_and_env():
    import canonical_guard as g

    cfg = {"verification_batching": {"service_time_model": "batch_wall", "scheduler": "capacity_knapsack", "budget_mode": "fixed",
                                     "verify_token_budget": 100, "max_batch_size": 32, "batch_wait_ms": 150.0},
           "controller": {"verifier_price_mode": "zero"}}
    with pytest.raises(g.GuardError):
        g.validate_canonical_config(cfg)             # legacy-style config without runtime: refused
    cfg["runtime"] = {"identity": rg.IDENTITY, "require_triton_fa": True}
    g.validate_canonical_config(cfg)
    with pytest.raises(g.GuardError):
        g.validate_canonical_config({**cfg, "runtime": {"identity": rg.IDENTITY, "require_triton_fa": False}})
    man = REPO / "runtime" / "tritonfa33" / "MANIFEST.json"
    if man.exists():
        shim = rg.load_manifest(man)["shim_dir"]
        g.assert_runtime_env({"VLLM_USE_TRITON_FLASH_ATTN": "1", "PYTHONPATH": shim})
        with pytest.raises(g.GuardError):
            g.assert_runtime_env({"VLLM_USE_TRITON_FLASH_ATTN": "0", "PYTHONPATH": shim})


def test_repository_canonical_launcher_is_corrected():
    s = " ".join(l for l in (REPO / "scripts" / "run_canonical_job.sh").read_text().splitlines() if not l.lstrip().startswith("#"))
    assert "export VLLM_USE_TRITON_FLASH_ATTN=1" in s
    assert "VLLM_USE_TRITON_FLASH_ATTN=0" not in s
    assert "pip install" not in s and "pip3 install" not in s
    assert "preflight_runtime.py" in s and "tritonfa33/shim" in s
    assert "rc=$?" in s and s.rstrip().endswith("exit $rc")           # real exit code propagated to Slurm


def test_launchers_selecting_the_defective_runtime_refuse_to_run(tmp_path):
    legacy = [p for p in (REPO / "scripts").glob("*.sh") if "VLLM_USE_TRITON_FLASH_ATTN=0" in p.read_text()]
    assert legacy, "expected labelled legacy launchers"
    for p in legacy:
        r = subprocess.run(["bash", str(p), "1", "1", "1", "1"], capture_output=True, text=True, cwd=tmp_path,
                           env={"PATH": "/usr/bin:/bin"})
        assert r.returncode == 64 and "REFUSING" in r.stderr, p.name
        assert "DEFECTIVE LEGACY RUNTIME GUARD" in p.read_text()
    new = (REPO / "scripts" / "run_canonical_job.sh").read_text()
    assert "DEFECTIVE LEGACY RUNTIME GUARD" not in new and "ALLOW_DEFECTIVE_LEGACY_RUNTIME=1" not in new


def test_new_canonical_configs_declare_the_runtime():
    import yaml

    for name in ("canonical_batchwall_tritonfa33_v1.yaml", "canonical_batchwall_tritonfa33_smoke_wiring.yaml"):
        p = REPO / "configs" / name
        if not p.exists():
            pytest.skip("config not generated yet")
        cfg = yaml.safe_load(p.read_text())
        assert cfg["runtime"]["identity"] == rg.IDENTITY and cfg["runtime"]["require_triton_fa"] is True
        assert cfg["verification_batching"]["service_time_model"] == "batch_wall"
        assert cfg["controller"]["verifier_price_mode"] == "zero"
        assert cfg["models"].get("target_enable_prefix_caching") is False


def test_in_process_check_ignores_only_the_repo_src_prefix():
    ok = {**GOOD_ENV, "PYTHONPATH": "/repo/legacy/src:" + SHIM + ":/other"}
    assert rg.check_environment(ok, MAN, strict_first=False) == []
    assert rg.check_environment({**GOOD_ENV, "PYTHONPATH": "/somewhere/else:" + SHIM}, MAN, strict_first=False)      # arbitrary prefix refused
    assert rg.check_environment({**GOOD_ENV, "PYTHONPATH": "/repo/legacy/src:/other:" + SHIM}, MAN, strict_first=False)   # shim not first
    assert rg.check_environment({**GOOD_ENV, "PYTHONPATH": SHIM + ":/repo/legacy/src"}, MAN, strict_first=False) == []


def test_guard_does_not_depend_on_sys_path_order_after_engine_start():
    src = (REPO / "legacy" / "src" / "edge_specsim" / "runtime_guard.py").read_text()
    assert "lists the user-site before" not in src and "sys_path_head" in src


def test_in_process_environment_check_tolerates_target_client_prefixing_pythonpath():
    mutated = {**GOOD_ENV, "PYTHONPATH": "/repo/legacy/src:" + GOOD_ENV["PYTHONPATH"]}   # what importing target_client does
    assert rg.check_environment(mutated, MAN)                                  # launch-time (strict) rule refuses it
    assert rg.check_environment(mutated, MAN, strict_first=False) == []         # in-process rule accepts, shim still present
    assert rg.check_environment({**mutated, "PYTHONPATH": "/repo/legacy/src"}, MAN, strict_first=False)   # shim missing
