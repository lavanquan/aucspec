"""CPU-only regression tests: fail-closed orchestration, semantic (not only argmax) gating, frozen-input integrity."""

from __future__ import annotations

import copy
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import rocm_fi_lib as fl  # noqa: E402
import rocm_qual_lib as q  # noqa: E402
import rocm_qual_orchestrate as orch  # noqa: E402
import verify_frozen_inputs as vfi  # noqa: E402


# ----------------------------------------------------------------------------- synthetic data
def make_frozen(n_dev=64, n_ho=32):
    sets = {}
    qi = 0
    for name, lengths, n in (("dev", [108, 138, 168, 205], n_dev), ("heldout", [125, 190], n_ho)):
        per = n // len(lengths)
        ctxs = []
        for gi, L in enumerate(lengths):
            for _ in range(per):
                qi += 1
                ctx = [(qi * 7 + t) % 5000 + 1 for t in range(L)]
                prop = [(qi * 13 + j) % 5000 + 10 for j in range(4)]
                fp = [{"top_ids": [prop[j] if j < 4 else 777, 99999], "top_logprobs": [-0.1, -2.0], "gap": 1.9} for j in range(5)]
                ctxs.append({"q_index": qi, "group": gi, "L": L, "prompt_len": 20, "history_len": L - 20, "context": ctx,
                             "context_sha": fl.token_hash(ctx), "proposal": prop, "fp32": fp})
        sets[name] = {"lengths": lengths, "contexts": ctxs}
    return {"sets": sets}


def good_record(kind, S, i, g):
    f = S[i]["fp32"]
    prop = S[i]["proposal"]
    if kind == "true":
        arg = [f[j]["top_ids"][0] for j in range(g + 1)]
        return {"slot": 0, "group_pos": 0, "accepted": g, "committed": prop[:g] + [arg[g]], "correction": None, "bonus": arg[g],
                "target_greedy": arg, "reached_eos": False, "corrupt_pos": None}
    toks, c = q.corrupt_proposal(f, prop, g, i)
    arg = [f[j]["top_ids"][0] for j in range(c + 1)]
    return {"slot": 0, "group_pos": 0, "accepted": c, "committed": prop[:c] + [arg[c]], "correction": arg[c], "bonus": None,
            "target_greedy": arg, "reached_eos": False, "corrupt_pos": c}


def make_runs(S, level):
    runs = {}
    for p in q.build_plan(len(S), level):
        recs = {}
        for pos, i in enumerate(p["ctx"]):
            if p["kind"] == "decode":
                recs[str(i)] = {"slot": pos, "group_pos": pos % p["B"], "tokens": list(S[i]["proposal"])}
            else:
                r = good_record(p["kind"], S, i, p["gamma"])
                r["group_pos"] = pos % p["B"]
                recs[str(i)] = r
        runs[q.key(p)] = {"item": {k: v for k, v in p.items() if k != "ctx"}, "ctx_order": p["ctx"], "records": recs, "calls": []}
    return runs


def make_result(frozen, mutate=None):
    res = {"status": "ok", "calls": 398,
           "runtime_evidence_end": {"live_engine": {"n_attention_layers": 28, "layers_triton_flash_attn": 28, "layers_naive_sdpa": 0, "triton_fa_calls": 12345}}}
    for s in ("dev", "heldout"):
        runs = make_runs(frozen["sets"][s]["contexts"], s)
        res[s] = {"runs": runs, "pass": True}
    if mutate:
        mutate(res, frozen)
    return res


@pytest.fixture(scope="module")
def frozen():
    return make_frozen()


# ----------------------------------------------------------------------------- Gate B: semantic gate
def test_perfect_synthetic_run_passes(frozen):
    v = q.qualification_verdict(make_result(frozen), frozen)
    assert v["ok"], v


def _first(res, s, prefix):
    k = next(k for k in res[s]["runs"] if k.startswith(prefix))
    return res[s]["runs"][k]


@pytest.mark.parametrize("field,val", [
    ("committed", "wrong_stream"),       # argmax list intact, committed stream wrong
    ("correction", 424242),
    ("bonus", 424242),
    ("accepted", "off_by_one"),
    ("reached_eos", True),
])
def test_correct_argmax_but_wrong_semantic_field_fails_the_gate(frozen, field, val):
    def mutate(res, fz):
        run = _first(res, "dev", "true_B16_g4_ident")
        for rec in run["records"].values():
            if field == "committed":
                rec["committed"] = rec["committed"][:-1] + [rec["committed"][-1] + 1]
            elif field == "accepted":
                rec["accepted"] -= 1
            elif field == "bonus":
                rec["bonus"] = val
            elif field == "correction":
                rec["correction"] = val
            else:
                rec["reached_eos"] = val

    res = make_result(frozen, mutate)
    # the argmax/target_greedy lists are still perfectly correct in this run
    run = _first(res, "dev", "true_B16_g4_ident")
    assert all(r["target_greedy"][:4] == frozen["sets"]["dev"]["contexts"][int(i)]["proposal"] for i, r in run["records"].items())
    v = q.qualification_verdict(res, frozen)
    assert not v["ok"] and any("dev gate failed" in r for r in v["reasons"])


def test_wrong_correction_in_corrupted_proposals_fails(frozen):
    def mutate(res, fz):
        for rec in _first(res, "dev", "corrupt_B16_g4_ident")["records"].values():
            rec["correction"] += 1
            rec["committed"][-1] += 1

    assert not q.qualification_verdict(make_result(frozen, mutate), frozen)["ok"]


def test_wrong_rejection_position_fails(frozen):
    def mutate(res, fz):
        for rec in _first(res, "heldout", "corrupt_B4_g4_ident")["records"].values():
            rec["accepted"] = 4            # verifier accepted a corrupted token

    v = q.qualification_verdict(make_result(frozen, mutate), frozen)
    assert not v["ok"] and any("heldout gate failed" in r for r in v["reasons"])


def test_decode_chain_and_decode_prefill_disagreement_fail(frozen):
    def mutate(res, fz):
        for rec in _first(res, "dev", "decode_B16_g4_ident")["records"].values():
            rec["tokens"] = [t + 1 for t in rec["tokens"]]

    assert not q.qualification_verdict(make_result(frozen, mutate), frozen)["ok"]


def test_permutation_dependence_fails_but_only_when_margin_clear(frozen):
    def mutate(res, fz):
        run = _first(res, "dev", "true_B8_g4_perm1")
        for rec in run["records"].values():
            rec["target_greedy"] = rec["target_greedy"][:-1] + [rec["target_greedy"][-1] + 1]

    assert not q.qualification_verdict(make_result(frozen, mutate), frozen)["ok"]


def test_near_tie_mismatches_are_counted_raw_but_do_not_fail_clear_gate(frozen):
    fz = copy.deepcopy(frozen)
    S = fz["sets"]["dev"]["contexts"]
    bad = [0, 1]                                              # two contexts with a bf16-tie-sized margin at position 2
    for i in bad:
        S[i]["fp32"][2]["gap"] = 0.07
    res = make_result(fz)
    run = _first(res, "dev", "true_B16_g4_ident")
    for i in bad:                                             # verifier flipped at the near tie: stream deviates from fp32
        r = run["records"][str(i)]
        r["accepted"], r["committed"], r["target_greedy"] = 2, S[i]["proposal"][:2] + [31337], S[i]["proposal"][:2] + [31337]
    ev = q.evaluate_set(S, res["dev"]["runs"], "dev")
    assert ev["pass"], ev["failures"]
    raw = ev["checks"]["true_B16_g4_ident:raw"]["committed_tokens"]
    clear = ev["checks"]["true_B16_g4_ident:committed_tokens"]
    assert raw[0] == raw[1] - 2 and clear[0] == clear[1] and clear[1] == raw[1] - 2     # difficult cases stay visible in raw


# ----------------------------------------------------------------------------- Gate A: fail-closed orchestration
def args(tmp_path, frozen_file, sha, **kw):
    base = dict(config="cfg.yaml", exclude_frozen=None, out_dir=str(tmp_path / "out"), smoke_run_tag="smoke_tag", smoke_n=8,
                reuse_frozen=str(frozen_file), expect_frozen_sha256=sha, expected_sha="abc")
    base.update(kw)
    return Namespace(**base)


def runner_factory(frozen, result=None, rc_qual=0, rc_pre=0, rc_smoke=0, rc_checks=0, write_result=True):
    calls = []

    def runner(name, cmd, timeout, logfile):
        calls.append(name)
        if name == "preflight":
            return rc_pre
        if name == "qualification":
            if write_result:
                out = Path(cmd[cmd.index("--out") + 1])
                out.write_text(json.dumps(result if result is not None else make_result(frozen)))
            return rc_qual
        if name == "integration_smoke":
            return rc_smoke
        if name == "integration_checks":
            return rc_checks
        raise AssertionError(name)

    runner.calls = calls
    return runner


@pytest.fixture
def ffile(tmp_path, frozen):
    p = tmp_path / "frozen.json"
    p.write_text(json.dumps(frozen))
    return p, vfi.sha256_file(p)


def test_all_pass_exits_zero_and_runs_the_smoke(tmp_path, frozen, ffile):
    r = runner_factory(frozen)
    assert orch.run_pipeline(args(tmp_path, *ffile), runner=r) == 0
    assert r.calls == ["preflight", "qualification", "integration_smoke", "integration_checks"]


def test_preflight_failure_is_nonzero_and_nothing_else_runs(tmp_path, frozen, ffile):
    r = runner_factory(frozen, rc_pre=65)
    assert orch.run_pipeline(args(tmp_path, *ffile), runner=r) == 65 and r.calls == ["preflight"]


def test_frozen_input_integrity_failure_stops_before_any_gpu_step(tmp_path, frozen, ffile):
    r = runner_factory(frozen)
    assert orch.run_pipeline(args(tmp_path, ffile[0], "0" * 64), runner=r) == 3 and r.calls == ["preflight"]
    broken = copy.deepcopy(frozen)
    broken["sets"]["dev"]["contexts"][0]["context"][5] += 1
    p = tmp_path / "broken.json"
    p.write_text(json.dumps(broken))
    assert orch.run_pipeline(args(tmp_path, p, vfi.sha256_file(p), out_dir=str(tmp_path / "o2")), runner=runner_factory(frozen)) == 3


def test_qualification_process_failure_or_missing_output_is_nonzero(tmp_path, frozen, ffile):
    assert orch.run_pipeline(args(tmp_path, *ffile), runner=runner_factory(frozen, rc_qual=1)) == 10
    assert orch.run_pipeline(args(tmp_path, *ffile, out_dir=str(tmp_path / "o2")), runner=runner_factory(frozen, write_result=False)) == 10


def test_dev_gate_failure_is_nonzero_and_skips_the_smoke(tmp_path, frozen, ffile):
    def mutate(res, fz):
        for rec in _first(res, "dev", "true_B32_g4_ident")["records"].values():
            rec["committed"] = rec["committed"][:-1] + [1]

    r = runner_factory(frozen, make_result(frozen, mutate))
    assert orch.run_pipeline(args(tmp_path, *ffile), runner=r) == 11 and "integration_smoke" not in r.calls


def test_heldout_gate_failure_is_nonzero_even_with_rc_zero_and_dev_pass(tmp_path, frozen, ffile):
    def mutate(res, fz):
        for rec in _first(res, "heldout", "true_B16_g4_ident")["records"].values():
            rec["bonus"] = -1

    r = runner_factory(frozen, make_result(frozen, mutate))
    assert orch.run_pipeline(args(tmp_path, *ffile), runner=r) == 11 and "integration_smoke" not in r.calls


def test_missing_or_incomplete_heldout_is_nonzero(tmp_path, frozen, ffile):
    def drop_ho(res, fz):
        del res["heldout"]

    def drop_run(res, fz):
        del res["dev"]["runs"]["true_B2_g4_perm2"]

    for k, m in enumerate((drop_ho, drop_run)):
        r = runner_factory(frozen, make_result(frozen, m))
        assert orch.run_pipeline(args(tmp_path, *ffile, out_dir=str(tmp_path / f"o{k}")), runner=r) == 11 and "integration_smoke" not in r.calls


def test_zero_observation_in_a_mandatory_comparison_is_nonzero(tmp_path, frozen, ffile):
    fz = copy.deepcopy(frozen)
    for c in fz["sets"]["dev"]["contexts"]:
        c["fp32"][1]["top_logprobs"], c["fp32"][1]["gap"] = [-0.1, -0.2], 0.1      # near tie everywhere at position 1
    p = tmp_path / "f2.json"
    p.write_text(json.dumps(fz))
    r = runner_factory(fz, make_result(fz))
    assert orch.run_pipeline(args(tmp_path, p, vfi.sha256_file(p)), runner=r) == 11 and "integration_smoke" not in r.calls


def test_missing_runtime_evidence_is_nonzero(tmp_path, frozen, ffile):
    def mutate(res, fz):
        res["runtime_evidence_end"]["live_engine"]["triton_fa_calls"] = 0

    r = runner_factory(frozen, make_result(frozen, mutate))
    assert orch.run_pipeline(args(tmp_path, *ffile), runner=r) == 11


def test_call_budget_overrun_is_nonzero(tmp_path, frozen, ffile):
    res = make_result(frozen)
    res["calls"] = 401
    assert orch.run_pipeline(args(tmp_path, *ffile), runner=runner_factory(frozen, res)) == 11


def test_smoke_failure_makes_the_whole_job_fail(tmp_path, frozen, ffile):
    assert orch.run_pipeline(args(tmp_path, *ffile), runner=runner_factory(frozen, rc_smoke=3)) == 12
    assert orch.run_pipeline(args(tmp_path, *ffile, out_dir=str(tmp_path / "o2")), runner=runner_factory(frozen, rc_checks=4)) == 12


# ----------------------------------------------------------------------------- Gate C: frozen inputs and runtime guard
def test_frozen_integrity_report(tmp_path, frozen):
    p = tmp_path / "f.json"
    p.write_text(json.dumps(frozen))
    rep = vfi.check_frozen(p, vfi.sha256_file(p))
    assert rep["ok"] and rep["counts"] == {"dev": 64, "heldout": 32}
    bad = copy.deepcopy(frozen)
    bad["sets"]["dev"]["contexts"].pop()
    p.write_text(json.dumps(bad))
    assert not vfi.check_frozen(p)["ok"]
    bad2 = copy.deepcopy(frozen)
    bad2["sets"]["heldout"]["contexts"][0]["fp32"][0]["top_ids"][0] += 1     # proposal no longer the fp32 chain
    p.write_text(json.dumps(bad2))
    assert not vfi.check_frozen(p)["ok"]
    assert not vfi.check_frozen(tmp_path / "missing.json")["ok"]


def test_previous_job_never_ran_vllm_check(tmp_path):
    ok = {"status": "failed", "calls": 0, "error": "GuardError"}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(ok))
    assert vfi.check_never_seen_by_vllm(p)["ok"]
    p.write_text(json.dumps({**ok, "calls": 3}))
    assert not vfi.check_never_seen_by_vllm(p)["ok"]
    p.write_text(json.dumps({**ok, "runtime_evidence": {}}))
    assert not vfi.check_never_seen_by_vllm(p)["ok"]


def test_runtime_guard_checks_batch_wall_and_zero_price():
    from edge_specsim import runtime_guard as rg

    good = {"verification_batching": {"service_time_model": "batch_wall"}, "controller": {"verifier_price_mode": "zero"}}
    assert rg.check_semantics_config(good)["service_time_model"] == "batch_wall"
    for cfg in ({"verification_batching": {"service_time_model": "amortized_per_request"}, "controller": {"verifier_price_mode": "zero"}},
                {"verification_batching": {"service_time_model": "batch_wall"}, "controller": {"verifier_price_mode": "calibrated_marginal"}},
                {"verification_batching": {"service_time_model": "batch_wall"}, "controller": {}}):
        with pytest.raises(rg.RuntimeGuardError):
            rg.check_semantics_config(cfg)


def test_integration_checks_fail_closed_on_missing_outputs(tmp_path):
    import integration_checks as ic

    rep = ic.check_run(tmp_path, 8, 1, 6, 48)
    assert rep["ok"] is False and rep["problems"]
