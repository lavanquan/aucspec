"""CPU-only audit tests for the V1 greedy verification path (synthetic prompt-logprob data).

They pin how `VLLMCandidateVerifier.verify_batch_sync` reads vLLM output, WITHOUT changing it:
index alignment, handling of the observed-token entry, tie behaviour, EOS and acceptance accounting.
"""

from __future__ import annotations

import importlib
import random
import sys
import types
from types import SimpleNamespace

import pytest

EOS = 99


@pytest.fixture
def tc(monkeypatch):
    """Import edge_specsim.target_client with torch/transformers/vllm stubbed."""
    stubs = {}
    stubs["torch"] = types.ModuleType("torch")
    t = types.ModuleType("transformers")
    t.AutoConfig = t.AutoTokenizer = object
    stubs["transformers"] = t
    for name in ("transformers.models", "transformers.models.auto"):
        stubs[name] = types.ModuleType(name)
    cm = types.ModuleType("transformers.models.auto.configuration_auto")
    cm.CONFIG_MAPPING = SimpleNamespace(register=lambda *a, **k: None)
    stubs["transformers.models.auto.configuration_auto"] = cm
    v = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    v.LLM = object
    v.SamplingParams = SamplingParams
    stubs["vllm"] = v
    vi = types.ModuleType("vllm.inputs")
    vi.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
    stubs["vllm.inputs"] = vi
    for k, mod in stubs.items():
        monkeypatch.setitem(sys.modules, k, mod)
    monkeypatch.delitem(sys.modules, "edge_specsim.target_client", raising=False)
    mod = importlib.import_module("edge_specsim.target_client")
    yield mod
    sys.modules.pop("edge_specsim.target_client", None)


def lp(d):
    """dict token_id -> logprob, as vLLM Logprob objects (insertion order preserved)."""
    return {int(k): SimpleNamespace(logprob=float(v)) for k, v in d.items()}


class FakeLLM:
    """Returns scripted prompt_logprobs: `entries` maps absolute prompt index -> {token: logprob}."""

    def __init__(self, entries, bonus):
        self.entries, self.bonus = entries, bonus

    def generate(self, prompts, sampling, use_tqdm=False):
        outs = []
        for p in prompts:
            ids = p["prompt_token_ids"]
            plp = [None] + [lp(self.entries.get(i, {ids[i]: -0.1})) for i in range(1, len(ids))]
            outs.append(SimpleNamespace(prompt_logprobs=plp, outputs=[SimpleNamespace(
                token_ids=[self.bonus], logprobs=[lp({self.bonus: -0.2})])]))
        return outs


def make_verifier(mod, entries, bonus=7):
    v = object.__new__(mod.VLLMCandidateVerifier)
    v.model_name = "fake"
    v.rng = random.Random(0)
    v._confirmed_prefix_hashes = {}
    v.tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens=True: "x")
    v.vocab_size = 100
    v.eos_token_id = EOS
    v.greedy_only = True
    v.effective_max_logprobs = 1
    v.sampling = object()
    v.llm = FakeLLM(entries, bonus)
    return v


def draft(tokens):
    from edge_specsim.models import DraftResult

    return DraftResult(client_id=0, token_ids=list(tokens), draft_logprobs=[], text="", latency_ms=0.0,
                       worker_id=-1, distribution_payload="delta_proposal")


CTX = [1, 2, 3, 4, 5]  # len 5 -> candidate positions are prompt indices 5, 6, 7


def run(v, proposal, remaining=96):
    return v.verify_batch_sync([(CTX, draft(proposal), remaining)])[0]


def test_reads_prompt_logprobs_at_len_ctx_plus_offset(tc):
    prop = [10, 11, 12]
    entries = {4: {50: -0.1}, 5: {10: -0.1}, 6: {11: -0.1}, 7: {12: -0.1}, 8: {51: -0.1}}
    r = run(make_verifier(tc, entries), prop)
    assert r.accepted_length == 3 and r.committed_token_ids == prop + [7] and r.bonus_token_id == 7


def test_off_by_one_in_the_data_would_be_visible(tc):
    # the correct argmax placed one index EARLIER than len(ctx)+offset is not what the verifier reads
    prop = [10, 11, 12]
    entries = {4: {10: -0.1}, 5: {11: -0.1}, 6: {12: -0.1}, 7: {52: -0.1}}
    r = run(make_verifier(tc, entries), prop)
    assert r.accepted_length == 0 and r.correction_token_id == 11  # it read index 5, not index 4


def test_observed_token_in_dict_does_not_hide_the_true_argmax(tc):
    # vLLM returns the top-k PLUS the observed prompt token: argmax is taken over both, so a
    # lower-probability observed token is rejected in favour of the top-1 entry.
    entries = {5: {20: -0.3, 10: -2.0}}
    r = run(make_verifier(tc, entries), [10])
    assert r.accepted_length == 0 and r.correction_token_id == 20 and r.target_greedy_token_ids == [20]


def test_k1_dict_with_only_the_observed_token_accepts(tc):
    r = run(make_verifier(tc, {5: {10: -0.4}}), [10])
    assert r.accepted_length == 1


def test_exact_tie_is_decided_by_dict_insertion_order(tc):
    # V1 `max` returns the FIRST maximal entry: with an exact logprob tie the outcome depends on
    # the order in which vLLM inserted the entries, not on the token id.
    other_first = run(make_verifier(tc, {5: {20: -1.0, 10: -1.0}}), [10])
    proposed_first = run(make_verifier(tc, {5: {10: -1.0, 20: -1.0}}), [10])
    assert other_first.accepted_length == 0 and other_first.correction_token_id == 20
    assert proposed_first.accepted_length == 1
    # the smaller id wins in neither rule by construction here: id 10 < 20 yet it loses when listed second
    assert 10 < 20


def test_eos_candidate_is_accepted_then_stops(tc):
    entries = {5: {10: -0.1}, 6: {EOS: -0.1}, 7: {12: -0.1}}
    r = run(make_verifier(tc, entries), [10, EOS, 12])
    assert r.accepted_length == 2 and r.reached_eos and r.committed_token_ids == [10, EOS]
    from importlib import import_module  # calibration accounting rejects EOS-truncated calls
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "scripts"))
    cal = import_module("calibrate_verifier")
    assert cal.check_realized([3], [r]) == (False, [2])


def test_full_acceptance_commits_gamma_plus_bonus(tc):
    prop = [10, 11]
    r = run(make_verifier(tc, {5: {10: -0.1}, 6: {11: -0.1}}, bonus=33), prop)
    assert (r.accepted_length, len(r.committed_token_ids), r.bonus_token_id, r.correction_token_id) == (2, 3, 33, None)
    assert r.target_greedy_token_ids == [10, 11, 33]
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "scripts"))
    import calibrate_verifier as cal
    assert cal.check_realized([2], [r]) == (True, [2])


def test_reject_at_offset_two_commits_prefix_plus_correction(tc):
    r = run(make_verifier(tc, {5: {10: -0.1}, 6: {11: -0.1}, 7: {40: -0.1, 12: -3.0}}), [10, 11, 12])
    assert (r.accepted_length, r.committed_token_ids, r.correction_token_id) == (2, [10, 11, 40], 40)
    assert r.bonus_token_id is None and r.target_greedy_token_ids == [10, 11, 40]


def test_bonus_token_comes_from_the_sampler_not_from_the_dict_rule(tc):
    # After full acceptance the extra token is outputs[0].token_ids[0] (vLLM's own argmax of the last
    # position), whereas candidate positions use the dict rule: two different argmax implementations.
    r = run(make_verifier(tc, {5: {10: -0.1}}, bonus=44), [10])
    assert r.bonus_token_id == 44 and r.committed_token_ids == [10, 44]


def test_remaining_tokens_truncates_commit_but_not_accepted_length(tc):
    r = run(make_verifier(tc, {5: {10: -0.1}, 6: {11: -0.1}}), [10, 11], remaining=1)
    assert r.accepted_length == 2 and r.committed_token_ids == [10]
