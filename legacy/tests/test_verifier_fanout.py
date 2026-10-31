"""CPU-only check that verify_mode="prefix_fanout" (gamma + 1 one-token
prompts sharing the cached context) commits exactly what the original
prompt_logprobs verification commits, against a fake deterministic vLLM.
torch / transformers / vllm are stubbed when missing.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_MISSING = {top for top in ("torch", "transformers", "vllm")
            if top not in sys.modules and importlib.util.find_spec(top) is None}
for _name in ("torch", "transformers", "transformers.models", "transformers.models.auto",
              "transformers.models.auto.configuration_auto", "vllm", "vllm.inputs"):
    if _name.split(".")[0] in _MISSING:
        _mod = types.ModuleType(_name)
        _mod.__spec__ = importlib.machinery.ModuleSpec(_name, None)
        sys.modules[_name] = _mod
_tr = sys.modules["transformers"]
for _attr in ("AutoConfig", "AutoTokenizer"):
    if not hasattr(_tr, _attr):
        setattr(_tr, _attr, object)
_ca = sys.modules["transformers.models.auto.configuration_auto"]
if not hasattr(_ca, "CONFIG_MAPPING"):
    _ca.CONFIG_MAPPING = SimpleNamespace(register=lambda *a, **k: None)
_vllm = sys.modules["vllm"]
if not hasattr(_vllm, "SamplingParams"):
    _vllm.LLM = object
    _vllm.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    sys.modules["vllm.inputs"].TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}

# load the real module even if another test stubbed edge_specsim.target_client
_path = Path(__file__).resolve().parents[1] / "src" / "edge_specsim" / "target_client.py"
_spec = importlib.util.spec_from_file_location("edge_specsim._target_client_real", _path)
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)

from edge_specsim.models import DraftResult  # noqa: E402

VOCAB = 50


def _dist(prefix):
    best = (sum(prefix) * 31 + 7 * len(prefix)) % VOCAB
    return {best: -0.1, (best + 1) % VOCAB: -2.5}


def _ids(prompt):
    return prompt["prompt_token_ids"] if isinstance(prompt, dict) else prompt.prompt_token_ids


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling, use_tqdm=False):
        if not isinstance(prompts, list):
            prompts = [prompts]
        self.calls.append((len(prompts), getattr(sampling, "prompt_logprobs", None)))
        outs = []
        for prompt in prompts:
            ids = list(_ids(prompt))
            d = _dist(ids)
            best = max(d, key=d.get)
            prompt_lp = None
            if getattr(sampling, "prompt_logprobs", None):
                prompt_lp = [None] + [_dist(ids[:i]) for i in range(1, len(ids))]
            outs.append(SimpleNamespace(
                prompt_logprobs=prompt_lp,
                outputs=[SimpleNamespace(token_ids=[best], logprobs=[d])],
            ))
        return outs


def _verifier(mode):
    v = tc.VLLMCandidateVerifier.__new__(tc.VLLMCandidateVerifier)
    v.verify_mode = mode
    v.enable_prefix_caching = mode == "prefix_fanout"
    v._last_context = {}
    v.llm = FakeLLM()
    v.sampling = SimpleNamespace(prompt_logprobs=2 if mode == "prompt_logprobs" else None, logprobs=2)
    v.prefill_sampling = SimpleNamespace(prompt_logprobs=None)
    v.greedy_only = True
    v.eos_token_id = None
    v.vocab_size = VOCAB
    v.rng = random.Random(0)
    v.tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens=True: "")
    return v


def _requests(contexts, round_index, gamma):
    reqs = []
    for client, ctx in enumerate(contexts):
        proposal, prefix = [], list(ctx)
        for k in range(gamma):
            d = _dist(prefix)
            tok = max(d, key=d.get) if k < (client + round_index) % (gamma + 1) else (k * 7 + 3) % VOCAB
            proposal.append(tok)
            prefix.append(tok)
        draft = DraftResult(client_id=client, token_ids=proposal, draft_logprobs=[None] * gamma,
                            text="", latency_ms=0.0, worker_id=-1, distribution_payload="delta_proposal")
        reqs.append((list(ctx), draft, 10 ** 6))
    return reqs


@pytest.mark.parametrize("gamma", [0, 1, 4])
def test_fanout_commits_match_prompt_logprobs(gamma):
    ref, fan = _verifier("prompt_logprobs"), _verifier("prefix_fanout")
    ctx_ref = [[1, 2, 3, 4], [5, 6], [7, 8, 9]]
    ctx_fan = [list(c) for c in ctx_ref]
    for r in range(5):
        res_ref = ref.verify_batch_sync(_requests(ctx_ref, r, gamma))
        res_fan = fan.verify_batch_sync(_requests(ctx_fan, r, gamma))
        for a, b in zip(res_ref, res_fan):
            assert a.accepted_length == b.accepted_length
            assert a.committed_token_ids == b.committed_token_ids
            assert a.target_greedy_token_ids == b.target_greedy_token_ids
            assert a.correction_token_id == b.correction_token_id
            assert a.bonus_token_id == b.bonus_token_id
        for c_ref, c_fan, a in zip(ctx_ref, ctx_fan, res_ref):
            c_ref.extend(a.committed_token_ids)
            c_fan.extend(a.committed_token_ids)


def test_fanout_prefills_only_new_contexts():
    v = _verifier("prefix_fanout")
    ctx = [[1, 2, 3], [4, 5]]
    res = v.verify_batch_sync(_requests(ctx, 0, 2))
    assert v.llm.calls[0] == (2, None)          # prefill of both new contexts
    assert v.llm.calls[1] == (2 * 3, None)      # gamma + 1 prompts per client
    for c, a in zip(ctx, res):
        c.extend(a.committed_token_ids)
    v.llm.calls.clear()
    v.verify_batch_sync(_requests(ctx, 1, 2))
    assert v.llm.calls == [(6, None)]           # contexts extend last round's: no prefill
    v.llm.calls.clear()
    v.verify_batch_sync(_requests([[9, 9], ctx[1]], 2, 2))
    assert v.llm.calls[0] == (1, None)          # client 0 started a new sample


def test_best_token_tie_breaks_to_smallest_id():
    assert tc.VLLMCandidateVerifier._best_token_id({9: -0.5, 3: -0.5, 4: -1.0}) == 3
