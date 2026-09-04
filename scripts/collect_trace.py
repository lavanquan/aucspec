"""Real accept/reject trace collection: Qwen2.5 draft models verified
against Qwen2.5-7B-Instruct target via vLLM batched candidate verification.

TASKS.md T4.2. Draft/target pairing from docs/model_choices.md. Reuses
legacy/src/edge_specsim/target_client.VLLMCandidateVerifier's batched
rejection-sampling verification directly (not reimplemented) for the
target side; draft proposals are generated greedily
(distribution_payload="delta_proposal") with plain
transformers.AutoModelForCausalLM.generate(), recomputing from the full
context each round rather than legacy's persistent-KV-cache DraftWorker --
simpler and still correct for trace collection, where efficiency matters
far less than it would for actual serving.

Writes one JSONL line per round in the schema sim/workloads/trace_loader.py
expects: {stream_id, domain, round_index, gamma, accepted, all_accepted}.

Usage (needs 2 GPUs -- target vLLM instance defaults to the first visible
device, draft model goes on --draft-device):
    salloc --gres=gpu:2 --partition=gpu-dev --account=pawsey1257-gpu \
        --exclusive=user srun python3 scripts/collect_trace.py

Prompt set: a small curated chat+code starter set (10 prompts each), not a
full ShareGPT/HumanEval download -- EXPERIMENTS.md's "chat + code mix"
requirement is met qualitatively; swap in a larger corpus later via
sim/workloads/trace_loader.py's schema without touching the collection
logic here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "legacy" / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from edge_specsim.models import DraftProposal  # noqa: E402
from edge_specsim.target_client import VLLMCandidateVerifier  # noqa: E402

CHAT_PROMPTS = [
    "Explain the difference between TCP and UDP in two sentences.",
    "What are three ways to reduce the carbon footprint of a data center?",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "Give me a recipe idea using chicken, rice, and broccoli.",
    "What is the capital of Australia and why was it chosen?",
    "Explain photosynthesis to a 10 year old.",
    "What are the pros and cons of remote work?",
    "Describe how a neural network learns from data.",
    "Write a short motivational message for someone starting a new job.",
    "What causes ocean tides?",
]

CODE_PROMPTS = [
    "Write a Python function that checks if a string is a palindrome.",
    "Write a Python function to compute the nth Fibonacci number iteratively.",
    "Write a Python function that merges two sorted lists into one sorted list.",
    "Write a Python function that counts the number of vowels in a string.",
    "Write a Python function that returns the factorial of a non-negative integer.",
    "Write a Python function that finds the maximum subarray sum (Kadane's algorithm).",
    "Write a Python function that reverses a singly linked list.",
    "Write a Python function that checks if a binary tree is height-balanced.",
    "Write a Python function that implements binary search on a sorted list.",
    "Write a Python function that removes duplicates from a list while preserving order.",
]


def load_draft_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    return tokenizer, model


def draft_greedy(tokenizer, model, device: str, context_token_ids: list[int], gamma: int) -> DraftProposal:
    """Greedily draft `gamma` tokens from `context_token_ids`. Uses
    distribution_payload="delta_proposal" (token ids only, no logprobs) --
    valid for lossless verification exactly when the draft is greedy, which
    it is here (do_sample=False)."""
    if gamma <= 0:
        return DraftProposal(
            client_id=0,
            token_ids=[],
            draft_logprobs=[],
            text="",
            latency_ms=0.0,
            worker_id=0,
            distribution_payload="delta_proposal",
        )
    input_ids = torch.tensor([context_token_ids], device=device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=gamma,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, input_ids.shape[1] :].tolist()
    return DraftProposal(
        client_id=0,
        token_ids=new_tokens,
        draft_logprobs=[],
        text="",
        latency_ms=0.0,
        worker_id=0,
        distribution_payload="delta_proposal",
    )


def collect_stream(
    stream_id: str,
    domain: str,
    prompt: str,
    gamma: int,
    num_rounds: int,
    verifier: VLLMCandidateVerifier,
    draft_tokenizer,
    draft_model,
    draft_device: str,
    out_f,
) -> None:
    history_token_ids: list[int] = []
    for round_index in range(num_rounds):
        context_token_ids = verifier.build_context_token_ids(prompt, history_token_ids)
        draft = draft_greedy(draft_tokenizer, draft_model, draft_device, context_token_ids, gamma)
        # A round delivers at most gamma+1 tokens (accepted prefix + one
        # bonus/correction token), so that is the natural per-round cap.
        remaining_tokens = gamma + 1
        result = verifier.verify_batch_sync([(context_token_ids, draft, remaining_tokens)])[0]
        accepted = result.accepted_length
        all_accepted = accepted == gamma
        record = {
            "stream_id": stream_id,
            "domain": domain,
            "round_index": round_index,
            "gamma": gamma,
            "accepted": accepted,
            "all_accepted": all_accepted,
        }
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()
        if not result.committed_token_ids or result.reached_eos:
            break
        history_token_ids.extend(result.committed_token_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--draft-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--rounds-per-prompt", type=int, default=15)
    parser.add_argument("--draft-device", default="cuda:1")
    parser.add_argument("--out", default="results/trace_real.jsonl")
    args = parser.parse_args()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading target verifier: {args.target_model}", flush=True)
    verifier = VLLMCandidateVerifier(model_name=args.target_model)
    print(f"Loading draft model: {args.draft_model} on {args.draft_device}", flush=True)
    draft_tokenizer, draft_model = load_draft_model(args.draft_model, args.draft_device)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, prompt in enumerate(CHAT_PROMPTS):
            collect_stream(
                stream_id=f"chat_{i}",
                domain="chat",
                prompt=prompt,
                gamma=args.gamma,
                num_rounds=args.rounds_per_prompt,
                verifier=verifier,
                draft_tokenizer=draft_tokenizer,
                draft_model=draft_model,
                draft_device=args.draft_device,
                out_f=out_f,
            )
            print(f"done chat_{i}", flush=True)
        for i, prompt in enumerate(CODE_PROMPTS):
            collect_stream(
                stream_id=f"code_{i}",
                domain="code",
                prompt=prompt,
                gamma=args.gamma,
                num_rounds=args.rounds_per_prompt,
                verifier=verifier,
                draft_tokenizer=draft_tokenizer,
                draft_model=draft_model,
                draft_device=args.draft_device,
                out_f=out_f,
            )
            print(f"done code_{i}", flush=True)

    print(f"TRACE_COLLECTION_DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
