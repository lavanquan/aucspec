# Hugging Face causal speculative runtime

This is the non-vLLM correctness path for integrating aucspec with SpecPlan on
Setonix.

The existing aucspec target verifier uses vLLM prompt-logprob and ordinary
generation paths. SpecPlan qualification found batch or execution-shape
dependent greedy outputs under the tested vLLM 0.9.1, 0.18.0, and
0.31.0/TRITON_ATTN runtime lines on MI250X.

The speculative accept/correct semantics previously passed an 8/8 sequential
causal oracle. This implementation keeps aucspec greedy speculative semantics
but replaces target execution with Transformers AutoModelForCausalLM.

For a context of length C and g draft tokens, one causal target forward runs on
context plus draft. Logits at C-1+j predict draft token j. The final logits
position predicts the bonus token. A mismatch commits the target argmax as the
correction token. Exact ties use torch.argmax, therefore the smallest token ID.

scripts/specplan_worker_hf.py implements SpecPlan external-runtime protocol
version 1 over stdin/stdout JSONL. It loads target on cuda:0 and draft on
cuda:1 by default and verifies that both tokenizers have identical vocabularies.

The first implementation is correctness-first: one target full causal forward
per speculative cycle, deterministic draft generation from the full confirmed
context, no target batching, and no target KV persistence. It is not a
throughput baseline. Any later cache or batching optimization must rerun the
same exactness gates.


For qualification, the worker also exposes an optional target_only operation.
It runs repeated one-token greedy target decisions through the same
HFGreedyCandidateVerifier instance, so target-only and speculative outputs can
be compared without loading a second target model.
