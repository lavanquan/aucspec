# Qwen Edge Speculative-Decoding Simulator — Candidate Verification Version

Architecture:

- GPU 0: `Qwen/Qwen2.5-7B-Instruct` loaded by vLLM as the target verifier.
- GPU 1–3: persistent `Qwen/Qwen2.5-1.5B-Instruct` draft workers.
- CPU: dataset streaming, virtual clients, online controller, RTT/uplink delay,
  censored acceptance learning, batch formation, and CSV metrics.

## What changed

The target now receives the candidate token IDs directly. For each verification
batch, the code forms token sequences:

```text
context tokens + proposed candidate tokens
```

It sends all sequences to one `vLLM.LLM.generate(...)` call with
`prompt_logprobs=1` and `max_tokens=1`:

1. Prompt log-probabilities expose the target model's greedy token at every
   candidate position.
2. Candidate tokens are accepted until the first position where the candidate
   differs from the target argmax.
3. On a mismatch, the target argmax at that position is committed as the
   correction token.
4. If every candidate is accepted, the one generated token is committed as the
   bonus token.

This is real batched candidate verification rather than asking the target to
autoregressively regenerate `gamma + 1` tokens before comparison. It uses a
single batched vLLM call for all requests collected by `VerificationBatcher`.

## Supported datasets

The project intentionally supports only:

- `gsm8k` → `openai/gsm8k`, config `main`.
- `math` → `EleutherAI/hendrycks_math`.
- `cnn_dailymail` → `abisee/cnn_dailymail`, version `3.0.0`.

Datasets are loaded in streaming mode. Set `dataset.num_questions` to run a
small test without iterating through the full dataset.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Some gated or rate-limited Hugging Face environments may require:

```bash
huggingface-cli login
```

## Run

Do **not** start a separate vLLM HTTP server. The target is loaded directly by
the simulator on GPU 0.

```bash
python scripts/run_simulation.py --config configs/default.yaml
```

You can override the dataset and sample count from the command line:

```bash
python scripts/run_simulation.py \
  --config configs/default.yaml \
  --dataset gsm8k \
  --num-questions 5
```

Add `--detailed-log` if you want full per-round token traces in `results/rounds.csv`.
Add `--num-clients 4` if you want to split the questions across a different number of clients.

MATH example:

```bash
python scripts/run_simulation.py \
  --dataset math \
  --num-questions 8 \
  --math-subject geometry \
  --split test
```

Summarize results:

```bash
python scripts/summarize.py results/rounds.csv
```

Inspect the full trace for one prompt:

```bash
python scripts/trace_sample.py gsm8k-test-1 --csv results/rounds.csv
python scripts/trace_sample.py gsm8k-test-1 --csv results/rounds.csv --json --output trace.json
python scripts/trace_sample.py gsm8k-test-1 --csv results/rounds.csv --trace
python scripts/trace_sample.py gsm8k-test-1 --csv results/rounds.csv --model Qwen/Qwen2.5-7B-Instruct
```

## Select a dataset and question count

### GSM8K, 10 questions

```yaml
dataset:
  name: gsm8k
  split: test
  num_questions: 10
  max_new_tokens: 256
```

### MATH, 20 questions across all subjects

```yaml
dataset:
  name: math
  split: test
  num_questions: 20
  math_subject: all
  max_new_tokens: 512
```

A single subject can also be selected, for example:

```yaml
  math_subject: geometry
```

### CNN/DailyMail, 5 articles

```yaml
dataset:
  name: cnn_dailymail
  split: test
  num_questions: 5
  max_new_tokens: 160
  max_article_chars: 12000
```

## Important assumptions

- Target and draft models must share the same tokenizer/vocabulary. The selected
  Qwen2.5 models satisfy this intended setup.
- Verification is greedy (`temperature=0`). Exact stochastic rejection sampling
  is not implemented.
- vLLM returns prompt log-probabilities for the candidate positions. The code
  chooses the highest-log-probability token at each position as the target
  greedy decision.
- Draft workers currently re-prefill each client's context on every round. This
  is functionally correct but not yet an optimized per-client KV-cache design.
- The target batch latency is divided equally among requests for per-client CSV
  accounting. The complete batch ID and actual batch size are also recorded.

## Per-client KV cache version

This revision adds persistent draft-side KV state and target-side vLLM prefix
cache reuse.

### Draft cache

Each virtual client is permanently assigned to one draft worker. The worker keeps
one `DynamicCache` per client on its GPU. A speculative round works as follows:

1. Start from the client's confirmed-prefix cache.
2. Propose `gamma` greedy tokens by advancing that cache token by token.
3. Crop the cache back to the confirmed prefix because the proposal is tentative.
4. After target verification, advance the cache only with the committed tokens
   (accepted candidate prefix plus correction/bonus token).
5. Reset the client cache when the client starts a new dataset sample.

This avoids re-prefilling the complete client context on every draft round.

### Target cache

The target is still implemented with vLLM. The public offline `LLM.generate`
API does not expose Hugging Face-style `past_key_values` objects that the
application can manually save and reload for each client. Instead the target is
created with:

```python
enable_prefix_caching=True
```

Every verification request contains the client's confirmed context followed by
candidate tokens. vLLM Automatic Prefix Caching hashes token blocks and reuses
matching target KV blocks from earlier rounds. Thus the application resubmits
token IDs, while vLLM owns allocation, lookup, eviction, and reuse of target KV
blocks internally.

This distinction is important:

- Draft side: explicit application-managed cache per client.
- Target side: vLLM-managed prefix KV blocks, logically separated by token
  prefix rather than an exported Python cache object.

When GPU memory pressure causes vLLM to evict a target prefix block, that prefix
will be recomputed on a later request. Increase target GPU memory headroom or
reduce active clients/context length if cache churn is excessive.
