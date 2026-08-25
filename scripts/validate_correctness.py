from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_specsim.dataset import load_samples
from edge_specsim.draft_worker import DraftRequest, DraftWorker
from edge_specsim.models import DraftResult, PromptSample
from edge_specsim.target_client import VLLMCandidateVerifier


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _apply_overrides(
    cfg: dict,
    dataset_name: str | None,
    num_questions: int | None,
    split: str | None,
    math_subject: str | None,
    target_model: str | None,
    draft_model: str | None,
) -> dict:
    dataset_cfg = cfg.setdefault("dataset", {})
    if dataset_name is not None:
        dataset_cfg["name"] = dataset_name
    if num_questions is not None:
        dataset_cfg["num_questions"] = num_questions
    if split is not None:
        dataset_cfg["split"] = split
    if math_subject is not None:
        dataset_cfg["math_subject"] = math_subject

    model_cfg = cfg.setdefault("models", {})
    if target_model is not None:
        model_cfg["target"] = target_model
    if draft_model is not None:
        model_cfg["draft"] = draft_model
    return cfg


def _validate_greedy_config(cfg: dict) -> float:
    decoding_cfg = cfg.setdefault("decoding", {})
    mode = str(decoding_cfg.get("mode", "greedy")).lower()
    temperature = float(
        decoding_cfg.get("temperature", cfg.get("simulation", {}).get("temperature", 0.0))
    )
    if mode != "greedy" or temperature != 0.0:
        raise ValueError(
            "validate_correctness.py requires greedy decoding. Set "
            "`decoding.mode: greedy` and `decoding.temperature: 0.0`."
        )
    return temperature


async def _run_speculative_greedy(
    sample: PromptSample,
    target: VLLMCandidateVerifier,
    worker: DraftWorker,
    gamma_max: int,
    temperature: float,
) -> list[int]:
    client_id = 0
    history_token_ids: list[int] = []
    remaining = sample.max_new_tokens
    await worker.reset_client(client_id)
    target.set_confirmed_prefix(client_id, [])

    while remaining > 0:
        context_token_ids = target.build_context_token_ids(sample.prompt, history_token_ids)
        gamma = min(gamma_max, max(0, remaining - 1))
        draft_confirmed_length = await worker.confirmed_length(client_id)
        if draft_confirmed_length is not None:
            assert draft_confirmed_length == len(context_token_ids)

        if gamma > 0:
            draft = await worker.generate(
                DraftRequest(
                    client_id=client_id,
                    context_token_ids=context_token_ids,
                    gamma=gamma,
                    temperature=temperature,
                )
            )
        else:
            draft = DraftResult(client_id, [], [], "", 0.0, worker.worker_id)

        verification = target.verify_batch_sync(
            [(context_token_ids, draft, remaining)]
        )[0]
        committed = verification.committed_token_ids
        await worker.commit(client_id, committed)
        history_token_ids.extend(committed)
        target.set_confirmed_prefix(client_id, history_token_ids)

        committed_length = await worker.confirmed_length(client_id)
        expected_context_length = len(
            target.build_context_token_ids(sample.prompt, history_token_ids)
        )
        assert committed_length == expected_context_length
        assert target.confirmed_prefix_hash(client_id) == target.prefix_hash(history_token_ids)

        remaining -= len(committed)
        if verification.reached_eos or not committed:
            break

    return history_token_ids


async def _validate_samples(cfg: dict, verbose: bool) -> None:
    samples = load_samples(cfg["dataset"])
    model_cfg = cfg["models"]
    gamma_max = int(cfg["simulation"]["gamma_max"])
    temperature = _validate_greedy_config(cfg)

    target = VLLMCandidateVerifier(
        model_name=model_cfg["target"],
        gpu_memory_utilization=float(model_cfg.get("target_gpu_memory_utilization", 0.88)),
        max_model_len=int(model_cfg.get("target_max_model_len", 8192)),
        dtype=str(model_cfg.get("target_dtype", "auto")),
        enforce_eager=bool(model_cfg.get("target_enforce_eager", False)),
        enable_prefix_caching=bool(model_cfg.get("target_enable_prefix_caching", True)),
        sampling_temperature=temperature,
        seed=int(cfg["simulation"]["seed"]),
    )
    worker = DraftWorker(0, model_cfg["draft"], model_cfg["draft_devices"][0])

    mismatches: list[str] = []
    for sample in samples:
        target_only_token_ids = target.generate_target_only_greedy(
            sample.prompt, sample.max_new_tokens
        )
        speculative_token_ids = await _run_speculative_greedy(
            sample, target, worker, gamma_max, temperature
        )

        if target_only_token_ids != speculative_token_ids:
            mismatches.append(sample.sample_id)
            target_text = target.tokenizer.decode(target_only_token_ids, skip_special_tokens=True)
            speculative_text = target.tokenizer.decode(
                speculative_token_ids, skip_special_tokens=True
            )
            raise AssertionError(
                f"Mismatch for sample_id={sample.sample_id}\n"
                f"target_only_token_ids={target_only_token_ids}\n"
                f"speculative_token_ids={speculative_token_ids}\n"
                f"target_only_text={target_text!r}\n"
                f"speculative_text={speculative_text!r}"
            )

        if verbose:
            print(
                f"sample_id={sample.sample_id} ok tokens={len(speculative_token_ids)}"
            )

    print(f"Validated {len(samples)} sample(s): target-only greedy == speculative greedy")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare target-only greedy decoding against speculative greedy decoding."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=["gsm8k", "math", "cnn_dailymail"])
    parser.add_argument("--num-questions", type=int)
    parser.add_argument("--split")
    parser.add_argument("--math-subject")
    parser.add_argument("--target-model")
    parser.add_argument("--draft-model")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    cfg = _apply_overrides(
        cfg,
        dataset_name=args.dataset,
        num_questions=args.num_questions,
        split=args.split,
        math_subject=args.math_subject,
        target_model=args.target_model,
        draft_model=args.draft_model,
    )
    asyncio.run(_validate_samples(cfg, verbose=args.verbose))


if __name__ == "__main__":
    main()
