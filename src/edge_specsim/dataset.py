from __future__ import annotations

from collections import deque
from itertools import cycle
import time
from typing import Any, Iterable, Iterator

from .models import PromptSample

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _require_datasets():
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets first: pip install datasets") from exc
    return load_dataset, DownloadConfig


def _dataset_cache_dir(cfg: dict[str, Any]) -> str | None:
    cache_dir = cfg.get("cache_dir")
    if cache_dir is None:
        return None
    return str(cache_dir)


def _load_hf_dataset(
    cfg: dict[str, Any],
    path: str,
    name: str | None = None,
    *,
    split: str,
    streaming: bool,
):
    load_dataset, DownloadConfig = _require_datasets()
    cache_dir = _dataset_cache_dir(cfg)
    retries = max(0, int(cfg.get("hf_retries", 3)))
    retry_delay_seconds = max(0.0, float(cfg.get("hf_retry_delay_seconds", 2.0)))
    prefer_local_cache = bool(cfg.get("prefer_local_cache", True))

    attempts: list[dict[str, Any]] = []
    if prefer_local_cache:
        attempts.append({"local_files_only": True})
    attempts.extend({"local_files_only": False} for _ in range(retries + 1))

    last_exc: Exception | None = None
    for attempt_index, attempt_kwargs in enumerate(attempts):
        try:
            download_config = DownloadConfig(
                cache_dir=cache_dir,
                local_files_only=bool(attempt_kwargs["local_files_only"]),
                max_retries=max(1, retries),
            )
            return load_dataset(
                path,
                name,
                split=split,
                streaming=streaming,
                cache_dir=cache_dir,
                download_config=download_config,
            )
        except Exception as exc:
            last_exc = exc
            is_last_attempt = attempt_index == len(attempts) - 1
            if is_last_attempt:
                break
            sleep_seconds = retry_delay_seconds * (2**attempt_index)
            if sleep_seconds > 0.0:
                time.sleep(sleep_seconds)
    assert last_exc is not None
    raise RuntimeError(
        f"Failed to load dataset path={path!r} name={name!r} split={split!r} "
        f"after {len(attempts)} attempts. "
        "If this is a transient Hugging Face network error, rerun after the cache warms."
    ) from last_exc


def _maybe_shuffle(stream: Any, cfg: dict[str, Any], seed_offset: int = 0):
    if cfg.get("shuffle", True):
        shuffle_seed = int(cfg.get("seed", 42)) + seed_offset
        try:
            return stream.shuffle(
                seed=shuffle_seed,
                buffer_size=int(cfg.get("shuffle_buffer", 1000)),
            )
        except TypeError:
            return stream.shuffle(seed=shuffle_seed)
    return stream


def _take(iterator: Iterable[PromptSample], count: int) -> list[PromptSample]:
    result: list[PromptSample] = []
    for sample in iterator:
        result.append(sample)
        if len(result) >= count:
            break
    if not result:
        raise ValueError("The selected dataset produced no samples")
    return result


def _gsm8k_samples(cfg: dict[str, Any]) -> Iterator[PromptSample]:
    split = cfg.get("split", "test")
    streaming = bool(cfg.get("streaming", False))
    ds = _load_hf_dataset(
        cfg,
        "openai/gsm8k",
        "main",
        split=split,
        streaming=streaming,
    )
    ds = _maybe_shuffle(ds, cfg)
    max_new = int(cfg.get("max_new_tokens", 256))
    for index, row in enumerate(ds):
        question = str(row["question"]).strip()
        yield PromptSample(
            sample_id=f"gsm8k-{split}-{index}",
            dataset_name="gsm8k",
            prompt=(
                "Solve the following grade-school mathematics problem. "
                "Show the reasoning clearly, then give the final answer.\n\n"
                f"Question: {question}\n\nAnswer:"
            ),
            max_new_tokens=max_new,
            reference=str(row.get("answer", "")),
        )


def _math_subject_stream(cfg: dict[str, Any], subject: str, seed_offset: int):
    split = cfg.get("split", "test")
    ds = _load_hf_dataset(
        cfg,
        "EleutherAI/hendrycks_math",
        subject,
        split=split,
        streaming=bool(cfg.get("streaming", True)),
    )
    return _maybe_shuffle(ds, cfg, seed_offset)


def _math_samples(cfg: dict[str, Any]) -> Iterator[PromptSample]:
    selected = cfg.get("math_subject", "all")
    subjects = MATH_SUBJECTS if selected == "all" else [selected]
    invalid = [subject for subject in subjects if subject not in MATH_SUBJECTS]
    if invalid:
        raise ValueError(f"Unsupported MATH subject: {invalid[0]}")

    streams = [iter(_math_subject_stream(cfg, subject, i)) for i, subject in enumerate(subjects)]
    active = list(zip(subjects, streams))
    max_new = int(cfg.get("max_new_tokens", 512))
    index = 0
    while active:
        next_active = []
        for subject, iterator in active:
            try:
                row = next(iterator)
            except StopIteration:
                continue
            problem = str(row["problem"]).strip()
            yield PromptSample(
                sample_id=f"math-{subject}-{index}",
                dataset_name="math",
                prompt=(
                    "Solve the following competition mathematics problem. "
                    "Give a rigorous derivation and place the final answer at the end.\n\n"
                    f"Problem: {problem}\n\nSolution:"
                ),
                max_new_tokens=max_new,
                reference=str(row.get("solution", "")),
            )
            index += 1
            next_active.append((subject, iterator))
        active = next_active


def _cnn_dailymail_samples(cfg: dict[str, Any]) -> Iterator[PromptSample]:
    split = cfg.get("split", "test")
    ds = _load_hf_dataset(
        cfg,
        "abisee/cnn_dailymail",
        "3.0.0",
        split=split,
        streaming=bool(cfg.get("streaming", True)),
    )
    ds = _maybe_shuffle(ds, cfg)
    max_new = int(cfg.get("max_new_tokens", 160))
    max_article_chars = int(cfg.get("max_article_chars", 12000))
    for index, row in enumerate(ds):
        article = str(row["article"]).strip()[:max_article_chars]
        yield PromptSample(
            sample_id=str(row.get("id", f"cnn-{split}-{index}")),
            dataset_name="cnn_dailymail",
            prompt=(
                "Summarize the following news article faithfully and concisely. "
                "Do not add facts that are absent from the article.\n\n"
                f"Article:\n{article}\n\nSummary:"
            ),
            max_new_tokens=max_new,
            reference=str(row.get("highlights", "")),
        )


def load_samples(cfg: dict[str, Any]) -> list[PromptSample]:
    name = str(cfg.get("name", "gsm8k")).lower()
    count = int(cfg.get("num_questions", cfg.get("number_questions", 10)))
    if count <= 0:
        raise ValueError("dataset.num_questions must be greater than zero")

    if name == "gsm8k":
        iterator = _gsm8k_samples(cfg)
    elif name in {"math", "hendrycks_math"}:
        iterator = _math_samples(cfg)
    elif name in {"cnn_dailymail", "cnn/dailymail", "cnn-dailymail"}:
        iterator = _cnn_dailymail_samples(cfg)
    else:
        raise ValueError("dataset.name must be one of: gsm8k, math, cnn_dailymail")
    return _take(iterator, count)


def shard_round_robin(samples: list[PromptSample], num_clients: int) -> list[deque[PromptSample]]:
    shards = [deque() for _ in range(num_clients)]
    for index, sample in enumerate(samples):
        shards[index % num_clients].append(sample)
    return shards
