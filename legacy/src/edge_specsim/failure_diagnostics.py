"""Opt-in failure diagnostics (C1.3).  Exception-only: nothing is written unless a verifier batch raises or the run aborts; no per-batch I/O.
Stores metadata only (ids, lengths, counts, memory numbers, traceback) -- never tensors, prompts or tokens."""
from __future__ import annotations

import json
import traceback
from pathlib import Path


def gpu_memory_snapshot() -> dict:
    """Free/total/allocated/reserved bytes of the current device if torch+CUDA/HIP is usable, else an explanatory stub."""
    try:
        import torch  # lazy: keeps CPU tests torch-free

        if not torch.cuda.is_available():
            return {"available": False, "reason": "cuda/hip not available"}
        free, total = torch.cuda.mem_get_info()
        return {"available": True, "device": int(torch.cuda.current_device()), "free_bytes": int(free), "total_bytes": int(total),
                "allocated_bytes": int(torch.cuda.memory_allocated()), "reserved_bytes": int(torch.cuda.memory_reserved()),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated())}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def build_batch_failure_report(batch, meta, exc: BaseException, *, last_ok: dict | None, queue_state: dict,
                               target_settings: dict | None = None, memory_fn=gpu_memory_snapshot) -> dict:
    ctx = [len(r.context_token_ids) for r in batch]
    cand = [len(r.candidate_token_ids) for r in batch]
    return {
        "kind": "verifier_batch_failure",
        "failed_batch_id": meta.batch_id,
        "member_count": len(batch),
        "members": [[r.client_id, getattr(r, "request_id", 0)] for r in batch],
        "context_lengths": ctx,
        "context_tokens_sum": sum(ctx),
        "context_tokens_max": max(ctx, default=0),
        "gamma_per_request": cand,
        "total_processed_tokens": sum(ctx) + sum(cand),
        "target_settings": dict(target_settings or {}),
        "virtual_batch_ready_ms": meta.batch_ready_ms,
        "virtual_batch_start_ms": meta.batch_start_ms,
        "queue_state": queue_state,
        "last_completed_batch": last_ok,
        "gpu_memory": memory_fn(),
        "exception": f"{type(exc).__name__}: {exc}",
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-6000:],
    }


def write_json(path: Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
