"""Download the draft/target model weights chosen in docs/model_choices.md
(TASKS.md T0.4) into a local cache under the project directory.

Not part of sim/ (sim/ needs no GPU or model weights) -- this is a one-off
data-prep script for T4.1/T4.2's real trace collection.
"""

from __future__ import annotations

import sys
import time

from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

CACHE_DIR = "/software/projects/pawsey1257/quanla/aucspec/models"


def main() -> None:
    for repo_id in MODELS:
        print(f"=== downloading {repo_id} ===", flush=True)
        start = time.time()
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=CACHE_DIR,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"],
        )
        elapsed = time.time() - start
        print(f"=== done {repo_id} -> {path} ({elapsed:.0f}s) ===", flush=True)


if __name__ == "__main__":
    main()
    print("ALL_DOWNLOADS_DONE")
