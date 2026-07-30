#!/usr/bin/env python3
"""Download tokenizer-only files from Hugging Face Hub.

This intentionally downloads only tokenizer/config metadata and excludes model
weights so local directories can be loaded with transformers.AutoTokenizer.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/QwQ-32B",
    "deepseek-ai/DeepSeek-V2.5",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V3.1-Terminus",
    "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V4-Pro",
    "MiniMaxAI/MiniMax-M2.5",
]

TOKENIZER_PATTERNS = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "config.json",
    "generation_config.json",
]

CORE_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download tokenizer files from Hugging Face Hub."
    )
    parser.add_argument(
        "--output-root",
        default="hub/tokenizer_only",
        help="Directory where tokenizer-only model folders are saved.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Hugging Face model id to download. Repeat to override the default list.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision/branch/tag.",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing a Hugging Face access token.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = args.models or DEFAULT_MODELS
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    token = os.environ.get(args.token_env) or None

    print(f"Output root: {output_root}")
    print(f"Model count: {len(models)}")
    print("Tokenizer file patterns:")
    for pattern in TOKENIZER_PATTERNS:
        print(f"  - {pattern}")

    results = []
    for model_id in models:
        local_dir = output_root / safe_model_dir_name(model_id)
        print(f"\nDownloading tokenizer files: {model_id}")
        print(f"  local_dir={local_dir}")
        try:
            downloaded_dir = download_one(
                model_id=model_id,
                local_dir=local_dir,
                revision=args.revision,
                token=token,
            )
            status = inspect_tokenizer_dir(Path(downloaded_dir))
            results.append((model_id, downloaded_dir, status, ""))
            print(f"  saved_to={downloaded_dir}")
            print(f"  files={', '.join(status['files']) if status['files'] else '(none)'}")
            print(f"  has_core_tokenizer_file={status['has_core_tokenizer_file']}")
        except Exception as exc:
            results.append((model_id, str(local_dir), {"files": [], "has_core_tokenizer_file": False}, str(exc)))
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    print("\nSummary:")
    failed = 0
    for model_id, downloaded_dir, status, error in results:
        marker = "OK" if status["has_core_tokenizer_file"] else "CHECK"
        if error:
            failed += 1
            marker = "FAILED"
        print(f"[{marker}] {model_id} -> {downloaded_dir}")
        if error:
            print(f"      {error}")
        elif not status["has_core_tokenizer_file"]:
            print("      No tokenizer.json/tokenizer.model/vocab.json found; verify model id or permissions.")
    return 1 if failed else 0


def download_one(
    model_id: str,
    local_dir: Path,
    revision: str | None,
    token: str | None,
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install it with: pip install huggingface_hub"
        ) from exc

    return snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=local_dir,
        allow_patterns=TOKENIZER_PATTERNS,
        token=token,
        max_workers=4,
    )


def inspect_tokenizer_dir(path: Path) -> dict:
    files = sorted(p.name for p in path.iterdir() if p.is_file()) if path.exists() else []
    return {
        "files": files,
        "has_core_tokenizer_file": bool(CORE_TOKENIZER_FILES & set(files)),
    }


def safe_model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "__")


if __name__ == "__main__":
    raise SystemExit(main())
