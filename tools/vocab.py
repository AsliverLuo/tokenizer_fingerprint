#!/usr/bin/env python3
"""Export an observed next-token vocabulary for one OpenAI-compatible model."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_PROBES = TOOL_DIR / "assets" / "reference_bank" / "probes_used.json"
DEFAULT_WORKSPACE = Path.cwd() / "vocab_workspace"
DEFAULT_SYSTEM_PROMPT = "Continue the text directly. Output only the continuation."
DEFAULT_REQUEST_HYPERPARAMETERS = {
    "max_tokens": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query max_tokens=1 for many probes and export the observed "
            "next-token vocabulary of one model."
        )
    )
    parser.add_argument(
        "--workspace",
        default=str(DEFAULT_WORKSPACE),
        help="Workspace for logs, selected probes, raw JSONL, and vocab outputs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Probe JSON path(s). Multiple files are merged and deduplicated by id. "
            "If omitted, the bundled BCS probes are used."
        ),
    )
    parser.add_argument(
        "--probes",
        default=None,
        help="Alias for a single probe JSON file. Ignored when --datasets is set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N loaded probes.",
    )
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Reuse an existing raw JSONL checkpoint and query only missing probes.",
    )
    parser.add_argument(
        "--parallel_num",
        type=int,
        default=5,
        help="Concurrent target-model requests.",
    )
    parser.add_argument(
        "--target_base_url",
        required=True,
        help="OpenAI-compatible API base URL, for example http://127.0.0.1:8000/v1.",
    )
    parser.add_argument(
        "--target_api_key",
        default=None,
        help="API key. Defaults to OPENAI_API_KEY or local-test-key.",
    )
    parser.add_argument(
        "--target_model",
        required=True,
        help="Model name sent to the API.",
    )
    parser.add_argument(
        "--target_temperature",
        type=float,
        default=0.0,
        help="Target sampling temperature. Defaults to 0.",
    )
    parser.add_argument(
        "--target_max_tokens",
        type=int,
        default=None,
        help="Override max_tokens for target queries. Defaults to 1.",
    )
    parser.add_argument(
        "--target_top_p",
        type=float,
        default=None,
        help="Override top_p for target queries. Defaults to 1.",
    )
    parser.add_argument(
        "--target_presence_penalty",
        type=float,
        default=None,
        help="Override presence_penalty for target queries. Defaults to 0.",
    )
    parser.add_argument(
        "--target_frequency_penalty",
        type=float,
        default=None,
        help="Override frequency_penalty for target queries. Defaults to 0.",
    )
    parser.add_argument(
        "--target_extra_body_json",
        default=None,
        help="JSON object merged into the target request body for backend-specific hyperparameters.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="Provider protocol. OpenAI-compatible is the default.",
    )
    parser.add_argument(
        "--endpoint",
        choices=["auto", "chat_completions", "completions"],
        default="auto",
        help="API endpoint mode. auto uses the existing query engine default and fallbacks.",
    )
    parser.add_argument(
        "--message_mode",
        choices=["user_prompt", "assistant_prefill", "deepseek_chat_prefix", "chat_prefix_completion"],
        default="assistant_prefill",
        help="Chat message construction mode used when endpoint is chat_completions.",
    )
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for chat-style APIs.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Per-request retry count.",
    )
    parser.add_argument(
        "--request_interval",
        type=float,
        default=0.0,
        help="Minimum interval between requests, in seconds.",
    )
    parser.add_argument(
        "--strip_leading_newlines",
        action="store_true",
        help="Normalize outputs by stripping leading CR/LF before writing raw tokens.",
    )
    parser.add_argument(
        "--raw-output",
        default=None,
        help="Optional raw JSONL output path. Defaults to workspace/raw_tokens/<model>.jsonl.",
    )
    parser.add_argument(
        "--vocab-output",
        default=None,
        help="Optional next-token frequency CSV path. Defaults to workspace/result/<model>_nexttoken_vocab.csv.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional summary JSON path. Defaults to workspace/result/summary.json.",
    )
    parser.add_argument(
        "--top-token-limit",
        type=int,
        default=100,
        help="Number of top tokens to include in summary.json.",
    )
    return parser.parse_args(argv)


def setup_workspace(workspace: Path) -> dict[str, Path]:
    paths = {
        "workspace": workspace,
        "logs": workspace / "logs",
        "intermediate": workspace / "intermediate",
        "raw_tokens": workspace / "raw_tokens",
        "result": workspace / "result",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def configure_logging(log_path: Path) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)


def write_status(status_path: Path, payload: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def safe_model_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def parse_extra_body_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--target_extra_body_json must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--target_extra_body_json must decode to a JSON object")
    return parsed


def build_request_extra_body(args: argparse.Namespace) -> dict[str, Any]:
    extra_body = parse_extra_body_json(args.target_extra_body_json)

    if args.target_max_tokens is not None:
        if args.target_max_tokens <= 0:
            raise ValueError("--target_max_tokens must be a positive integer")
        extra_body["max_tokens"] = int(args.target_max_tokens)

    if args.target_temperature is not None:
        if args.target_temperature < 0:
            raise ValueError("--target_temperature must be non-negative")
        extra_body["temperature"] = float(args.target_temperature)

    if args.target_top_p is not None:
        if not 0 <= args.target_top_p <= 1:
            raise ValueError("--target_top_p must be between 0 and 1")
        extra_body["top_p"] = float(args.target_top_p)

    if args.target_presence_penalty is not None:
        extra_body["presence_penalty"] = float(args.target_presence_penalty)

    if args.target_frequency_penalty is not None:
        extra_body["frequency_penalty"] = float(args.target_frequency_penalty)

    return extra_body


def effective_request_hyperparameters(extra_body: dict[str, Any]) -> dict[str, Any]:
    effective = dict(DEFAULT_REQUEST_HYPERPARAMETERS)
    for key in DEFAULT_REQUEST_HYPERPARAMETERS:
        if key in extra_body:
            effective[key] = extra_body[key]
    custom_extra_body = {
        key: value
        for key, value in extra_body.items()
        if key not in DEFAULT_REQUEST_HYPERPARAMETERS
    }
    return {
        "effective": effective,
        "extra_body": custom_extra_body,
        "overrides": {
            key: value
            for key, value in effective.items()
            if value != DEFAULT_REQUEST_HYPERPARAMETERS[key]
        },
    }


def log_request_hyperparameters(metadata: dict[str, Any]) -> None:
    logging.info("Target request hyperparameters: %s", metadata["effective"])
    if metadata["overrides"] or metadata["extra_body"]:
        logging.warning(
            "Non-default target request hyperparameters are enabled. "
            "Observed next-token vocabularies are comparable only when the "
            "same hyperparameters are used. overrides=%s extra_body_keys=%s",
            metadata["overrides"],
            sorted(metadata["extra_body"]),
        )


def ensure_no_proxy_for_target(base_url: str) -> None:
    host = urlparse(base_url).hostname
    if not host:
        return
    additions = [host]
    is_loopback = host in {"127.0.0.1", "localhost", "::1"}
    if is_loopback:
        additions.extend(["127.0.0.1", "localhost", "::1"])
    for env_name in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(env_name, "")
        values = [item.strip() for item in existing.split(",") if item.strip()]
        for item in additions:
            if item not in values:
                values.append(item)
        os.environ[env_name] = ",".join(values)
    if is_loopback:
        for env_name in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
            os.environ.pop(env_name, None)


def load_and_prepare_probes(args: argparse.Namespace, intermediate_dir: Path):
    from tokenizer_fingerprint.probe_generator import load_probes, save_probes

    dataset_paths: list[str] = []
    if args.datasets:
        dataset_paths = list(args.datasets)
    elif args.probes:
        dataset_paths = [args.probes]

    if dataset_paths:
        merged = []
        seen: set[str] = set()
        for raw_path in dataset_paths:
            probe_path = Path(raw_path).expanduser()
            if not probe_path.exists():
                raise FileNotFoundError(f"probe dataset not found: {probe_path}")
            for probe in load_probes(probe_path):
                if probe.id not in seen:
                    merged.append(probe)
                    seen.add(probe.id)
        probes = merged
        probe_source = ",".join(str(Path(p).expanduser()) for p in dataset_paths)
    else:
        if not DEFAULT_PROBES.exists():
            raise FileNotFoundError(f"default probe file not found: {DEFAULT_PROBES}")
        probes = load_probes(DEFAULT_PROBES)
        probe_source = str(DEFAULT_PROBES)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        probes = probes[: args.limit]
    if not probes:
        raise ValueError("no probes loaded")

    selected_path = intermediate_dir / "selected_probes.json"
    save_probes(probes, selected_path)
    return probes, probe_source, selected_path


def parse_raw_results(path: Path) -> list[Any]:
    from tokenizer_fingerprint.schema import SingleTokenResult

    if not path.exists():
        return []
    results = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(SingleTokenResult(**json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid raw token line {line_no} in {path}: {exc}") from exc
    return results


def append_valid_jsonl(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    rows: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            logging.warning("Skipping incomplete JSONL line in %s", src)
            continue
        rows.append(line)
    if not rows:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")
    return len(rows)


def load_completed_results(path: Path, allowed_probe_ids: set[str]) -> tuple[list[Any], set[str]]:
    selected = []
    completed: set[str] = set()
    for result in parse_raw_results(path):
        if result.probe_id not in allowed_probe_ids:
            continue
        if result.probe_id in completed:
            continue
        selected.append(result)
        completed.add(result.probe_id)
    return selected, completed


def write_raw_results(path: Path, results: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def write_vocab_csv(path: Path, model_name: str, raw_results: list[Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    empty_count = 0
    for result in raw_results:
        token = result.output_text
        counts[token] += 1
        if result.is_empty or token == "":
            empty_count += 1

    total = sum(counts.values())
    rows = []
    for rank, (token, frequency) in enumerate(
        sorted(counts.items(), key=lambda item: (-item[1], item[0])),
        start=1,
    ):
        rows.append(
            {
                "model_name": model_name,
                "rank": rank,
                "next_token_json": json.dumps(token, ensure_ascii=False),
                "next_token_display": token,
                "frequency": frequency,
                "share": f"{frequency / total:.8f}" if total else "0.00000000",
                "char_length": len(token),
                "byte_length": len(token.encode("utf-8")),
                "is_empty_token": int(token == ""),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "rank",
        "next_token_json",
        "next_token_display",
        "frequency",
        "share",
        "char_length",
        "byte_length",
        "is_empty_token",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "total_token_count": total,
        "unique_token_count": len(counts),
        "empty_token_count": empty_count,
        "empty_token_rate": empty_count / total if total else 0.0,
        "top_tokens": rows,
    }


async def query_missing_probes(
    probes: list[Any],
    args: argparse.Namespace,
    raw_results_path: Path,
    completed_count: int,
    total_count: int,
    status_path: Path,
) -> list[Any]:
    from tokenizer_fingerprint.query_engine import APIConfig, query_model

    api_key = args.target_api_key or os.environ.get("OPENAI_API_KEY", "local-test-key")
    endpoint = "chat_completions" if args.endpoint == "auto" else args.endpoint
    request_extra_body = build_request_extra_body(args)
    api_config = APIConfig.from_dict(
        {
            "model": args.target_model,
            "api_key": api_key,
            "base_url": args.target_base_url,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "request_interval": args.request_interval,
            "system_prompt": args.system_prompt,
            "message_mode": args.message_mode,
            "endpoint": endpoint,
            "extra_body": request_extra_body,
            "output_normalization": {
                "strip_leading_newlines": bool(args.strip_leading_newlines),
            },
        },
        provider=args.provider,
    )
    concurrency = max(1, int(args.parallel_num or 1))
    temp_raw_path = raw_results_path.with_name(
        f"{raw_results_path.stem}.part-{os.getpid()}{raw_results_path.suffix}"
    )

    last_progress = {"finished": completed_count}

    def progress_callback(chunk_finished: int, chunk_total: int) -> None:
        finished = completed_count + min(chunk_finished, chunk_total)
        if finished <= last_progress["finished"]:
            return
        last_progress["finished"] = finished
        write_status(
            status_path,
            {
                "type": "progress",
                "stage": "",
                "finished_count": finished,
                "total_count": total_count,
            },
        )
        logging.info("词表采样进度：已完成 %s / %s", finished, total_count)

    try:
        results = await query_model(
            probes=probes,
            config=api_config,
            model_name=args.target_model,
            concurrency=concurrency,
            progress_callback=progress_callback,
            raw_results_path=temp_raw_path,
        )
    except Exception:
        appended = append_valid_jsonl(temp_raw_path, raw_results_path)
        if appended:
            logging.info("Preserved %s partial raw token result(s)", appended)
        try:
            temp_raw_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    append_valid_jsonl(temp_raw_path, raw_results_path)
    try:
        temp_raw_path.unlink(missing_ok=True)
    except OSError:
        pass
    return results


def run_vocab(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    status_path = paths["logs"] / "status.log"
    safe_name = safe_model_name(args.target_model)
    raw_results_path = (
        Path(args.raw_output).expanduser()
        if args.raw_output
        else paths["raw_tokens"] / f"{safe_name}.jsonl"
    )
    vocab_output_path = (
        Path(args.vocab_output).expanduser()
        if args.vocab_output
        else paths["result"] / f"{safe_name}_nexttoken_vocab.csv"
    )
    summary_output_path = (
        Path(args.summary_output).expanduser()
        if args.summary_output
        else paths["result"] / "summary.json"
    )

    probes, probe_source, selected_probe_path = load_and_prepare_probes(args, paths["intermediate"])
    total_count = len(probes)
    write_status(status_path, {"type": "total", "total_count": total_count, "stages": []})

    ensure_no_proxy_for_target(args.target_base_url)
    request_hyperparameters = effective_request_hyperparameters(
        build_request_extra_body(args)
    )
    log_request_hyperparameters(request_hyperparameters)

    probe_ids = {probe.id for probe in probes}
    if args.continue_run:
        existing_results, completed_ids = load_completed_results(raw_results_path, probe_ids)
    else:
        existing_results, completed_ids = [], set()
        raw_results_path.parent.mkdir(parents=True, exist_ok=True)
        raw_results_path.write_text("", encoding="utf-8")

    pending_probes = [probe for probe in probes if probe.id not in completed_ids]
    if completed_ids:
        write_status(
            status_path,
            {
                "type": "progress",
                "stage": "",
                "finished_count": len(completed_ids),
                "total_count": total_count,
            },
        )
        logging.info("Resume checkpoint found: %s/%s probes already completed", len(completed_ids), total_count)

    logging.info("Starting next-token vocabulary export")
    logging.info("Probe source: %s", probe_source)
    logging.info("Selected probes: %s", selected_probe_path)
    logging.info("Target model: %s", args.target_model)
    logging.info("Target base URL: %s", args.target_base_url)
    logging.info("Raw JSONL output: %s", raw_results_path)

    start = time.monotonic()
    if pending_probes:
        new_results = asyncio.run(
            query_missing_probes(
                probes=pending_probes,
                args=args,
                raw_results_path=raw_results_path,
                completed_count=len(completed_ids),
                total_count=total_count,
                status_path=status_path,
            )
        )
    else:
        logging.info("No missing probes; using checkpointed raw token results")
        new_results = []

    all_results, _ = load_completed_results(raw_results_path, probe_ids)
    if not all_results and (existing_results or new_results):
        all_results = existing_results + new_results
        write_raw_results(raw_results_path, all_results)

    vocab_stats = write_vocab_csv(vocab_output_path, args.target_model, all_results)
    elapsed = time.monotonic() - start
    top_limit = max(0, int(args.top_token_limit))
    summary = {
        "target_model": args.target_model,
        "target_base_url": args.target_base_url,
        "probe_source": probe_source,
        "selected_probe_file": str(selected_probe_path),
        "raw_results_file": str(raw_results_path),
        "vocab_file": str(vocab_output_path),
        "total_count": total_count,
        "completed_count": len(all_results),
        "failed_count": total_count - len(all_results),
        "resume_enabled": bool(args.continue_run),
        "elapsed_seconds": elapsed,
        "request_hyperparameters": request_hyperparameters,
        "total_token_count": vocab_stats["total_token_count"],
        "unique_token_count": vocab_stats["unique_token_count"],
        "empty_token_count": vocab_stats["empty_token_count"],
        "empty_token_rate": vocab_stats["empty_token_rate"],
        "top_tokens": vocab_stats["top_tokens"][:top_limit],
    }
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (paths["workspace"] / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if len(all_results) < total_count:
        raise ValueError(f"only {len(all_results)}/{total_count} probes completed")

    write_status(
        status_path,
        {
            "type": "progress",
            "stage": "",
            "finished_count": total_count,
            "total_count": total_count,
        },
    )
    write_status(status_path, {"type": "completed", "total_count": total_count, "stages": []})
    logging.info("Next-token vocabulary export completed in %.2fs", elapsed)
    logging.info("Vocab CSV written to %s", vocab_output_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = Path(args.workspace).expanduser().resolve()
    paths = setup_workspace(workspace)
    configure_logging(paths["logs"] / "run.log")
    status_path = paths["logs"] / "status.log"

    try:
        summary = run_vocab(args, paths)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "raw_results_file": summary["raw_results_file"],
                    "vocab_file": summary["vocab_file"],
                    "unique_token_count": summary["unique_token_count"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:
        logging.error("Next-token vocabulary export failed: %s", exc)
        write_status(
            status_path,
            {
                "type": "failed",
                "total_count": 0,
                "stages": [],
                "error": str(exc),
            },
        )
        error_payload = {"status": "failed", "error": str(exc)}
        (paths["result"] / "summary.json").write_text(
            json.dumps(error_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(error_payload, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
