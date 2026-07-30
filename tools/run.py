#!/usr/bin/env python3
"""Platform entry point for strict BCS tokenizer fingerprint detection."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import signal
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE = TOOL_DIR / "assets" / "reference_bank"
DEFAULT_WORKSPACE = Path.cwd() / "workspace"
DEFAULT_REQUEST_HYPERPARAMETERS = {
    "max_tokens": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict BCS tokenizer fingerprint detection against an OpenAI-compatible target model."
    )
    parser.add_argument("--mode", choices=["detect", "vocab"], default="detect", help="Run BCS detection or next-token simulated vocabulary extraction.")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="Workspace for logs, intermediate files, and results.")
    parser.add_argument("--datasets", nargs="+", default=None, help="Probe JSON path(s). Multiple files are merged by probe id.")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N probes after merging/loading.")
    parser.add_argument("--continue", dest="continue_run", action="store_true", help="Resume from existing raw token checkpoint in the workspace.")
    parser.add_argument("--parallel_num", type=int, default=5, help="Concurrent requests to the target model.")
    parser.add_argument("--target_base_url", default=None, help="OpenAI-compatible API base URL, for example http://127.0.0.1:8000/v1.")
    parser.add_argument("--target_api_key", default=None, help="API key for the target model. Local servers may accept any non-empty value.")
    parser.add_argument("--target_model", default=None, help="Target model name sent to the OpenAI-compatible API.")
    parser.add_argument("--target_temperature", type=float, default=0.0, help="Target sampling temperature. Defaults to 0 for deterministic BCS/next-token queries.")
    parser.add_argument("--target_max_tokens", type=int, default=None, help="Override max_tokens for target queries. Defaults to 1.")
    parser.add_argument("--target_top_p", type=float, default=None, help="Override top_p for target queries. Defaults to 1.")
    parser.add_argument("--target_presence_penalty", type=float, default=None, help="Override presence_penalty for target queries. Defaults to 0.")
    parser.add_argument("--target_frequency_penalty", type=float, default=None, help="Override frequency_penalty for target queries. Defaults to 0.")
    parser.add_argument("--target_extra_body_json", default=None, help="JSON object merged into the target request body, for backend-specific hyperparameters such as seed, stop, or enable_thinking.")
    parser.add_argument("--attack_base_url", default=None, help="Accepted for platform compatibility; not used by this BCS tool.")
    parser.add_argument("--attack_api_key", default=None, help="Accepted for platform compatibility; not used by this BCS tool.")
    parser.add_argument("--attack_model", default=None, help="Accepted for platform compatibility; not used by this BCS tool.")
    parser.add_argument("--attack_temperature", type=float, default=None, help="Accepted for platform compatibility; not used by this BCS tool.")
    parser.add_argument("--eval_base_url", default=None, help="Accepted for platform compatibility; not used by this single-stage BCS tool.")
    parser.add_argument("--eval_api_key", default=None, help="Accepted for platform compatibility; not used by this single-stage BCS tool.")
    parser.add_argument("--eval_model", default=None, help="Accepted for platform compatibility; not used by this single-stage BCS tool.")
    parser.add_argument("--eval_temperature", type=float, default=None, help="Accepted for platform compatibility; not used by this single-stage BCS tool.")
    parser.add_argument("--eval_scope", choices=["test", "eval", "full"], default="full", help="Single-stage BCS supports test/full. eval-only is not applicable.")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="Reference bank directory. Defaults to bundled assets/reference_bank.")
    parser.add_argument("--auto_save_batch_size", type=int, default=None, help="Flush raw query checkpoints every N completed query tasks. Use 1 for per-task saving.")
    return parser.parse_args(argv)


def setup_workspace(workspace: Path) -> dict[str, Path]:
    paths = {
        "workspace": workspace,
        "logs": workspace / "logs",
        "intermediate": workspace / "intermediate",
        "result": workspace / "result",
        "status": workspace / "status.log",
        "query_results": workspace / "intermediate" / "query_results.jsonl",
        "evaluation_results": workspace / "result" / "evaluation_results.jsonl",
        "timing": workspace / "result" / "timing.json",
    }
    for key, path in paths.items():
        if key in {"status", "query_results", "evaluation_results", "timing"}:
            continue
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


def status_output_paths(status_path: Path) -> list[Path]:
    paths = [status_path]
    if status_path.name == "status.log" and status_path.parent.name == "logs":
        paths.append(status_path.parent.parent / "status.log")
    return list(dict.fromkeys(paths))


def write_status(status_path: Path, payload: dict[str, Any], initialize: bool = False) -> None:
    # Each execution starts with a fresh total line, then appends progress.
    mode = "w" if initialize else "a"
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    for output_path in status_output_paths(status_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open(mode, encoding="utf-8") as f:
            f.write(line)


def read_declared_total(status_path: Path) -> int:
    if not status_path.exists():
        return 0
    try:
        total_count = 0
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                if payload.get("type") == "total":
                    total_count = int(payload.get("total_count", 0))
        return total_count
    except Exception:
        return 0
    return 0


def compute_query_total_count(
    probes: list[Any],
    stability_ids: set[str],
    stability_repeats: int,
) -> int:
    return len(probes) + len(stability_ids) * max(0, stability_repeats - 1)


def required_counts_by_probe(
    probes: list[Any],
    stability_ids: set[str],
    stability_repeats: int,
) -> dict[str, int]:
    return {
        probe.id: 1 + (max(0, stability_repeats - 1) if probe.id in stability_ids else 0)
        for probe in probes
    }


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
                row = json.loads(line)
                results.append(SingleTokenResult(**row))
            except Exception as exc:
                raise ValueError(f"invalid raw token checkpoint line {line_no} in {path}: {exc}") from exc
    return results


def load_checkpoint_results(
    path: Path,
    required_counts: dict[str, int],
) -> tuple[list[Any], dict[str, int]]:
    counts = {probe_id: 0 for probe_id in required_counts}
    selected = []
    for result in parse_raw_results(path):
        required = required_counts.get(result.probe_id)
        if required is None:
            continue
        if counts[result.probe_id] >= required:
            continue
        selected.append(result)
        counts[result.probe_id] += 1
    return selected, counts


def append_jsonl_file(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    lines = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            logging.warning("Skipping incomplete JSONL checkpoint line in %s", src)
            continue
        lines.append(line)
    if not lines:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return len(lines)


def merge_part_checkpoints(raw_results_path: Path) -> int:
    raw_results_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    for part_path in sorted(
        raw_results_path.parent.glob(f"{raw_results_path.stem}.part-*{raw_results_path.suffix}")
    ):
        total += append_jsonl_file(part_path, raw_results_path)
        try:
            part_path.unlink(missing_ok=True)
        except OSError as exc:
            logging.warning("Could not remove temporary checkpoint %s: %s", part_path, exc)
    return total


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def format_stage_timings(stage_timings: dict[str, float]) -> str:
    if not stage_timings:
        return "none"
    return ", ".join(f"{name}={seconds:.2f}s" for name, seconds in stage_timings.items())


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


def log_request_hyperparameters(metadata: dict[str, Any], mode: str) -> None:
    effective = metadata["effective"]
    overrides = metadata["overrides"]
    extra_body = metadata["extra_body"]
    logging.info("Target request hyperparameters for %s: %s", mode, effective)
    if overrides or extra_body:
        logging.warning(
            "Non-default target request hyperparameters are enabled. "
            "BCS/reference-bank scores are comparable only when the same "
            "hyperparameters were used to build the reference bank. "
            "overrides=%s extra_body_keys=%s",
            overrides,
            sorted(extra_body),
        )


def enum_or_str_value(value: Any, default: str = "other") -> str:
    if value is None:
        return default
    if hasattr(value, "value"):
        return str(value.value)
    text = str(value)
    return text if text else default


def is_failed_token_result(result: Any) -> bool:
    raw_response = getattr(result, "raw_response", {})
    return isinstance(raw_response, dict) and "error" in raw_response


def extract_transition_features_compat(results: list[Any], probes: list[Any]):
    from tokenizer_fingerprint.schema import TransitionFeatures

    probe_by_id = {probe.id: probe for probe in probes}
    category_counts: dict[str, Counter[str]] = {}

    for result in results:
        probe = probe_by_id.get(getattr(result, "probe_id", None))
        if probe is None:
            continue
        category = enum_or_str_value(getattr(probe, "category", ""), default="unknown")
        token_type = enum_or_str_value(getattr(result, "token_type", ""), default="other")
        transition_key = f"{category}→{token_type}"
        if category not in category_counts:
            category_counts[category] = Counter()
        category_counts[category][transition_key] += 1

    transition_matrix: dict[str, float] = {}
    for category in sorted(category_counts):
        total = sum(category_counts[category].values())
        if total <= 0:
            continue
        for transition_key, count in sorted(category_counts[category].items()):
            transition_matrix[transition_key] = count / total

    return TransitionFeatures(transition_matrix=transition_matrix)


def extract_fingerprint_compat(
    *,
    model_name: str,
    family: str,
    results: list[Any],
    probes: list[Any],
    primary_extractor,
):
    """Work around a packaged .so bug that passes defaultdict into TransitionFeatures."""
    try:
        return primary_extractor(
            model_name=model_name,
            family=family,
            results=results,
            probes=probes,
        )
    except TypeError as exc:
        if "collections.defaultdict" not in str(exc):
            raise
        logging.info(
            "Using Python compatibility extractor for packaged extract_fingerprint issue: %s",
            exc,
        )

    from tokenizer_fingerprint.feature_extractor import extract_surface_features, extract_type_features
    from tokenizer_fingerprint.schema import ModelFingerprint, SurfaceFeatures, TypeFeatures

    usable_results = [result for result in results if not is_failed_token_result(result)]
    failed_queries = len(results) - len(usable_results)

    if usable_results:
        surface = extract_surface_features(usable_results)
        type_feat = extract_type_features(usable_results)
    else:
        surface = SurfaceFeatures()
        type_feat = TypeFeatures()

    transition = extract_transition_features_compat(usable_results, probes)
    metadata = {
        "total_queries": len(results),
        "successful_queries": len(usable_results),
        "failed_queries": failed_queries,
        "failure_rate": failed_queries / len(results) if results else 0.0,
        "extractor": "python_compat_defaultdict_fix",
    }

    return ModelFingerprint(
        model_name=model_name,
        family=family,
        surface=surface,
        type_feat=type_feat,
        transition=transition,
        n_probes=len(usable_results),
        raw_results=usable_results,
        metadata=metadata,
    )


def compute_stability_variance_compat(results: list[Any]) -> float:
    outputs_by_probe: dict[str, list[str]] = {}
    for result in results:
        if is_failed_token_result(result):
            continue
        probe_id = str(getattr(result, "probe_id", ""))
        if not probe_id:
            continue
        outputs_by_probe.setdefault(probe_id, []).append(str(getattr(result, "output_text", "")))

    variances = []
    for outputs in outputs_by_probe.values():
        if len(outputs) <= 1:
            continue
        counts = Counter(outputs)
        variances.append(1.0 - max(counts.values()) / len(outputs))

    return sum(variances) / len(variances) if variances else 0.0


def patch_similarity_defaultdict_bug(similarity_module: Any) -> None:
    if getattr(similarity_module, "_tkfp_defaultdict_patch_applied", False):
        return
    similarity_module.compute_stability_variance = compute_stability_variance_compat
    similarity_module._tkfp_defaultdict_patch_applied = True
    logging.debug("Patched tokenizer_fingerprint.similarity.compute_stability_variance")


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_valid_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping incomplete JSONL line %s in %s", line_no, path)
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                logging.warning("Skipping non-object JSONL line %s in %s", line_no, path)
    return rows


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sync_query_results_file(raw_results_path: Path, query_results_path: Path) -> int:
    if raw_results_path == query_results_path:
        return len(read_valid_jsonl_rows(raw_results_path))
    rows = read_valid_jsonl_rows(raw_results_path)
    write_jsonl_rows(query_results_path, rows)
    return len(rows)


def summarize_raw_results(raw_results_path: Path) -> dict[str, Any]:
    rows = read_valid_jsonl_rows(raw_results_path)
    total_rows = len(rows)
    failed_rows = 0
    empty_rows = 0
    usable_rows = 0
    usable_probe_ids: set[str] = set()
    failed_probe_ids: set[str] = set()

    for row in rows:
        probe_id = str(row.get("probe_id", ""))
        raw_response = row.get("raw_response", {})
        failed = isinstance(raw_response, dict) and "error" in raw_response
        if failed:
            failed_rows += 1
            if probe_id:
                failed_probe_ids.add(probe_id)
            continue
        usable_rows += 1
        if probe_id:
            usable_probe_ids.add(probe_id)
        if row.get("output_text", "") == "":
            empty_rows += 1

    return {
        "total_rows": total_rows,
        "usable_rows": usable_rows,
        "failed_rows": failed_rows,
        "empty_rows": empty_rows,
        "unique_usable_probe_count": len(usable_probe_ids),
        "unique_failed_probe_count": len(failed_probe_ids - usable_probe_ids),
    }


def write_evaluation_results_jsonl(
    *,
    mode: str,
    target_model: str,
    probes: list[Any],
    raw_results_path: Path,
    output_path: Path,
    result_payload: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, int]:
    probe_by_id = {probe.id: probe for probe in probes}
    rows = []
    pass_count = 0
    fail_count = 0

    label = (result_payload or {}).get("label")
    confidence = (result_payload or {}).get("confidence")
    diagnosis = (result_payload or {}).get("diagnosis", "")

    for raw_row in read_valid_jsonl_rows(raw_results_path):
        probe_id = str(raw_row.get("probe_id", ""))
        probe = probe_by_id.get(probe_id)
        raw_response = raw_row.get("raw_response", {})
        failed = isinstance(raw_response, dict) and "error" in raw_response
        output_text = raw_row.get("output_text", "")
        error_text = ""
        if failed:
            error_text = str(raw_response.get("error", "query failed"))
            fail_count += 1
        else:
            pass_count += 1

        if mode == "detect":
            eval_response = (
                f"strict BCS model-level decision: {label}; confidence={confidence}"
                if label is not None
                else "strict BCS query result captured"
            )
            if diagnosis:
                eval_response = f"{eval_response}; diagnosis={diagnosis}"
        else:
            unique_tokens = (summary or {}).get("unique_next_tokens")
            eval_response = (
                f"next-token vocabulary extraction completed; unique_next_tokens={unique_tokens}"
                if unique_tokens is not None
                else "next-token query result captured"
            )

        row = {
            "seed": probe.text if probe else "",
            "prompt": probe.text if probe else "",
            "response": output_text,
            "eval_response": error_text if failed else eval_response,
            "eval_result": -1 if failed else 1,
            "remark": error_text if failed else "",
            "probe_id": probe_id,
            "model_name": raw_row.get("model_name", target_model),
            "mode": mode,
            "category": getattr(probe, "category", ""),
            "token_type": raw_row.get("token_type"),
            "char_length": raw_row.get("char_length"),
            "byte_length": raw_row.get("byte_length"),
            "is_empty": raw_row.get("is_empty"),
            "latency_ms": raw_row.get("latency_ms"),
        }
        if label is not None:
            row["bcs_label"] = label
            row["bcs_confidence"] = confidence
        rows.append(row)

    write_jsonl_rows(output_path, rows)
    return {
        "evaluation_rows": len(rows),
        "evaluation_pass_count": pass_count,
        "evaluation_fail_count": fail_count,
    }


def write_summary_files(paths: dict[str, Path], summary: dict[str, Any]) -> None:
    write_json_file(paths["result"] / "summary.json", summary)
    write_json_file(paths["workspace"] / "summary.json", summary)


def write_timing_files(
    paths: dict[str, Path],
    *,
    mode: str,
    status: str,
    elapsed_seconds: float,
    stage_timings: dict[str, float],
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": mode,
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "stage_elapsed_seconds": stage_timings,
    }
    if error:
        payload["error"] = error
    write_json_file(paths["timing"], payload)
    write_json_file(paths["workspace"] / "timing.json", payload)
    return payload


def write_failure_artifacts(
    paths: dict[str, Path],
    run_state: dict[str, Any],
    error: str,
) -> dict[str, Any]:
    mode = str(run_state.get("mode") or "unknown")
    target_model = run_state.get("target_model")
    stage_timings = run_state.get("stage_timings") or {}
    started_at = run_state.get("started_at")
    elapsed = time.monotonic() - started_at if started_at else 0.0

    raw_results_path = run_state.get("raw_results_path")
    if raw_results_path:
        raw_results_path = Path(raw_results_path)
        if raw_results_path.exists():
            sync_query_results_file(raw_results_path, paths["query_results"])
        else:
            write_jsonl_rows(paths["query_results"], [])
    else:
        raw_results_path = None
        write_jsonl_rows(paths["query_results"], [])

    probes = run_state.get("probes") or []
    if raw_results_path and raw_results_path.exists() and probes:
        evaluation_stats = write_evaluation_results_jsonl(
            mode=mode,
            target_model=str(target_model or ""),
            probes=probes,
            raw_results_path=raw_results_path,
            output_path=paths["evaluation_results"],
            result_payload={"label": "failed", "confidence": 0.0, "diagnosis": error},
        )
        raw_summary = summarize_raw_results(raw_results_path)
    else:
        write_jsonl_rows(paths["evaluation_results"], [])
        evaluation_stats = {
            "evaluation_rows": 0,
            "evaluation_pass_count": 0,
            "evaluation_fail_count": 0,
        }
        raw_summary = {
            "total_rows": 0,
            "usable_rows": 0,
            "failed_rows": 0,
            "empty_rows": 0,
            "unique_usable_probe_count": 0,
            "unique_failed_probe_count": 0,
        }

    timing_payload = write_timing_files(
        paths,
        mode=mode,
        status="failed",
        elapsed_seconds=elapsed,
        stage_timings=stage_timings,
        error=error,
    )

    total_count = int(run_state.get("total_count") or 0)
    finished_count = int(run_state.get("finished_count") or raw_summary["usable_rows"])
    failed_count = max(0, total_count - finished_count) if total_count else raw_summary["failed_rows"]
    summary = {
        "status": "failed",
        "mode": mode,
        "total_count": total_count,
        "pass_count": finished_count,
        "fail_count": failed_count,
        "na_count": 0,
        "completed_count": finished_count,
        "failed_count": failed_count,
        "error": error,
        "target_model": target_model,
        "elapsed_seconds": elapsed,
        "stage_elapsed_seconds": stage_timings,
        "query_rows": raw_summary["total_rows"],
        "query_pass_count": raw_summary["usable_rows"],
        "query_fail_count": raw_summary["failed_rows"],
        "empty_output_count": raw_summary["empty_rows"],
        "query_results_file": str(paths["query_results"]),
        "evaluation_results_file": str(paths["evaluation_results"]),
        "timing_file": str(paths["timing"]),
        "timing": timing_payload,
        **evaluation_stats,
    }
    write_summary_files(paths, summary)

    result_payload = {
        "status": "failed",
        "mode": mode,
        "target_model": target_model,
        "workspace": str(paths["workspace"]),
        "error": error,
        "total_count": summary["total_count"],
        "pass_count": summary["pass_count"],
        "fail_count": summary["fail_count"],
        "na_count": summary["na_count"],
        "completed_count": summary["completed_count"],
        "failed_count": summary["failed_count"],
        "elapsed_seconds": elapsed,
        "stage_elapsed_seconds": stage_timings,
        "summary": summary,
    }
    if raw_results_path:
        result_payload["raw_results_file"] = str(raw_results_path)
    result_payload["query_results_file"] = str(paths["query_results"])
    result_payload["evaluation_results_file"] = str(paths["evaluation_results"])
    result_payload["timing_file"] = str(paths["timing"])
    write_json_file(paths["result"] / "result.json", result_payload)
    return result_payload


def collect_next_token_counts(raw_results_path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    total_rows = 0
    usable_rows = 0
    failed_rows = 0
    empty_rows = 0
    model_names: set[str] = set()

    with raw_results_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            row = json.loads(line)
            model_name = row.get("model_name")
            if model_name:
                model_names.add(str(model_name))
            raw_response = row.get("raw_response", {})
            if isinstance(raw_response, dict) and "error" in raw_response:
                failed_rows += 1
                continue
            token = row.get("output_text", "")
            if not isinstance(token, str):
                raise ValueError(
                    f"invalid output_text at {raw_results_path}:{line_no}: "
                    f"expected str, got {type(token).__name__}"
                )
            counts[token] += 1
            usable_rows += 1
            if token == "":
                empty_rows += 1

    return {
        "counts": counts,
        "total_rows": total_rows,
        "usable_rows": usable_rows,
        "failed_rows": failed_rows,
        "empty_rows": empty_rows,
        "model_names": sorted(model_names),
    }


def write_simulated_vocab_csv(model_name: str, counts: Counter[str], output_path: Path) -> None:
    total = sum(counts.values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_name",
                "next_token_json",
                "frequency",
                "share",
                "rank",
                "char_length",
                "byte_length",
            ]
        )
        for rank, (token, frequency) in enumerate(
            sorted(counts.items(), key=lambda item: (-item[1], item[0])),
            start=1,
        ):
            writer.writerow(
                [
                    model_name,
                    json.dumps(token, ensure_ascii=False),
                    frequency,
                    f"{frequency / total:.8f}" if total else "0.00000000",
                    rank,
                    len(token),
                    len(token.encode("utf-8")),
                ]
            )


def write_simulated_vocab_json(
    model_name: str,
    counts: Counter[str],
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    total = sum(counts.values())
    tokens = []
    for rank, (token, frequency) in enumerate(
        sorted(counts.items(), key=lambda item: (-item[1], item[0])),
        start=1,
    ):
        tokens.append(
            {
                "token": token,
                "frequency": frequency,
                "share": frequency / total if total else 0.0,
                "rank": rank,
                "char_length": len(token),
                "byte_length": len(token.encode("utf-8")),
            }
        )
    payload = {
        "model_name": model_name,
        "total_next_tokens": total,
        "unique_next_tokens": len(counts),
        "metadata": metadata,
        "tokens": tokens,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_failed_status_once(
    run_state: dict[str, Any],
    status_path: Path,
    error: str,
) -> None:
    if run_state.get("terminal_written"):
        return
    if not run_state.get("total_written"):
        write_status(status_path, {"type": "total", "total_count": 0, "stages": []}, initialize=True)
        run_state["total_written"] = True
        run_state["total_count"] = 0
    total_count = int(run_state.get("total_count") or read_declared_total(status_path))
    write_status(
        status_path,
        {"type": "failed", "total_count": total_count, "stages": [], "error": error},
    )
    run_state["terminal_written"] = True


def preserve_current_checkpoint(run_state: dict[str, Any]) -> int:
    raw_path = run_state.get("raw_results_path")
    temp_path = run_state.get("current_temp_raw_path")
    if not raw_path or not temp_path:
        return 0

    raw_path = Path(raw_path)
    temp_path = Path(temp_path)
    appended = append_jsonl_file(temp_path, raw_path)
    try:
        temp_path.unlink(missing_ok=True)
    except OSError as exc:
        logging.warning("Could not remove temporary checkpoint %s: %s", temp_path, exc)
    run_state["current_temp_raw_path"] = None
    query_results_path = run_state.get("query_results_path")
    if appended and query_results_path:
        sync_query_results_file(raw_path, Path(query_results_path))
    return appended


def install_signal_handlers(run_state: dict[str, Any], status_path: Path, paths: dict[str, Path]) -> None:
    def handle_signal(signum, frame):
        signame = signal.Signals(signum).name
        run_state["terminating"] = True
        logging.warning("Received %s; preserving checkpoint and exiting", signame)
        appended = preserve_current_checkpoint(run_state)
        if appended:
            logging.info("Preserved %s partial raw result(s) before exit", appended)
        write_failed_status_once(
            run_state,
            status_path,
            f"terminated by signal {signame}",
        )
        write_failure_artifacts(paths, run_state, f"terminated by signal {signame}")
        logging.info(
            "测试总耗时：%.2fs",
            time.monotonic() - run_state.get("started_at", time.monotonic()),
        )
        logging.info(
            "阶段耗时汇总：%s",
            format_stage_timings(run_state.get("stage_timings") or {}),
        )
        logging.shutdown()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)


def save_config(workspace: Path, args: argparse.Namespace) -> None:
    config = vars(args).copy()
    for key in ("target_api_key", "attack_api_key", "eval_api_key"):
        if config.get(key):
            config[key] = "***"
    (workspace / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def safe_model_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def load_and_prepare_probes(args: argparse.Namespace, intermediate_dir: Path):
    from tokenizer_fingerprint.probe_generator import load_probes, save_probes

    if args.datasets:
        merged = []
        seen: set[str] = set()
        for raw_path in args.datasets:
            probe_path = Path(raw_path)
            if not probe_path.is_absolute():
                raise ValueError(f"--datasets path must be absolute: {probe_path}")
            if not probe_path.exists():
                raise FileNotFoundError(f"probe dataset not found: {probe_path}")
            for probe in load_probes(probe_path):
                if probe.id not in seen:
                    merged.append(probe)
                    seen.add(probe.id)

        merged_path = intermediate_dir / "merged_probes.json"
        save_probes(merged, merged_path)
        probes = merged
        probe_source = str(merged_path)
    else:
        reference_dir = Path(args.reference)
        default_probe_path = reference_dir / "probes_used.json"
        if not default_probe_path.exists():
            default_probe_path = DEFAULT_REFERENCE / "probes_used.json"
        if not default_probe_path.exists():
            raise FileNotFoundError(f"default probe file not found: {default_probe_path}")
        probes = load_probes(default_probe_path)
        probe_source = str(default_probe_path)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        probes = probes[: args.limit]

    if not probes:
        raise ValueError("no probes loaded")

    selected_path = intermediate_dir / "selected_probes.json"
    save_probes(probes, selected_path)
    return probes, probe_source, selected_path


def validate_target_args(args: argparse.Namespace) -> None:
    if args.eval_scope == "eval":
        raise ValueError("--eval_scope eval is not supported; BCS has no separate eval-only stage")
    missing = []
    if not args.target_base_url:
        missing.append("--target_base_url")
    if not args.target_model:
        missing.append("--target_model")
    if missing:
        raise ValueError(f"missing required target parameter(s): {', '.join(missing)}")


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


def run_detection(args: argparse.Namespace, paths: dict[str, Path], run_state: dict[str, Any]):
    from tokenizer_fingerprint.feature_extractor import extract_fingerprint
    from tokenizer_fingerprint.query_engine import APIConfig, query_model
    from tokenizer_fingerprint.reference_bank import ReferenceBank
    from tokenizer_fingerprint.schema import DetectionResult
    import tokenizer_fingerprint.similarity as similarity_module

    patch_similarity_defaultdict_bug(similarity_module)

    run_state["mode"] = "detect"
    reference_dir = Path(args.reference)
    if not reference_dir.exists():
        raise FileNotFoundError(f"reference bank not found: {reference_dir}")

    probes, probe_source, selected_probe_path = load_and_prepare_probes(args, paths["intermediate"])
    run_state["probes"] = probes
    run_state["target_model"] = args.target_model
    status_path = paths["logs"] / "status.log"
    stability_ratio = 0.1
    stability_repeats = 3
    rng = random.Random(42)
    stability_count = max(1, int(len(probes) * stability_ratio))
    stability_ids = set(
        p.id for p in rng.sample(probes, min(stability_count, len(probes)))
    )
    status_total_count = compute_query_total_count(
        probes,
        stability_ids,
        stability_repeats,
    )
    required_counts = required_counts_by_probe(
        probes,
        stability_ids,
        stability_repeats,
    )
    write_status(
        status_path,
        {"type": "total", "total_count": status_total_count, "stages": []},
        initialize=True,
    )
    run_state["total_written"] = True
    run_state["total_count"] = status_total_count

    validate_target_args(args)
    ensure_no_proxy_for_target(args.target_base_url)

    bank = ReferenceBank.load(reference_dir)
    if not bank.list_models():
        raise ValueError(f"reference bank is empty or invalid: {reference_dir}")

    api_key = args.target_api_key
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "local-test-key")

    request_extra_body = build_request_extra_body(args)
    request_hyperparameters = effective_request_hyperparameters(request_extra_body)
    api_config = APIConfig.from_dict(
        {
            "model": args.target_model,
            "api_key": api_key,
            "base_url": args.target_base_url,
            "extra_body": request_extra_body,
        },
        provider="openai",
    )

    concurrency = max(1, int(args.parallel_num or 1))
    raw_results_path = paths["intermediate"] / "raw_tokens" / f"{safe_model_name(args.target_model)}.jsonl"
    run_state["raw_results_path"] = raw_results_path
    run_state["query_results_path"] = paths["query_results"]

    logging.info("Starting strict BCS detection")
    logging.info("Reference bank: %s", reference_dir)
    logging.info("Probe source: %s", probe_source)
    logging.info("Selected probes: %s", selected_probe_path)
    logging.info("Target model: %s", args.target_model)
    logging.info("Target base URL: %s", args.target_base_url)
    log_request_hyperparameters(request_hyperparameters, "detect")
    unused_compat_args = [
        name for name in (
            "attack_base_url",
            "attack_api_key",
            "attack_model",
            "attack_temperature",
            "eval_base_url",
            "eval_api_key",
            "eval_model",
            "eval_temperature",
        )
        if getattr(args, name, None) not in (None, "")
    ]
    if unused_compat_args:
        logging.info(
            "Ignoring compatibility-only parameter(s) not used by BCS: %s",
            ", ".join(unused_compat_args),
        )
    if args.eval_scope == "test":
        logging.info("--eval_scope test selected; running the single BCS target-query workflow")
    if args.continue_run:
        logging.info("--continue was provided; loading raw token checkpoint if present")
    if args.auto_save_batch_size is not None:
        if args.auto_save_batch_size <= 0:
            raise ValueError("--auto_save_batch_size must be a positive integer")
        logging.info(
            "Raw query results will be flushed every %s completed query task(s)",
            args.auto_save_batch_size,
        )

    if args.continue_run:
        merged_parts = merge_part_checkpoints(raw_results_path)
        if merged_parts:
            logging.info(
                "Merged %s raw result(s) from temporary checkpoints before resume",
                merged_parts,
            )
        existing_results, existing_counts = load_checkpoint_results(
            raw_results_path,
            required_counts,
        )
    else:
        existing_results = []
        existing_counts = {probe_id: 0 for probe_id in required_counts}
        raw_results_path.parent.mkdir(parents=True, exist_ok=True)
        raw_results_path.write_text("", encoding="utf-8")
        sync_query_results_file(raw_results_path, paths["query_results"])

    existing_completed = sum(
        min(existing_counts.get(probe.id, 0), required_counts[probe.id])
        for probe in probes
    )

    overall_start = time.monotonic()
    stage_timings: dict[str, float] = {}
    run_state["stage_timings"] = stage_timings
    last_progress = {"finished": -1}

    def status_progress(finished_count: int) -> None:
        finished = min(int(finished_count), status_total_count)
        if finished <= last_progress["finished"]:
            return
        last_progress["finished"] = finished
        run_state["finished_count"] = finished
        write_status(
            status_path,
            {
                "type": "progress",
                "stage": "",
                "finished_count": finished,
                "total_count": status_total_count,
            },
        )
        logging.info("测试进度：已完成 %s / %s", finished, status_total_count)

    logging.info(
        "Probes: %s unique, %s query tasks including %s stability probe(s)",
        len(probes),
        status_total_count,
        len(stability_ids),
    )
    if existing_completed:
        logging.info(
            "Resume checkpoint found: %s/%s query tasks already completed",
            existing_completed,
            status_total_count,
        )
        status_progress(existing_completed)

    pending_units: list[Any] = []
    for probe in probes:
        missing = required_counts[probe.id] - existing_counts.get(probe.id, 0)
        for _ in range(max(0, missing)):
            pending_units.append(probe)

    results = list(existing_results)
    completed_so_far = existing_completed
    save_batch_size = args.auto_save_batch_size or len(pending_units) or 1
    pending_chunks = chunked(pending_units, save_batch_size)
    query_start = time.monotonic()

    for chunk_index, chunk_probes in enumerate(
        pending_chunks,
        1,
    ):
        chunk_total = len(chunk_probes)
        temp_raw_path = raw_results_path.with_name(
            f"{raw_results_path.stem}.part-{os.getpid()}-{chunk_index}{raw_results_path.suffix}"
        )
        run_state["current_temp_raw_path"] = temp_raw_path

        def chunk_progress(chunk_finished: int, chunk_reported_total: int) -> None:
            status_progress(completed_so_far + min(chunk_finished, chunk_reported_total))

        logging.info(
            "Querying %s missing task(s); checkpoint flush batch size=%s",
            chunk_total,
            save_batch_size,
        )
        try:
            chunk_results = asyncio.run(
                query_model(
                    probes=chunk_probes,
                    config=api_config,
                    model_name=args.target_model,
                    concurrency=concurrency,
                    progress_callback=chunk_progress,
                    raw_results_path=temp_raw_path,
                )
            )
        except Exception:
            appended = append_jsonl_file(temp_raw_path, raw_results_path)
            if appended:
                sync_query_results_file(raw_results_path, paths["query_results"])
                logging.info(
                    "Preserved %s partial raw result(s) in %s",
                    appended,
                    raw_results_path,
                )
            try:
                temp_raw_path.unlink(missing_ok=True)
            except OSError:
                pass
            run_state["current_temp_raw_path"] = None
            raise

        append_jsonl_file(temp_raw_path, raw_results_path)
        sync_query_results_file(raw_results_path, paths["query_results"])
        try:
            temp_raw_path.unlink(missing_ok=True)
        except OSError:
            pass
        run_state["current_temp_raw_path"] = None
        results.extend(chunk_results)
        completed_so_far += chunk_total
        status_progress(completed_so_far)

    if not pending_units:
        logging.info("No missing query tasks; using checkpointed raw results")
        sync_query_results_file(raw_results_path, paths["query_results"])
    stage_timings["query"] = time.monotonic() - query_start
    logging.info("阶段耗时：query=%.2fs", stage_timings["query"])
    logging.info("Collected %s responses", len(results))

    logging.info("Extracting fingerprint...")
    extract_start = time.monotonic()
    target_fp = extract_fingerprint_compat(
        model_name=args.target_model,
        family="unknown",
        results=results,
        probes=probes,
        primary_extractor=extract_fingerprint,
    )
    stage_timings["extract_fingerprint"] = time.monotonic() - extract_start
    logging.info("阶段耗时：extract_fingerprint=%.2fs", stage_timings["extract_fingerprint"])
    if target_fp.n_probes == 0:
        failed = target_fp.metadata.get("failed_queries", len(results))
        total = target_fp.metadata.get("total_queries", len(results))
        raise ValueError(
            f"All target queries failed ({failed}/{total}); "
            "check whether the target model supports the chat/completions API."
        )

    logging.info("Running similarity analysis...")
    similarity_start = time.monotonic()
    reference_fps = bank.all_fingerprints()
    bank_stats = bank.statistics
    if bank_stats is None and len(reference_fps) >= 2:
        bank_stats = bank.compute_statistics()

    weights = {
        "char_length": 0.25,
        "byte_length": 0.25,
        "token_type": 0.25,
        "transition": 0.25,
    }
    thresholds = {
        "ood_threshold": 0.4,
        "family_threshold": 0.7,
        "wrapped_threshold": 0.85,
        "variance_threshold": 0.15,
    }
    decision = similarity_module.make_decision(
        target_fp=target_fp,
        reference_fps=reference_fps,
        weights=weights,
        thresholds=thresholds,
        bank_stats=bank_stats,
    )
    stage_timings["similarity_analysis"] = time.monotonic() - similarity_start
    logging.info("阶段耗时：similarity_analysis=%.2fs", stage_timings["similarity_analysis"])
    elapsed = time.monotonic() - overall_start

    result = DetectionResult(
        target_model=args.target_model,
        label=decision["label"],
        confidence=decision["confidence"],
        top_matches=decision["top_matches"],
        stability_variance=decision["stability_variance"],
        bootstrap_mean=decision["bootstrap_mean"],
        bootstrap_std=decision["bootstrap_std"],
        details={
            "n_probes": len(probes),
            "n_results": len(results),
            "elapsed_seconds": elapsed,
            "stage_elapsed_seconds": stage_timings,
            "scoring_method": "bcs",
            "thresholds": thresholds,
            "request_hyperparameters": request_hyperparameters,
            "top1_score": decision.get("top1_score", 0.0),
            "top2_score": decision.get("top2_score", 0.0),
            "top1_minus_top2": decision.get("top1_minus_top2", 0.0),
        },
        target_fingerprint=target_fp,
        same_source_of=decision.get("same_source_of"),
        evidence=decision.get("evidence", {}),
        diagnosis=decision.get("diagnosis", ""),
    )

    result_payload = result.to_dict()
    result_payload["workspace"] = str(paths["workspace"])
    result_payload["reference"] = str(reference_dir)
    result_payload["probe_source"] = probe_source
    result_payload["selected_probe_file"] = str(selected_probe_path)
    result_payload["raw_results_file"] = str(raw_results_path)
    result_payload["resume_enabled"] = args.continue_run
    result_payload["query_total_count"] = status_total_count
    result_payload["checkpoint_reused_count"] = existing_completed
    result_payload["queried_count"] = status_total_count - existing_completed
    result_payload["elapsed_seconds"] = elapsed
    result_payload["stage_elapsed_seconds"] = stage_timings
    result_payload["request_hyperparameters"] = request_hyperparameters
    result_payload["query_results_file"] = str(paths["query_results"])
    result_payload["evaluation_results_file"] = str(paths["evaluation_results"])
    result_payload["timing_file"] = str(paths["timing"])

    result_path = paths["result"] / "result.json"
    write_json_file(result_path, result_payload)

    fingerprint_path = paths["result"] / "target_fingerprint.json"
    if result.target_fingerprint is not None:
        result.target_fingerprint.save(fingerprint_path, include_raw_results=True)

    raw_summary = summarize_raw_results(raw_results_path)
    per_probe_pass_count = min(len(probes), raw_summary["unique_usable_probe_count"])
    per_probe_fail_count = max(0, len(probes) - per_probe_pass_count)
    evaluation_stats = write_evaluation_results_jsonl(
        mode="detect",
        target_model=args.target_model,
        probes=probes,
        raw_results_path=raw_results_path,
        output_path=paths["evaluation_results"],
        result_payload=result_payload,
    )
    timing_payload = write_timing_files(
        paths,
        mode="detect",
        status="completed",
        elapsed_seconds=elapsed,
        stage_timings=stage_timings,
    )
    summary = {
        "total_count": len(probes),
        "pass_count": per_probe_pass_count,
        "fail_count": per_probe_fail_count,
        "na_count": 0,
        "query_total_count": status_total_count,
        "completed_count": len(probes),
        "failed_count": per_probe_fail_count,
        "query_rows": raw_summary["total_rows"],
        "query_pass_count": raw_summary["usable_rows"],
        "query_fail_count": raw_summary["failed_rows"],
        "empty_output_count": raw_summary["empty_rows"],
        "checkpoint_reused_count": existing_completed,
        "queried_count": status_total_count - existing_completed,
        "target_model": args.target_model,
        "label": result.label,
        "confidence": result.confidence,
        "same_source_of": result.same_source_of,
        "top_matches": result.top_matches[:5],
        "elapsed_seconds": elapsed,
        "stage_elapsed_seconds": stage_timings,
        "request_hyperparameters": request_hyperparameters,
        "result_file": str(result_path),
        "target_fingerprint_file": str(fingerprint_path),
        "query_results_file": str(paths["query_results"]),
        "evaluation_results_file": str(paths["evaluation_results"]),
        "timing_file": str(paths["timing"]),
        "timing": timing_payload,
        **evaluation_stats,
    }
    write_summary_files(paths, summary)
    result_payload["summary"] = summary
    for key in (
        "total_count",
        "pass_count",
        "fail_count",
        "na_count",
        "completed_count",
        "failed_count",
    ):
        result_payload[key] = summary[key]
    write_json_file(result_path, result_payload)

    if last_progress["finished"] < status_total_count:
        status_progress(status_total_count)
    write_status(
        status_path,
        {"type": "completed", "total_count": status_total_count, "stages": []},
    )
    run_state["terminal_written"] = True

    logging.info("测试总耗时：%.2fs", elapsed)
    logging.info("阶段耗时汇总：%s", format_stage_timings(stage_timings))
    logging.info("Detection completed in %.2fs", elapsed)
    logging.info("Result written to %s", result_path)
    logging.info("Target fingerprint written to %s", fingerprint_path)
    return result_payload


def run_vocab_extraction(args: argparse.Namespace, paths: dict[str, Path], run_state: dict[str, Any]):
    """Query next tokens and build a frequency-based simulated vocabulary."""
    from tokenizer_fingerprint.query_engine import APIConfig, query_model

    run_state["mode"] = "vocab"
    probes, probe_source, selected_probe_path = load_and_prepare_probes(args, paths["intermediate"])
    run_state["probes"] = probes
    run_state["target_model"] = args.target_model
    status_path = paths["logs"] / "status.log"
    status_total_count = len(probes)
    required_counts = {probe.id: 1 for probe in probes}
    write_status(
        status_path,
        {"type": "total", "total_count": status_total_count, "stages": []},
        initialize=True,
    )
    run_state["total_written"] = True
    run_state["total_count"] = status_total_count

    validate_target_args(args)
    ensure_no_proxy_for_target(args.target_base_url)

    api_key = args.target_api_key
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "local-test-key")

    request_extra_body = build_request_extra_body(args)
    request_hyperparameters = effective_request_hyperparameters(request_extra_body)
    api_config = APIConfig.from_dict(
        {
            "model": args.target_model,
            "api_key": api_key,
            "base_url": args.target_base_url,
            "extra_body": request_extra_body,
        },
        provider="openai",
    )

    concurrency = max(1, int(args.parallel_num or 1))
    safe_name = safe_model_name(args.target_model)
    raw_results_path = paths["result"] / "raw_tokens" / f"{safe_name}.jsonl"
    run_state["raw_results_path"] = raw_results_path
    run_state["query_results_path"] = paths["query_results"]

    logging.info("Starting next-token simulated vocabulary extraction")
    logging.info("Probe source: %s", probe_source)
    logging.info("Selected probes: %s", selected_probe_path)
    logging.info("Target model: %s", args.target_model)
    logging.info("Target base URL: %s", args.target_base_url)
    log_request_hyperparameters(request_hyperparameters, "vocab")
    if args.continue_run:
        logging.info("--continue was provided; loading raw token checkpoint if present")
    if args.auto_save_batch_size is not None:
        if args.auto_save_batch_size <= 0:
            raise ValueError("--auto_save_batch_size must be a positive integer")
        logging.info(
            "Raw query results will be flushed every %s completed query task(s)",
            args.auto_save_batch_size,
        )

    if args.continue_run:
        merged_parts = merge_part_checkpoints(raw_results_path)
        if merged_parts:
            logging.info(
                "Merged %s raw result(s) from temporary checkpoints before resume",
                merged_parts,
            )
        existing_results, existing_counts = load_checkpoint_results(
            raw_results_path,
            required_counts,
        )
    else:
        existing_results = []
        existing_counts = {probe_id: 0 for probe_id in required_counts}
        raw_results_path.parent.mkdir(parents=True, exist_ok=True)
        raw_results_path.write_text("", encoding="utf-8")
        sync_query_results_file(raw_results_path, paths["query_results"])

    existing_completed = sum(
        min(existing_counts.get(probe.id, 0), required_counts[probe.id])
        for probe in probes
    )

    overall_start = time.monotonic()
    stage_timings: dict[str, float] = {}
    run_state["stage_timings"] = stage_timings
    last_progress = {"finished": -1}

    def status_progress(finished_count: int) -> None:
        finished = min(int(finished_count), status_total_count)
        if finished <= last_progress["finished"]:
            return
        last_progress["finished"] = finished
        run_state["finished_count"] = finished
        write_status(
            status_path,
            {
                "type": "progress",
                "stage": "",
                "finished_count": finished,
                "total_count": status_total_count,
            },
        )
        logging.info("词表拟合进度：已完成 %s / %s", finished, status_total_count)

    logging.info("Probes: %s next-token query task(s)", len(probes))
    if existing_completed:
        logging.info(
            "Resume checkpoint found: %s/%s query tasks already completed",
            existing_completed,
            status_total_count,
        )
        status_progress(existing_completed)

    pending_units = [
        probe
        for probe in probes
        if existing_counts.get(probe.id, 0) < required_counts[probe.id]
    ]
    completed_so_far = existing_completed
    save_batch_size = args.auto_save_batch_size or len(pending_units) or 1
    pending_chunks = chunked(pending_units, save_batch_size)
    query_start = time.monotonic()

    for chunk_index, chunk_probes in enumerate(pending_chunks, 1):
        chunk_total = len(chunk_probes)
        temp_raw_path = raw_results_path.with_name(
            f"{raw_results_path.stem}.part-{os.getpid()}-{chunk_index}{raw_results_path.suffix}"
        )
        run_state["current_temp_raw_path"] = temp_raw_path

        def chunk_progress(chunk_finished: int, chunk_reported_total: int) -> None:
            status_progress(completed_so_far + min(chunk_finished, chunk_reported_total))

        logging.info(
            "Querying %s missing next-token task(s); checkpoint flush batch size=%s",
            chunk_total,
            save_batch_size,
        )
        try:
            asyncio.run(
                query_model(
                    probes=chunk_probes,
                    config=api_config,
                    model_name=args.target_model,
                    concurrency=concurrency,
                    progress_callback=chunk_progress,
                    raw_results_path=temp_raw_path,
                )
            )
        except Exception:
            appended = append_jsonl_file(temp_raw_path, raw_results_path)
            if appended:
                sync_query_results_file(raw_results_path, paths["query_results"])
                logging.info(
                    "Preserved %s partial raw result(s) in %s",
                    appended,
                    raw_results_path,
                )
            try:
                temp_raw_path.unlink(missing_ok=True)
            except OSError:
                pass
            run_state["current_temp_raw_path"] = None
            raise

        append_jsonl_file(temp_raw_path, raw_results_path)
        sync_query_results_file(raw_results_path, paths["query_results"])
        try:
            temp_raw_path.unlink(missing_ok=True)
        except OSError:
            pass
        run_state["current_temp_raw_path"] = None
        completed_so_far += chunk_total
        status_progress(completed_so_far)

    if not pending_units:
        logging.info("No missing query tasks; using checkpointed raw results")
        sync_query_results_file(raw_results_path, paths["query_results"])
    stage_timings["query"] = time.monotonic() - query_start
    logging.info("阶段耗时：query=%.2fs", stage_timings["query"])

    vocab_start = time.monotonic()
    stats = collect_next_token_counts(raw_results_path)
    counts: Counter[str] = stats["counts"]
    if stats["usable_rows"] == 0:
        raise ValueError(
            f"All next-token queries failed ({stats['failed_rows']}/{stats['total_rows']}); "
            "check whether the target model supports the chat/completions API."
        )
    vocab_csv_path = paths["result"] / "simulated_vocab.csv"
    vocab_json_path = paths["result"] / "simulated_vocab.json"
    write_simulated_vocab_csv(args.target_model, counts, vocab_csv_path)
    write_simulated_vocab_json(
        args.target_model,
        counts,
        vocab_json_path,
        metadata={
            "scoring_method": "next_token_frequency",
            "probe_source": probe_source,
            "selected_probe_file": str(selected_probe_path),
            "raw_results_file": str(raw_results_path),
            "request_hyperparameters": request_hyperparameters,
        },
    )
    stage_timings["build_vocab"] = time.monotonic() - vocab_start
    elapsed = time.monotonic() - overall_start

    raw_copy_path = paths["result"] / f"{safe_name}_raw_tokens.jsonl"
    if raw_copy_path != raw_results_path:
        shutil.copyfile(raw_results_path, raw_copy_path)

    evaluation_stats = write_evaluation_results_jsonl(
        mode="vocab",
        target_model=args.target_model,
        probes=probes,
        raw_results_path=raw_results_path,
        output_path=paths["evaluation_results"],
        summary={
            "unique_next_tokens": len(counts),
        },
    )
    timing_payload = write_timing_files(
        paths,
        mode="vocab",
        status="completed",
        elapsed_seconds=elapsed,
        stage_timings=stage_timings,
    )
    summary = {
        "mode": "vocab",
        "total_count": len(probes),
        "pass_count": stats["usable_rows"],
        "fail_count": stats["failed_rows"],
        "na_count": 0,
        "query_total_count": status_total_count,
        "completed_count": stats["total_rows"],
        "usable_count": stats["usable_rows"],
        "failed_count": stats["failed_rows"],
        "empty_output_count": stats["empty_rows"],
        "checkpoint_reused_count": existing_completed,
        "queried_count": status_total_count - existing_completed,
        "target_model": args.target_model,
        "unique_next_tokens": len(counts),
        "elapsed_seconds": elapsed,
        "stage_elapsed_seconds": stage_timings,
        "request_hyperparameters": request_hyperparameters,
        "probe_source": probe_source,
        "selected_probe_file": str(selected_probe_path),
        "raw_results_file": str(raw_results_path),
        "query_results_file": str(paths["query_results"]),
        "raw_tokens_file": str(raw_results_path),
        "raw_tokens_compat_file": str(raw_copy_path),
        "evaluation_results_file": str(paths["evaluation_results"]),
        "simulated_vocab_csv": str(vocab_csv_path),
        "simulated_vocab_json": str(vocab_json_path),
        "timing_file": str(paths["timing"]),
        "timing": timing_payload,
        **evaluation_stats,
        "top_tokens": [
            {
                "token": token,
                "frequency": frequency,
                "share": frequency / stats["usable_rows"] if stats["usable_rows"] else 0.0,
                "rank": rank,
            }
            for rank, (token, frequency) in enumerate(
                sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:20],
                start=1,
            )
        ],
    }
    write_summary_files(paths, summary)

    result_payload = {
        "status": "completed",
        "mode": "vocab",
        "target_model": args.target_model,
        "workspace": str(paths["workspace"]),
        "total_count": summary["total_count"],
        "pass_count": summary["pass_count"],
        "fail_count": summary["fail_count"],
        "na_count": summary["na_count"],
        "completed_count": summary["completed_count"],
        "failed_count": summary["failed_count"],
        "raw_tokens_file": str(raw_results_path),
        "query_results_file": str(paths["query_results"]),
        "evaluation_results_file": str(paths["evaluation_results"]),
        "simulated_vocab_csv": str(vocab_csv_path),
        "simulated_vocab_json": str(vocab_json_path),
        "timing_file": str(paths["timing"]),
        "request_hyperparameters": request_hyperparameters,
        "summary": summary,
    }
    result_path = paths["result"] / "result.json"
    write_json_file(result_path, result_payload)

    if last_progress["finished"] < status_total_count:
        status_progress(status_total_count)
    write_status(
        status_path,
        {"type": "completed", "total_count": status_total_count, "stages": []},
    )
    run_state["terminal_written"] = True

    logging.info("词表拟合总耗时：%.2fs", elapsed)
    logging.info("阶段耗时汇总：%s", format_stage_timings(stage_timings))
    logging.info("Raw next-token JSONL written to %s", raw_results_path)
    logging.info("Simulated vocab CSV written to %s", vocab_csv_path)
    logging.info("Simulated vocab JSON written to %s", vocab_json_path)
    return result_payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = Path(args.workspace).expanduser().resolve()
    paths = setup_workspace(workspace)
    configure_logging(paths["logs"] / "run.log")
    save_config(workspace, args)
    status_path = paths["logs"] / "status.log"
    run_state: dict[str, Any] = {
        "mode": args.mode,
        "target_model": args.target_model,
        "started_at": time.monotonic(),
        "stage_timings": {},
        "finished_count": 0,
        "total_written": False,
        "total_count": 0,
        "terminal_written": False,
        "raw_results_path": None,
        "query_results_path": paths["query_results"],
        "current_temp_raw_path": None,
        "terminating": False,
    }
    install_signal_handlers(run_state, status_path, paths)
    logging.info("Workspace: %s", workspace)
    logging.info("Run log: %s", paths["logs"] / "run.log")
    logging.info("Progress status: %s and %s", status_path, paths["status"])
    logging.info("Result directory: %s", paths["result"])

    try:
        if args.mode == "vocab":
            payload = run_vocab_extraction(args, paths, run_state)
            summary = payload["summary"]
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "result": "vocab",
                        "unique_next_tokens": summary["unique_next_tokens"],
                        "raw_tokens_file": payload["raw_tokens_file"],
                        "simulated_vocab_csv": payload["simulated_vocab_csv"],
                    },
                    ensure_ascii=False,
                )
            )
        else:
            payload = run_detection(args, paths, run_state)
            print(json.dumps({"status": "completed", "result": payload["label"], "confidence": payload["confidence"]}, ensure_ascii=False))
        return 0
    except Exception as exc:
        logging.error("Detection failed: %s", exc)
        logging.debug("Traceback:\n%s", traceback.format_exc())
        preserve_current_checkpoint(run_state)
        write_failed_status_once(run_state, status_path, str(exc))
        error_payload = write_failure_artifacts(paths, run_state, str(exc))
        logging.info(
            "测试总耗时：%.2fs",
            error_payload.get("elapsed_seconds", 0.0),
        )
        logging.info(
            "阶段耗时汇总：%s",
            format_stage_timings(error_payload.get("stage_elapsed_seconds", {})),
        )
        print(
            json.dumps(
                {"status": "failed", "error": str(exc), "result_file": str(paths["result"] / "result.json")},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
