"""Build an observed next-token vocabulary bank from raw token outputs.

This script implements the "model token set" baseline:
each reference model is represented by the set/frequency distribution of
observed next-token output_text values across probes. It intentionally does not
use probe alignment; it is a sidecar analysis for comparison with BCS.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

EMPTY_TOKEN_TEXT = "<EMPTY>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and evaluate a pure next-token observed vocabulary bank."
    )
    parser.add_argument(
        "--reference",
        default="reference_bank_core10000_en_cleaned",
        help="Reference bank directory containing index.json and raw_tokens/*.jsonl.",
    )
    parser.add_argument(
        "--truth-table",
        default="pair_level_truth_table.csv",
        help="YAML-style truth table for offline evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default="nexttoken_vocab_bank_core10000_model_set",
        help="Output directory for vocabulary bank and comparison files.",
    )
    parser.add_argument(
        "--normalize-token-text",
        action="store_true",
        help="Strip output_text before counting. Default preserves raw token text.",
    )
    parser.add_argument(
        "--top-token-limit",
        type=int,
        default=50,
        help="Number of top tokens to include per model in model_vocab.json.",
    )
    parser.add_argument(
        "--rare-max-count",
        type=int,
        default=3,
        help="Token frequency cutoff for rare_token_jaccard.",
    )
    parser.add_argument(
        "--weighted-jaccard-weight",
        type=float,
        default=0.60,
        help="Weight of weighted_jaccard in vocab_score.",
    )
    parser.add_argument(
        "--jaccard-weight",
        type=float,
        default=0.25,
        help="Weight of set jaccard in vocab_score.",
    )
    parser.add_argument(
        "--rare-jaccard-weight",
        type=float,
        default=0.15,
        help="Weight of rare_token_jaccard in vocab_score.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_dir = Path(args.reference)
    output_dir = Path(args.output_dir)
    index = load_json(reference_dir / "index.json")
    model_names = [item["model_name"] for item in index]
    families = {item["model_name"]: item["family"] for item in index}
    expected_probe_counts = {
        item["model_name"]: int(item.get("n_probes", 0))
        for item in index
    }
    truth = load_truth_table(Path(args.truth_table))
    validation_warnings = validate_truth(truth, model_names)

    vocab = build_model_vocab(
        reference_dir=reference_dir,
        model_names=model_names,
        families=families,
        expected_probe_counts=expected_probe_counts,
        normalize_tokens=args.normalize_token_text,
        top_token_limit=args.top_token_limit,
    )
    pairs = compute_pair_metrics(
        vocab=vocab,
        model_names=model_names,
        families=families,
        rare_max_count=args.rare_max_count,
        weighted_jaccard_weight=args.weighted_jaccard_weight,
        jaccard_weight=args.jaccard_weight,
        rare_jaccard_weight=args.rare_jaccard_weight,
    )
    nearest = build_nearest_neighbors(model_names, pairs)
    model_rows = model_top_rows(model_names, nearest)
    threshold_eval = evaluate_metric(model_rows, truth)
    topk_eval = evaluate_topk(model_names, nearest, truth)
    containment_rows = build_containment_rows(pairs)
    containment_nearest = build_containment_nearest(model_names, containment_rows)
    containment_model_rows = containment_top_rows(model_names, containment_nearest)
    containment_threshold_eval = evaluate_directional_metric(
        containment_model_rows,
        truth,
        score_field="shell_confidence",
        margin_field="shell_margin",
    )
    containment_topk_eval = evaluate_directional_topk(
        model_names,
        containment_nearest,
        truth,
        score_field="shell_confidence",
    )

    output = {
        "reference": str(reference_dir),
        "truth_table": str(args.truth_table),
        "normalize_token_text": args.normalize_token_text,
        "score_weights": {
            "weighted_jaccard": args.weighted_jaccard_weight,
            "jaccard": args.jaccard_weight,
            "rare_token_jaccard": args.rare_jaccard_weight,
        },
        "rare_max_count": args.rare_max_count,
        "n_models": len(model_names),
        "n_pairs": len(pairs),
        "validation_warnings": validation_warnings,
        "models": model_rows,
        "pairs": pairs,
        "nearest_neighbors": nearest,
        "threshold_eval": threshold_eval,
        "topk_eval": topk_eval,
        "containment": {
            "definition": "shell_confidence(a_to_b) = |V_a intersection V_b| / |V_a|",
            "direction": "model_a is the probed/source model; model_b is the candidate shell/wrapper vocabulary.",
            "rows": containment_rows,
            "models": containment_model_rows,
            "nearest_neighbors": containment_nearest,
            "threshold_eval": containment_threshold_eval,
            "topk_eval": containment_topk_eval,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "model_vocab.json", vocab)
    write_csv(output_dir / "pair_vocab_compare.csv", pairs)
    write_csv(output_dir / "directed_vocab_containment.csv", containment_rows)
    write_json(output_dir / "pair_vocab_compare.json", output)
    write_report(output_dir / "NEXTTOKEN_VOCAB_BANK_REPORT.md", output, vocab)

    print(f"Wrote {output_dir / 'model_vocab.json'}")
    print(f"Wrote {output_dir / 'pair_vocab_compare.csv'}")
    print(f"Wrote {output_dir / 'directed_vocab_containment.csv'}")
    print(f"Wrote {output_dir / 'pair_vocab_compare.json'}")
    print(f"Wrote {output_dir / 'NEXTTOKEN_VOCAB_BANK_REPORT.md'}")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_truth_table(path: Path) -> dict[str, list[str]]:
    truth: dict[str, list[str]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip() == "model_truth:":
            continue
        model_match = re.match(r"  ([^:]+):\s*$", line)
        if model_match:
            current = model_match.group(1)
            truth[current] = []
            continue
        target_match = re.match(r"    - (.+?)\s*$", line)
        if target_match and current:
            target = target_match.group(1).strip()
            if target != "none":
                truth[current].append(target)
    return truth


def validate_truth(truth: dict[str, list[str]], model_names: list[str]) -> list[str]:
    model_set = set(model_names)
    warnings: list[str] = []
    missing = sorted(model_set - set(truth))
    extra = sorted(set(truth) - model_set)
    if missing:
        warnings.append(
            "truth table missing models treated as none: " + ", ".join(missing)
        )
        for model in missing:
            truth[model] = []
    if extra:
        warnings.append("truth table has extra models ignored: " + ", ".join(extra))
    missing_targets = {
        model: [target for target in targets if target not in model_set]
        for model, targets in truth.items()
    }
    missing_targets = {k: v for k, v in missing_targets.items() if v}
    if missing_targets:
        raise ValueError(f"truth table targets not in model list: {missing_targets}")
    return warnings


def normalize_token_text(output_text: str, is_empty: bool, normalize: bool) -> str:
    if is_empty or output_text == "":
        return EMPTY_TOKEN_TEXT
    if normalize:
        stripped = output_text.strip()
        return stripped if stripped else EMPTY_TOKEN_TEXT
    return output_text


def build_model_vocab(
    reference_dir: Path,
    model_names: list[str],
    families: dict[str, str],
    expected_probe_counts: dict[str, int],
    normalize_tokens: bool,
    top_token_limit: int,
) -> dict[str, Any]:
    vocab: dict[str, Any] = {}
    for model in model_names:
        raw_path = reference_dir / "raw_tokens" / f"{model}.jsonl"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw token file: {raw_path}")
        token_counts: Counter[str] = Counter()
        total = 0
        empty = 0
        probe_ids: set[str] = set()
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                probe_ids.add(row["probe_id"])
                is_empty = bool(row.get("is_empty", False))
                token = normalize_token_text(
                    row.get("output_text", ""),
                    is_empty,
                    normalize_tokens,
                )
                token_counts[token] += 1
                total += 1
                if token == EMPTY_TOKEN_TEXT:
                    empty += 1
        expected = expected_probe_counts[model]
        vocab[model] = {
            "family": families[model],
            "n_probes": len(probe_ids),
            "total_token_count": total,
            "expected_n_probes": expected,
            "probe_count_matches_index": len(probe_ids) == expected,
            "unique_token_count": len(token_counts),
            "empty_token_count": empty,
            "empty_token_rate": empty / total if total else 0.0,
            "top_tokens": [
                {"token": token, "count": count}
                for token, count in token_counts.most_common(top_token_limit)
            ],
            "token_counts": dict(token_counts),
        }
    return vocab


def compute_pair_metrics(
    vocab: dict[str, Any],
    model_names: list[str],
    families: dict[str, str],
    rare_max_count: int,
    weighted_jaccard_weight: float,
    jaccard_weight: float,
    rare_jaccard_weight: float,
) -> list[dict[str, Any]]:
    weight_total = weighted_jaccard_weight + jaccard_weight + rare_jaccard_weight
    if weight_total <= 0:
        raise ValueError("score weights must sum to a positive value")
    pairs: list[dict[str, Any]] = []
    for i, model_a in enumerate(model_names):
        counts_a = Counter(vocab[model_a]["token_counts"])
        set_a = set(counts_a)
        rare_a = {token for token, count in counts_a.items() if count <= rare_max_count}
        for model_b in model_names[i + 1:]:
            counts_b = Counter(vocab[model_b]["token_counts"])
            set_b = set(counts_b)
            rare_b = {token for token, count in counts_b.items() if count <= rare_max_count}
            intersection = set_a & set_b
            union = set_a | set_b
            rare_union = rare_a | rare_b
            rare_intersection = rare_a & rare_b
            jaccard = len(intersection) / len(union) if union else 0.0
            coverage_a_to_b = len(intersection) / len(set_a) if set_a else 0.0
            coverage_b_to_a = len(intersection) / len(set_b) if set_b else 0.0
            weighted_jaccard = compute_weighted_jaccard(counts_a, counts_b)
            rare_token_jaccard = (
                len(rare_intersection) / len(rare_union) if rare_union else 0.0
            )
            vocab_score = (
                weighted_jaccard_weight * weighted_jaccard
                + jaccard_weight * jaccard
                + rare_jaccard_weight * rare_token_jaccard
            ) / weight_total
            pairs.append(
                {
                    "model_a": model_a,
                    "family_a": families[model_a],
                    "model_b": model_b,
                    "family_b": families[model_b],
                    "same_family": families[model_a] == families[model_b],
                    "vocab_score": vocab_score,
                    "jaccard": jaccard,
                    "coverage_a_to_b": coverage_a_to_b,
                    "coverage_b_to_a": coverage_b_to_a,
                    "weighted_jaccard": weighted_jaccard,
                    "rare_token_jaccard": rare_token_jaccard,
                    "intersection_token_count": len(intersection),
                    "union_token_count": len(union),
                    "unique_token_count_a": len(set_a),
                    "unique_token_count_b": len(set_b),
                    "rare_token_count_a": len(rare_a),
                    "rare_token_count_b": len(rare_b),
                    "rare_intersection_token_count": len(rare_intersection),
                }
            )
    pairs.sort(key=lambda row: row["vocab_score"], reverse=True)
    return pairs


def compute_weighted_jaccard(
    counts_a: Counter[str],
    counts_b: Counter[str],
) -> float:
    all_tokens = set(counts_a) | set(counts_b)
    if not all_tokens:
        return 0.0
    numerator = sum(min(counts_a[token], counts_b[token]) for token in all_tokens)
    denominator = sum(max(counts_a[token], counts_b[token]) for token in all_tokens)
    return numerator / denominator if denominator else 0.0


def build_nearest_neighbors(
    model_names: list[str],
    pairs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    nearest = {model: [] for model in model_names}
    for row in pairs:
        nearest[row["model_a"]].append(
            {
                "model": row["model_b"],
                "family": row["family_b"],
                "vocab_score": row["vocab_score"],
                "jaccard": row["jaccard"],
                "weighted_jaccard": row["weighted_jaccard"],
                "rare_token_jaccard": row["rare_token_jaccard"],
            }
        )
        nearest[row["model_b"]].append(
            {
                "model": row["model_a"],
                "family": row["family_a"],
                "vocab_score": row["vocab_score"],
                "jaccard": row["jaccard"],
                "weighted_jaccard": row["weighted_jaccard"],
                "rare_token_jaccard": row["rare_token_jaccard"],
            }
        )
    for model in model_names:
        nearest[model].sort(key=lambda item: item["vocab_score"], reverse=True)
    return nearest


def build_containment_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in pairs:
        rows.append(
            {
                "source_model": row["model_a"],
                "source_family": row["family_a"],
                "candidate_model": row["model_b"],
                "candidate_family": row["family_b"],
                "same_family": row["same_family"],
                "shell_confidence": row["coverage_a_to_b"],
                "source_vocab_size": row["unique_token_count_a"],
                "candidate_vocab_size": row["unique_token_count_b"],
                "intersection_token_count": row["intersection_token_count"],
                "jaccard": row["jaccard"],
                "weighted_jaccard": row["weighted_jaccard"],
                "rare_token_jaccard": row["rare_token_jaccard"],
                "reverse_shell_confidence": row["coverage_b_to_a"],
                "containment_gap": row["coverage_a_to_b"] - row["coverage_b_to_a"],
            }
        )
        rows.append(
            {
                "source_model": row["model_b"],
                "source_family": row["family_b"],
                "candidate_model": row["model_a"],
                "candidate_family": row["family_a"],
                "same_family": row["same_family"],
                "shell_confidence": row["coverage_b_to_a"],
                "source_vocab_size": row["unique_token_count_b"],
                "candidate_vocab_size": row["unique_token_count_a"],
                "intersection_token_count": row["intersection_token_count"],
                "jaccard": row["jaccard"],
                "weighted_jaccard": row["weighted_jaccard"],
                "rare_token_jaccard": row["rare_token_jaccard"],
                "reverse_shell_confidence": row["coverage_a_to_b"],
                "containment_gap": row["coverage_b_to_a"] - row["coverage_a_to_b"],
            }
        )
    rows.sort(key=lambda row: row["shell_confidence"], reverse=True)
    return rows


def build_containment_nearest(
    model_names: list[str],
    containment_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    nearest = {model: [] for model in model_names}
    for row in containment_rows:
        nearest[row["source_model"]].append(
            {
                "model": row["candidate_model"],
                "family": row["candidate_family"],
                "shell_confidence": row["shell_confidence"],
                "source_vocab_size": row["source_vocab_size"],
                "candidate_vocab_size": row["candidate_vocab_size"],
                "intersection_token_count": row["intersection_token_count"],
                "containment_gap": row["containment_gap"],
                "jaccard": row["jaccard"],
                "weighted_jaccard": row["weighted_jaccard"],
            }
        )
    for model in model_names:
        nearest[model].sort(key=lambda item: item["shell_confidence"], reverse=True)
    return nearest


def containment_top_rows(
    model_names: list[str],
    nearest: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for model in model_names:
        top1 = nearest[model][0]
        top2 = nearest[model][1]
        rows.append(
            {
                "model": model,
                "top1_model": top1["model"],
                "top1_family": top1["family"],
                "shell_confidence": top1["shell_confidence"],
                "shell_margin": top1["shell_confidence"] - top2["shell_confidence"],
                "source_vocab_size": top1["source_vocab_size"],
                "candidate_vocab_size": top1["candidate_vocab_size"],
                "intersection_token_count": top1["intersection_token_count"],
                "containment_gap": top1["containment_gap"],
                "jaccard": top1["jaccard"],
                "weighted_jaccard": top1["weighted_jaccard"],
            }
        )
    return rows


def model_top_rows(
    model_names: list[str],
    nearest: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for model in model_names:
        top1 = nearest[model][0]
        top2 = nearest[model][1]
        rows.append(
            {
                "model": model,
                "top1_model": top1["model"],
                "top1_family": top1["family"],
                "top1_vocab_score": top1["vocab_score"],
                "vocab_margin": top1["vocab_score"] - top2["vocab_score"],
                "top1_jaccard": top1["jaccard"],
                "top1_weighted_jaccard": top1["weighted_jaccard"],
                "top1_rare_token_jaccard": top1["rare_token_jaccard"],
            }
        )
    return rows


def evaluate_topk(
    model_names: list[str],
    nearest: dict[str, list[dict[str, Any]]],
    truth: dict[str, list[str]],
) -> dict[str, Any]:
    positive_models = [model for model in model_names if truth[model]]
    result: dict[str, Any] = {"n_positive_models": len(positive_models)}
    for k in (1, 3, 5):
        hits = 0
        for model in positive_models:
            candidates = {item["model"] for item in nearest[model][:k]}
            if candidates & set(truth[model]):
                hits += 1
        result[f"top{k}_hits"] = hits
        result[f"top{k}_target_hit_rate"] = (
            hits / len(positive_models) if positive_models else 0.0
        )
    return result


def evaluate_directional_topk(
    model_names: list[str],
    nearest: dict[str, list[dict[str, Any]]],
    truth: dict[str, list[str]],
    score_field: str,
) -> dict[str, Any]:
    positive_models = [model for model in model_names if truth[model]]
    result: dict[str, Any] = {"n_positive_models": len(positive_models)}
    for k in (1, 3, 5):
        hits = 0
        for model in positive_models:
            ranked = sorted(nearest[model], key=lambda item: item[score_field], reverse=True)
            candidates = {item["model"] for item in ranked[:k]}
            if candidates & set(truth[model]):
                hits += 1
        result[f"top{k}_hits"] = hits
        result[f"top{k}_target_hit_rate"] = (
            hits / len(positive_models) if positive_models else 0.0
        )
    return result


def evaluate_directional_metric(
    model_rows: list[dict[str, Any]],
    truth: dict[str, list[str]],
    score_field: str,
    margin_field: str,
) -> dict[str, Any]:
    score_values = sorted({round(row[score_field], 4) for row in model_rows})
    margin_values = sorted({round(row[margin_field], 4) for row in model_rows} | {0.0})
    results = []
    for score_threshold in score_values:
        for margin_threshold in margin_values:
            results.append(
                evaluate_directional_threshold(
                    model_rows,
                    truth,
                    score_field,
                    margin_field,
                    score_threshold,
                    margin_threshold,
                )
            )
    best_accuracy = max(
        results,
        key=lambda row: (row["accuracy"], row["f1"], row["precision"]),
    )
    best_f1 = max(
        results,
        key=lambda row: (row["f1"], row["accuracy"], row["precision"]),
    )
    high_precision = [
        row for row in results
        if row["precision"] >= 0.95 and row["tp"] > 0
    ]
    best_high_precision = (
        max(high_precision, key=lambda row: (row["recall"], row["accuracy"], row["f1"]))
        if high_precision
        else None
    )
    selected_thresholds = [
        evaluate_directional_threshold(model_rows, truth, score_field, margin_field, 0.50, 0.00),
        evaluate_directional_threshold(model_rows, truth, score_field, margin_field, 0.60, 0.00),
        evaluate_directional_threshold(model_rows, truth, score_field, margin_field, 0.70, 0.00),
        evaluate_directional_threshold(model_rows, truth, score_field, margin_field, 0.80, 0.00),
    ]
    return {
        "best_accuracy": best_accuracy,
        "best_f1": best_f1,
        "best_high_precision": best_high_precision,
        "selected": selected_thresholds,
    }


def evaluate_directional_threshold(
    model_rows: list[dict[str, Any]],
    truth: dict[str, list[str]],
    score_field: str,
    margin_field: str,
    score_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    false_positives = []
    false_negatives = []
    for row in model_rows:
        positives = truth[row["model"]]
        predicted = (
            row[score_field] >= score_threshold
            and row[margin_field] >= margin_threshold
        )
        hit = bool(positives) and row["top1_model"] in positives
        if predicted and hit:
            tp += 1
        elif predicted and not hit:
            fp += 1
            false_positives.append(row)
        elif not predicted and positives:
            fn += 1
            false_negatives.append(row)
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / len(model_rows) if model_rows else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "score_threshold": score_threshold,
        "margin_threshold": margin_threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def evaluate_metric(
    model_rows: list[dict[str, Any]],
    truth: dict[str, list[str]],
) -> dict[str, Any]:
    score_values = sorted({round(row["top1_vocab_score"], 4) for row in model_rows})
    margin_values = sorted({round(row["vocab_margin"], 4) for row in model_rows} | {0.0})
    results = []
    for score_threshold in score_values:
        for margin_threshold in margin_values:
            results.append(
                evaluate_threshold(model_rows, truth, score_threshold, margin_threshold)
            )
    best_accuracy = max(
        results,
        key=lambda row: (row["accuracy"], row["f1"], row["precision"]),
    )
    best_f1 = max(
        results,
        key=lambda row: (row["f1"], row["accuracy"], row["precision"]),
    )
    high_precision = [
        row for row in results
        if row["precision"] >= 0.95 and row["tp"] > 0
    ]
    best_high_precision = (
        max(high_precision, key=lambda row: (row["recall"], row["accuracy"], row["f1"]))
        if high_precision
        else None
    )
    selected_thresholds = [
        evaluate_threshold(model_rows, truth, 0.20, 0.00),
        evaluate_threshold(model_rows, truth, 0.30, 0.00),
        evaluate_threshold(model_rows, truth, 0.40, 0.00),
        evaluate_threshold(model_rows, truth, 0.40, 0.05),
        evaluate_threshold(model_rows, truth, 0.50, 0.00),
    ]
    return {
        "best_accuracy": best_accuracy,
        "best_f1": best_f1,
        "best_high_precision": best_high_precision,
        "selected": selected_thresholds,
    }


def evaluate_threshold(
    model_rows: list[dict[str, Any]],
    truth: dict[str, list[str]],
    score_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    false_positives = []
    false_negatives = []
    for row in model_rows:
        positives = truth[row["model"]]
        predicted = (
            row["top1_vocab_score"] >= score_threshold
            and row["vocab_margin"] >= margin_threshold
        )
        hit = bool(positives) and row["top1_model"] in positives
        if predicted and hit:
            tp += 1
        elif predicted and not hit:
            fp += 1
            false_positives.append(row)
        elif not predicted and positives:
            fn += 1
            false_negatives.append(row)
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / len(model_rows) if model_rows else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "score_threshold": score_threshold,
        "margin_threshold": margin_threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, output: dict[str, Any], vocab: dict[str, Any]) -> None:
    token_counts = [row["unique_token_count"] for row in vocab.values()]
    pair_scores = [row["vocab_score"] for row in output["pairs"]]
    same_scores = [row["vocab_score"] for row in output["pairs"] if row["same_family"]]
    cross_scores = [row["vocab_score"] for row in output["pairs"] if not row["same_family"]]
    lines = [
        "# Next-token 观测词表库分析报告",
        "",
        "## 摘要",
        "",
        f"本报告基于 `{output['reference']}` 中 `index.json` 的 `{output['n_models']}` 个正式模型生成纯 next-token 集合词表库。该方法只合并每个模型观测到的 `output_text`，不保留 probe 对齐关系，因此是 BCS 的补充指标，不是真实 tokenizer 完整词表。",
        "",
        f"模型唯一 token 数范围：`{min(token_counts)}` 到 `{max(token_counts)}`；pair 数量：`{output['n_pairs']}`。",
        f"vocab_score 范围：`{min(pair_scores):.4f}` 到 `{max(pair_scores):.4f}`；same-family 均值 `{mean(same_scores):.4f}`，cross-family 均值 `{mean(cross_scores):.4f}`。",
        "",
        "## Top-k 目标命中",
        "",
        "| 指标 | Top1 | Top3 | Top5 | 正例模型数 |",
        "| --- | ---: | ---: | ---: | ---: |",
        topk_table_row(output["topk_eval"]),
        "",
        "## 阈值扫描摘要",
        "",
        "| 规则 | score_threshold | margin_threshold | TP | FP | FN | TN | Precision | Recall | Accuracy | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        threshold_table_row("best_accuracy", output["threshold_eval"]["best_accuracy"]),
        threshold_table_row("best_f1", output["threshold_eval"]["best_f1"]),
        threshold_table_row("best_high_precision", output["threshold_eval"]["best_high_precision"]),
        "",
        "## Top1 最近邻",
        "",
        "| 模型 | top1 | vocab_score | margin | weighted_jaccard | jaccard | rare_jaccard |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["models"]:
        lines.append(
            f"| {row['model']} | {row['top1_model']} | "
            f"{row['top1_vocab_score']:.4f} | {row['vocab_margin']:.4f} | "
            f"{row['top1_weighted_jaccard']:.4f} | {row['top1_jaccard']:.4f} | "
            f"{row['top1_rare_token_jaccard']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 该词表库是黑盒观测词表：只包含 probe 触发过的 next-token 文本。",
            "- 纯 token 集合会丢失上下文，容易被通用高频 token 影响。",
            "- 如果该方法单独准确率不高，下一步应尝试 probe 对齐词表库，而不是继续调纯集合阈值。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def topk_table_row(row: dict[str, Any]) -> str:
    return (
        f"| vocab_score | {row['top1_target_hit_rate']:.3f} | "
        f"{row['top3_target_hit_rate']:.3f} | {row['top5_target_hit_rate']:.3f} | "
        f"{row['n_positive_models']} |"
    )


def threshold_table_row(name: str, row: dict[str, Any] | None) -> str:
    if not row:
        return f"| {name} | - | - | 0 | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |"
    return (
        f"| {name} | {row['score_threshold']:.4f} | {row['margin_threshold']:.4f} | "
        f"{row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} | "
        f"{row['precision']:.3f} | {row['recall']:.3f} | "
        f"{row['accuracy']:.3f} | {row['f1']:.3f} |"
    )


if __name__ == "__main__":
    main()
