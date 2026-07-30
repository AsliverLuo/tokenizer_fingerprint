"""
similarity_soft.py - Soft Boundary Consistency Score (Soft BCS)

This module is a non-breaking companion to similarity.py.
It keeps the same general structure but replaces the strict
"all boundary fields must match" score with a weighted soft score.

Soft BCS is intended for family-level affinity analysis:
  - exact boundary matches still get full credit
  - partial matches on length / token type / prefix still contribute
  - same-family sibling models should score higher than under strict BCS
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from .schema import BankStatistics, DetectionLabel, ModelFingerprint, SingleTokenResult


DEFAULT_SOFT_WEIGHTS = {
    "exact": 0.60,
    "char_length": 0.07,
    "byte_length": 0.07,
    "token_type": 0.16,
    "prefix": 0.07,
    "empty": 0.03,
}


def compute_similarity(
    target: ModelFingerprint,
    reference: ModelFingerprint,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """Compute Soft BCS."""
    return soft_boundary_consistency_score(target, reference, weights)


def soft_boundary_consistency_score(
    target: ModelFingerprint,
    reference: ModelFingerprint,
    weights: Optional[dict[str, float]] = None,
) -> float:
    return float(compute_similarity_breakdown(target, reference, weights)["soft_bcs"])


def compute_similarity_breakdown(
    target: ModelFingerprint,
    reference: ModelFingerprint,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Return soft score and its component match rates."""
    score_weights = _resolve_weights(weights)
    target_by_probe = _first_result_by_probe(target.raw_results)
    reference_by_probe = _first_result_by_probe(reference.raw_results)
    common_ids = sorted(set(target_by_probe) & set(reference_by_probe))
    common_count = len(common_ids)

    if common_count == 0:
        return {
            "soft_bcs": 0.0,
            "common_probe_count": 0,
            "target_probe_count": len(target_by_probe),
            "reference_probe_count": len(reference_by_probe),
            "boundary_exact_match_rate": 0.0,
            "char_length_match_rate": 0.0,
            "byte_length_match_rate": 0.0,
            "token_type_match_rate": 0.0,
            "prefix_match_rate": 0.0,
            "empty_match_rate": 0.0,
        }

    exact = 0
    char_len = 0
    byte_len = 0
    token_type = 0
    prefix = 0
    empty = 0

    for probe_id in common_ids:
        target_result = target_by_probe[probe_id]
        reference_result = reference_by_probe[probe_id]

        if _boundary_signature(target_result) == _boundary_signature(reference_result):
            exact += 1
        if target_result.char_length == reference_result.char_length:
            char_len += 1
        if target_result.byte_length == reference_result.byte_length:
            byte_len += 1
        if _normalized_token_type(target_result) == _normalized_token_type(reference_result):
            token_type += 1
        if (
            target_result.has_leading_space == reference_result.has_leading_space
            and target_result.has_leading_newline == reference_result.has_leading_newline
        ):
            prefix += 1
        if target_result.is_empty == reference_result.is_empty:
            empty += 1

    exact_rate = exact / common_count
    char_len_rate = char_len / common_count
    byte_len_rate = byte_len / common_count
    token_type_rate = token_type / common_count
    prefix_rate = prefix / common_count
    empty_rate = empty / common_count

    soft_bcs = (
        score_weights["exact"] * exact_rate
        + score_weights["char_length"] * char_len_rate
        + score_weights["byte_length"] * byte_len_rate
        + score_weights["token_type"] * token_type_rate
        + score_weights["prefix"] * prefix_rate
        + score_weights["empty"] * empty_rate
    )

    return {
        "soft_bcs": soft_bcs,
        "common_probe_count": common_count,
        "target_probe_count": len(target_by_probe),
        "reference_probe_count": len(reference_by_probe),
        "boundary_exact_match_rate": exact_rate,
        "char_length_match_rate": char_len_rate,
        "byte_length_match_rate": byte_len_rate,
        "token_type_match_rate": token_type_rate,
        "prefix_match_rate": prefix_rate,
        "empty_match_rate": empty_rate,
    }


def bootstrap_similarity(
    target: ModelFingerprint,
    reference: ModelFingerprint,
    weights: Optional[dict[str, float]] = None,
    n_iterations: int = 100,
    sample_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap Soft BCS over common probes."""
    score_weights = _resolve_weights(weights)
    target_by_probe = _first_result_by_probe(target.raw_results)
    reference_by_probe = _first_result_by_probe(reference.raw_results)
    common_ids = sorted(set(target_by_probe) & set(reference_by_probe))

    if not common_ids:
        return 0.0, 0.0

    rng = np.random.RandomState(seed)
    sample_n = max(1, int(len(common_ids) * sample_ratio))
    scores = []

    for _ in range(n_iterations):
        sampled_ids = rng.choice(common_ids, size=sample_n, replace=True)
        total = 0.0
        for probe_id in sampled_ids:
            total += _pair_soft_score(
                target_by_probe[probe_id],
                reference_by_probe[probe_id],
                score_weights,
            )
        scores.append(total / sample_n)

    return float(np.mean(scores)), float(np.std(scores))


def compute_stability_variance(results: list[SingleTokenResult]) -> float:
    """
    Keep the same repeated-query instability metric as similarity.py.
    """
    probe_outputs: dict[str, list[str]] = defaultdict(list)
    for result in results:
        if "error" in result.raw_response:
            continue
        probe_outputs[result.probe_id].append(result.output_text)

    repeated = {
        probe_id: outputs
        for probe_id, outputs in probe_outputs.items()
        if len(outputs) > 1
    }
    if not repeated:
        return 0.0

    inconsistent = sum(1 for outputs in repeated.values() if len(set(outputs)) > 1)
    return inconsistent / len(repeated)


def make_decision(
    target_fp: ModelFingerprint,
    reference_fps: list[ModelFingerprint],
    weights: Optional[dict[str, float]] = None,
    thresholds: Optional[dict[str, float]] = None,
    bootstrap_n: int = 100,
    bank_stats: Optional[BankStatistics] = None,
) -> dict:
    """基于多证据打分的 Soft BCS 判定。

    证据信号与 strict BCS 相同，但使用 Soft BCS 分数。
    自匹配过滤: 排除 reference_fps 中与目标同名的模型。
    """
    # ── 自匹配过滤 ──
    reference_fps = [
        fp for fp in reference_fps
        if fp.model_name != target_fp.model_name
    ]

    matches = []
    for reference_fp in reference_fps:
        score = compute_similarity(target_fp, reference_fp, weights)
        breakdown = compute_similarity_breakdown(target_fp, reference_fp, weights)
        bs_mean, bs_std = bootstrap_similarity(
            target_fp,
            reference_fp,
            weights,
            n_iterations=bootstrap_n,
        )
        matches.append({
            "model": reference_fp.model_name,
            "family": reference_fp.family,
            "score": score,
            "soft_bcs": score,
            "bootstrap_mean": bs_mean,
            "bootstrap_std": bs_std,
            "breakdown": breakdown,
        })

    matches.sort(key=lambda x: x["score"], reverse=True)

    stability_var = compute_stability_variance(target_fp.raw_results)
    top = matches[0] if matches else None
    second = matches[1] if len(matches) > 1 else None
    top1_bcs = top["score"] if top else 0.0
    top2_bcs = second["score"] if second else 0.0
    margin = top1_bcs - top2_bcs
    bs_mean = top["bootstrap_mean"] if top else 0.0
    bs_std = top["bootstrap_std"] if top else 0.0

    # ── 证据计算 ──
    reasons: list[str] = []
    evidence_for = 0
    evidence_against = 0

    if bank_stats and bank_stats.cross_family_std > 0:
        z_score = (top1_bcs - bank_stats.cross_family_mean) / bank_stats.cross_family_std
    else:
        z_score = 0.0

    if z_score >= 2.0:
        evidence_for += 2
        reasons.append(f"Soft BCS far above cross-family baseline (z={z_score:.1f})")
    elif z_score >= 1.5:
        evidence_for += 1
        reasons.append(f"Soft BCS above cross-family baseline (z={z_score:.1f})")
    elif z_score < 0.5 and bank_stats is not None:
        evidence_against += 2
        reasons.append(f"Soft BCS within cross-family noise range (z={z_score:.1f})")

    margin_sigma = margin / (bs_std + 1e-10)
    if margin_sigma >= 3.0:
        evidence_for += 2
        reasons.append(f"margin statistically robust ({margin_sigma:.0f}σ)")
    elif margin_sigma >= 2.0:
        evidence_for += 1
        reasons.append(f"margin statistically significant ({margin_sigma:.0f}σ)")
    else:
        evidence_against += 1
        reasons.append(f"margin not significant ({margin_sigma:.0f}σ)")

    top_family = top["family"] if top else ""
    top3 = matches[:3]
    fam_consistent = sum(1 for m in top3 if m["family"] == top_family)
    if fam_consistent >= 2:
        evidence_for += 1
        reasons.append(f"top-3 family consistent ({fam_consistent}/3={top_family})")
    else:
        evidence_against += 1
        reasons.append(f"top-3 family scattered ({fam_consistent}/3)")

    if stability_var > 0.15:
        evidence_against += 1
        reasons.append(f"target unstable (var={stability_var:.3f})")

    # ── 判定 ──
    if evidence_for >= 4:
        label = DetectionLabel.SAME_SOURCE
        confidence = min(0.95, 0.65 + 0.05 * evidence_for)
    elif evidence_for >= 3:
        label = DetectionLabel.SAME_SOURCE
        confidence = min(0.85, 0.55 + 0.05 * evidence_for)
    elif evidence_for >= 2 and evidence_against <= 1:
        label = DetectionLabel.SAME_SOURCE
        confidence = 0.50 + 0.05 * evidence_for
    else:
        label = DetectionLabel.NOT_SAME_SOURCE
        confidence = min(0.95, 0.55 + 0.10 * evidence_against)

    evidence = {
        "z_score_vs_cross_family": round(z_score, 2),
        "margin_sigma": round(margin_sigma, 1),
        "family_consistency": f"{fam_consistent}/{len(top3)}",
        "evidence_for": evidence_for,
        "evidence_against": evidence_against,
    }
    if bank_stats:
        evidence["cross_family_baseline"] = {
            "mean": round(bank_stats.cross_family_mean, 4),
            "std": round(bank_stats.cross_family_std, 4),
            "p95": round(bank_stats.cross_family_p95, 4),
            "p99": round(bank_stats.cross_family_p99, 4),
        }

    return {
        "label": label.value,
        "confidence": round(confidence, 4),
        "top_matches": matches[:5],
        "top1_score": top1_bcs,
        "top2_score": top2_bcs,
        "top1_minus_top2": margin,
        "stability_variance": stability_var,
        "bootstrap_mean": bs_mean,
        "bootstrap_std": bs_std,
        "same_source_of": top["model"] if label == DetectionLabel.SAME_SOURCE and top else None,
        "evidence": evidence,
        "diagnosis": "; ".join(reasons) if reasons else "no reference models to compare",
    }


def _pair_soft_score(
    target_result: SingleTokenResult,
    reference_result: SingleTokenResult,
    weights: dict[str, float],
) -> float:
    exact = float(_boundary_signature(target_result) == _boundary_signature(reference_result))
    char_len = float(target_result.char_length == reference_result.char_length)
    byte_len = float(target_result.byte_length == reference_result.byte_length)
    token_type = float(
        _normalized_token_type(target_result) == _normalized_token_type(reference_result)
    )
    prefix = float(
        target_result.has_leading_space == reference_result.has_leading_space
        and target_result.has_leading_newline == reference_result.has_leading_newline
    )
    empty = float(target_result.is_empty == reference_result.is_empty)

    return (
        weights["exact"] * exact
        + weights["char_length"] * char_len
        + weights["byte_length"] * byte_len
        + weights["token_type"] * token_type
        + weights["prefix"] * prefix
        + weights["empty"] * empty
    )


def _resolve_weights(weights: Optional[dict[str, float]]) -> dict[str, float]:
    merged = dict(DEFAULT_SOFT_WEIGHTS)
    if weights:
        merged.update(weights)

    total = sum(merged.values())
    if total <= 0:
        return dict(DEFAULT_SOFT_WEIGHTS)

    return {k: float(v) / total for k, v in merged.items()}


def _first_result_by_probe(results: list[SingleTokenResult]) -> dict[str, SingleTokenResult]:
    grouped: dict[str, SingleTokenResult] = {}
    for result in results:
        if "error" in result.raw_response:
            continue
        grouped.setdefault(result.probe_id, result)
    return grouped


def _normalized_token_type(result: SingleTokenResult) -> str:
    return result.token_type or "other"


def _boundary_signature(result: SingleTokenResult) -> tuple:
    return (
        result.char_length,
        result.byte_length,
        result.has_leading_space,
        result.has_leading_newline,
        _normalized_token_type(result),
        result.is_empty,
    )
