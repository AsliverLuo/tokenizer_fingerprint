"""
similarity.py - Boundary Consistency Score (BCS)

BCS 按 probe_id 对齐目标模型和参考模型的原始单 token 结果，
比较两者的边界签名是否完全一致：
  (char_length, byte_length, has_leading_space,
   has_leading_newline, token_type, is_empty)

BCS = 边界签名一致的 probe 数 / 共同 probe 数
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from .schema import BankStatistics, DetectionLabel, ModelFingerprint, SingleTokenResult


def compute_similarity(
    target: ModelFingerprint,
    reference: ModelFingerprint,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """计算 BCS。weights 参数保留作兼容，不参与打分。"""
    return boundary_consistency_score(target, reference)


def boundary_consistency_score(
    target: ModelFingerprint,
    reference: ModelFingerprint,
) -> float:
    return float(compute_similarity_breakdown(target, reference)["bcs"])


def compute_bank_statistics(
    reference_fps: list[ModelFingerprint],
    bank_compare_pairs: Optional[list[dict]] = None,
) -> BankStatistics:
    """计算参考库跨家族 BCS 分布统计。

    Args:
        reference_fps: 参考指纹列表
        bank_compare_pairs: 可选，来自 compare-bank 输出的 pairs 列表。
                           如果提供则直接提取统计量，否则实时计算。

    Returns:
        BankStatistics
    """
    cross_scores: list[float] = []

    if bank_compare_pairs:
        cross_scores = [
            float(p["score"]) for p in bank_compare_pairs
            if p.get("family_a") != p.get("family_b")
        ]
        families = set()
        for p in bank_compare_pairs:
            families.add(p.get("family_a", ""))
            families.add(p.get("family_b", ""))
        n_models = len(
            set(p["model_a"] for p in bank_compare_pairs)
            | set(p["model_b"] for p in bank_compare_pairs)
        )
    else:
        families: set[str] = set()
        models: set[str] = set()
        for i, fp1 in enumerate(reference_fps):
            models.add(fp1.model_name)
            families.add(fp1.family)
            for fp2 in reference_fps[i + 1:]:
                if fp1.family != fp2.family:
                    score = boundary_consistency_score(fp1, fp2)
                    cross_scores.append(score)
        for fp2 in reference_fps:
            models.add(fp2.model_name)
        n_models = len(models)

    cross_scores.sort()
    n = len(cross_scores)

    if n == 0:
        return BankStatistics(n_models=n_models)

    mean = sum(cross_scores) / n
    std = (
        (sum((s - mean) ** 2 for s in cross_scores) / (n - 1)) ** 0.5
        if n > 1
        else 0.0
    )

    def _percentile(data: list[float], pct: float) -> float:
        idx = min(len(data) - 1, max(0, int(len(data) * pct / 100)))
        return data[idx]

    return BankStatistics(
        cross_family_mean=mean,
        cross_family_std=std,
        cross_family_p95=_percentile(cross_scores, 95),
        cross_family_p99=_percentile(cross_scores, 99),
        cross_family_max=cross_scores[-1],
        n_cross_pairs=n,
        n_models=n_models,
    )


def compute_similarity_breakdown(
    target: ModelFingerprint,
    reference: ModelFingerprint,
) -> dict[str, float]:
    """返回 BCS 及边界一致性诊断分项。"""
    target_by_probe = _first_result_by_probe(target.raw_results)
    reference_by_probe = _first_result_by_probe(reference.raw_results)
    common_ids = sorted(set(target_by_probe) & set(reference_by_probe))
    common_count = len(common_ids)

    if common_count == 0:
        return {
            "bcs": 0.0,
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

    bcs = exact / common_count
    return {
        "bcs": bcs,
        "common_probe_count": common_count,
        "target_probe_count": len(target_by_probe),
        "reference_probe_count": len(reference_by_probe),
        "boundary_exact_match_rate": bcs,
        "char_length_match_rate": char_len / common_count,
        "byte_length_match_rate": byte_len / common_count,
        "token_type_match_rate": token_type / common_count,
        "prefix_match_rate": prefix / common_count,
        "empty_match_rate": empty / common_count,
    }


def bootstrap_similarity(
    target: ModelFingerprint,
    reference: ModelFingerprint,
    weights: Optional[dict[str, float]] = None,
    n_iterations: int = 100,
    sample_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[float, float]:
    """对共同 probe 做 bootstrap，返回 BCS 均值和标准差。"""
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
        matches = 0
        for probe_id in sampled_ids:
            if (
                _boundary_signature(target_by_probe[probe_id])
                == _boundary_signature(reference_by_probe[probe_id])
            ):
                matches += 1
        scores.append(matches / sample_n)

    return float(np.mean(scores)), float(np.std(scores))


def compute_stability_variance(results: list[SingleTokenResult]) -> float:
    """
    计算重复采样方差。

    对同一 probe_id 出现多次的结果，计算其 output_text 的不一致率。
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
    """基于多证据打分的 BCS 判定。

    证据信号:
      - Z-score: top1 BCS 高出跨家族分布均值多少个标准差
      - Margin 显著性: margin / bootstrap_std
      - Family 一致性: top-3 匹配中与 top1 同家族的占比
      - 稳定性: 目标模型自身输出的稳定性方差

    自匹配过滤: 排除 reference_fps 中与目标同名的模型。
    """
    # ── 自匹配过滤 ──
    reference_fps = [
        fp for fp in reference_fps
        if fp.model_name != target_fp.model_name
    ]

    # ── 计算所有参考模型的 BCS ──
    matches = []
    for reference_fp in reference_fps:
        score = compute_similarity(target_fp, reference_fp, weights)
        breakdown = compute_similarity_breakdown(target_fp, reference_fp)
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
            "bcs": score,
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
    evidence: dict[str, object] = {}
    reasons: list[str] = []
    evidence_for = 0
    evidence_against = 0

    # Z-score vs 跨家族分布
    if bank_stats and bank_stats.cross_family_std > 0:
        z_score = (top1_bcs - bank_stats.cross_family_mean) / bank_stats.cross_family_std
    else:
        z_score = 0.0

    if z_score >= 2.0:
        evidence_for += 2
        reasons.append(f"BCS far above cross-family baseline (z={z_score:.1f})")
    elif z_score >= 1.5:
        evidence_for += 1
        reasons.append(f"BCS above cross-family baseline (z={z_score:.1f})")
    elif z_score < 0.5 and bank_stats is not None:
        evidence_against += 2
        reasons.append(f"BCS within cross-family noise range (z={z_score:.1f})")

    # Margin 显著性
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

    # Family 一致性
    top_family = top["family"] if top else ""
    top3 = matches[:3]
    fam_consistent = sum(1 for m in top3 if m["family"] == top_family)
    if fam_consistent >= 2:
        evidence_for += 1
        reasons.append(f"top-3 family consistent ({fam_consistent}/3={top_family})")
    else:
        evidence_against += 1
        reasons.append(f"top-3 family scattered ({fam_consistent}/3)")

    # 稳定性
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

    # 组装 evidence
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


def _first_result_by_probe(
    results: list[SingleTokenResult],
) -> dict[str, SingleTokenResult]:
    by_probe = {}
    for result in results:
        if "error" in result.raw_response:
            continue
        by_probe.setdefault(result.probe_id, result)
    return by_probe


def _boundary_signature(result: SingleTokenResult) -> tuple:
    return (
        result.char_length,
        result.byte_length,
        result.has_leading_space,
        result.has_leading_newline,
        _normalized_token_type(result),
        result.is_empty,
    )


def _normalized_token_type(result: SingleTokenResult) -> str:
    if result.token_type and result.token_type != "other":
        return result.token_type
    if result.output_text == "":
        return "empty"
    if result.output_text.startswith(("\n", "\r")):
        return "newline_prefixed"
    if result.output_text.startswith((" ", "\t")):
        return "whitespace_prefixed"
    return result.token_type or "other"
