"""
detector.py — 三级检测主流程

Pipeline:
  1. Probe Generation  → 生成边界敏感 probe 集
  2. Query Execution   → 统一协议查询目标模型
  3. Feature Extraction → 抽取三组指纹特征
  4. Layer 1: Surface Fingerprint → 表面分词指纹
  5. Layer 2: Type/Boundary       → 边界与类型指纹
  6. Layer 3: Confirmation        → 一致性确认
  7. Decision                     → 四分类判定
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

from .schema import (
    Probe,
    SingleTokenResult,
    ModelFingerprint,
    DetectionResult,
    DetectionLabel,
)
from .probe_generator import generate_probes, load_probes, save_probes
from .query_engine import APIConfig, query_model
from .feature_extractor import extract_fingerprint
from .reference_bank import ReferenceBank
from .similarity import (
    compute_similarity,
    compute_similarity_breakdown,
    bootstrap_similarity,
    compute_stability_variance,
    make_decision,
)

logger = logging.getLogger(__name__)


class TokenizerFingerprintDetector:
    """
    Tokenizer 指纹检测器

    三级检测链设计：
      Layer 1: Surface Form Distribution → 快速筛查 tokenizer 外观风格
      Layer 2: Token Type / Boundary     → 更细的边界归因
      Layer 3: Consistency Confirmation  → 低预算一致性确认

    使用方式:
        detector = TokenizerFingerprintDetector(reference_bank)
        result = detector.detect(target_config, probes)
    """

    def __init__(
        self,
        reference_bank: ReferenceBank,
        weights: Optional[dict[str, float]] = None,
        thresholds: Optional[dict[str, float]] = None,
    ):
        self.reference_bank = reference_bank
        self.weights = weights or {
            "char_length": 0.25,
            "byte_length": 0.25,
            "token_type": 0.25,
            "transition": 0.25,
        }
        self.thresholds = thresholds or {
            "ood_threshold": 0.4,
            "family_threshold": 0.7,
            "wrapped_threshold": 0.85,
            "variance_threshold": 0.15,
        }

    async def detect_async(
        self,
        target_config: APIConfig,
        target_name: str,
        probes: list[Probe],
        concurrency: int = 5,
        stability_ratio: float = 0.1,
        stability_repeats: int = 3,
        raw_results_path: Optional[Path] = None,
    ) -> DetectionResult:
        """
        异步执行完整的三级检测。

        Args:
            target_config: 目标模型 API 配置
            target_name: 目标模型名称
            probes: probe 集合
            concurrency: 并发查询数
            stability_ratio: 稳定性采样比例
            stability_repeats: 稳定性重复次数

        Returns:
            DetectionResult
        """
        logger.info(f"Starting detection for: {target_name}")
        t0 = time.monotonic()

        # ── Step 1: 选择稳定性采样的 probes ──
        rng = random.Random(42)
        stability_count = max(1, int(len(probes) * stability_ratio))
        stability_ids = set(
            p.id for p in rng.sample(probes, min(stability_count, len(probes)))
        )
        logger.info(
            f"Probes: {len(probes)} total, {len(stability_ids)} for stability check"
        )
        # ── Step 2: 查询目标模型 ──
        logger.info("Querying target model...")
        results = await query_model(
            probes=probes,
            config=target_config,
            model_name=target_name,
            concurrency=concurrency,
            stability_repeat_ids=stability_ids,
            stability_repeat_count=stability_repeats,
            raw_results_path=raw_results_path,
        )
        logger.info(f"Collected {len(results)} responses")

        # ── Step 3: 抽取指纹 ──
        logger.info("Extracting fingerprint...")
        target_fp = extract_fingerprint(
            model_name=target_name,
            family="unknown",
            results=results,
            probes=probes,
        )
        if target_fp.n_probes == 0:
            failed = target_fp.metadata.get("failed_queries", len(results))
            total = target_fp.metadata.get("total_queries", len(results))
            raise ValueError(
                f"All target queries failed ({failed}/{total}); "
                "check whether the target model supports the chat/completions API."
            )

        # ── Step 4-6: 三级检测与决策 ──
        logger.info("Running similarity analysis...")
        reference_fps = self.reference_bank.all_fingerprints()

        # 获取或计算 bank statistics
        bank_stats = self.reference_bank.statistics
        if bank_stats is None and len(reference_fps) >= 2:
            bank_stats = self.reference_bank.compute_statistics()

        decision = make_decision(
            target_fp=target_fp,
            reference_fps=reference_fps,
            weights=self.weights,
            thresholds=self.thresholds,
            bank_stats=bank_stats,
        )

        elapsed = time.monotonic() - t0
        logger.info(
            f"Detection complete in {elapsed:.1f}s → {decision['label']} "
            f"(confidence={decision['confidence']:.3f})"
        )

        return DetectionResult(
            target_model=target_name,
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
                "scoring_method": "bcs",
                "thresholds": self.thresholds,
                "top1_score": decision.get("top1_score", 0.0),
                "top2_score": decision.get("top2_score", 0.0),
                "top1_minus_top2": decision.get("top1_minus_top2", 0.0),
            },
            target_fingerprint=target_fp,
            same_source_of=decision.get("same_source_of"),
            evidence=decision.get("evidence", {}),
            diagnosis=decision.get("diagnosis", ""),
        )

    def detect(
        self,
        target_config: APIConfig,
        target_name: str,
        probes: list[Probe],
        **kwargs,
    ) -> DetectionResult:
        """同步检测入口"""
        return asyncio.run(
            self.detect_async(target_config, target_name, probes, **kwargs)
        )


async def build_reference_fingerprint(
    model_config: APIConfig,
    model_name: str,
    family: str,
    probes: list[Probe],
    concurrency: int = 5,
    raw_results_path: Optional[Path] = None,
) -> ModelFingerprint:
    """
    为参考模型构建指纹。

    Args:
        model_config: 模型 API 配置
        model_name: 模型名称
        family: 模型家族
        probes: probe 集合
        concurrency: 并发数

    Returns:
        ModelFingerprint
    """
    logger.info(f"Building fingerprint for: {model_name} (family={family})")

    results = await query_model(
        probes=probes,
        config=model_config,
        model_name=model_name,
        concurrency=concurrency,
        raw_results_path=raw_results_path,
    )

    fp = extract_fingerprint(
        model_name=model_name,
        family=family,
        results=results,
        probes=probes,
    )
    if fp.n_probes == 0:
        failed = fp.metadata.get("failed_queries", len(results))
        total = fp.metadata.get("total_queries", len(results))
        raise ValueError(
            f"All reference queries failed ({failed}/{total}); "
            "check model name, base_url, API key, and provider-specific parameters."
        )

    logger.info(
        f"Fingerprint built: {fp.n_probes} probes, "
        f"types={list(fp.type_feat.type_distribution.keys())}"
    )

    return fp


def build_reference_fingerprint_sync(
    model_config: APIConfig,
    model_name: str,
    family: str,
    probes: list[Probe],
    concurrency: int = 5,
    raw_results_path: Optional[Path] = None,
) -> ModelFingerprint:
    """同步封装"""
    return asyncio.run(
        build_reference_fingerprint(
            model_config,
            model_name,
            family,
            probes,
            concurrency,
            raw_results_path,
        )
    )
