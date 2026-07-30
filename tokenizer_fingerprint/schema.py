"""
schema.py — 数据结构定义

定义 probe、查询结果、特征向量、参考库条目、检测结果等核心数据结构。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


# ── Probe 相关 ──────────────────────────────────────────────────

class ProbeCategory(str, Enum):
    """Probe 上下文类别"""
    CHINESE_NATURAL = "chinese_natural"
    ENGLISH_NATURAL = "english_natural"
    CODE = "code"
    CHINESE_ENGLISH_MIXED = "chinese_english_mixed"
    NUMBER_DATE_AMOUNT = "number_date_amount"
    JSON_YAML_MARKDOWN = "json_yaml_markdown"
    URL_PATH_EMAIL = "url_path_email"
    TRUNCATED_PARTIAL = "truncated_partial"


@dataclass
class Probe:
    """单条 probe prompt"""
    id: str
    text: str
    category: str
    truncation_ratio: float = 0.0
    source_lang: str = "mixed"
    metadata: dict = field(default_factory=dict)


# ── 查询结果 ────────────────────────────────────────────────────

class TokenType(str, Enum):
    """单 token 输出的类型标签"""
    CJK = "cjk"
    LATIN = "latin"
    DIGIT = "digit"
    PUNCTUATION = "punctuation"
    WHITESPACE_PREFIXED = "whitespace_prefixed"
    NEWLINE_PREFIXED = "newline_prefixed"
    CODE_LIKE = "code_like"
    JSON_LIKE = "json_like"
    URL_LIKE = "url_like"
    MIXED_SCRIPT = "mixed_script"
    EMOJI_SYMBOL = "emoji_symbol"
    EMPTY = "empty"
    OTHER = "other"


@dataclass
class SingleTokenResult:
    """单次查询的原始结果"""
    probe_id: str
    model_name: str
    output_text: str               # 模型输出的原始文本
    char_length: int = 0
    byte_length: int = 0
    has_leading_space: bool = False
    has_leading_newline: bool = False
    token_type: str = "other"
    is_empty: bool = False
    latency_ms: float = 0.0
    raw_response: dict = field(default_factory=dict)


# ── 特征向量 ────────────────────────────────────────────────────

@dataclass
class SurfaceFeatures:
    """第一层：表面长度特征"""
    char_len_hist: dict = field(default_factory=dict)      # {length: count}
    byte_len_hist: dict = field(default_factory=dict)      # {length: count}
    byte_per_char_ratio_mean: float = 0.0
    leading_space_rate: float = 0.0
    leading_newline_rate: float = 0.0
    empty_output_rate: float = 0.0


@dataclass
class TypeFeatures:
    """第二层：token 类型分布特征"""
    type_distribution: dict = field(default_factory=dict)  # {type: proportion}


@dataclass
class TransitionFeatures:
    """第二层扩展：上下文→输出类型转移特征"""
    # {(context_category, output_type): probability}
    transition_matrix: dict = field(default_factory=dict)


@dataclass
class ModelFingerprint:
    """模型指纹 = 三组特征的组合"""
    model_name: str
    family: str
    surface: SurfaceFeatures = field(default_factory=SurfaceFeatures)
    type_feat: TypeFeatures = field(default_factory=TypeFeatures)
    transition: TransitionFeatures = field(default_factory=TransitionFeatures)
    n_probes: int = 0
    raw_results: list[SingleTokenResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(
        self,
        include_raw_results: bool = False,
        include_raw_response: bool = False,
    ) -> dict:
        d = asdict(self)
        if not include_raw_results:
            d.pop("raw_results", None)
        elif not include_raw_response:
            for r in d.get("raw_results", []):
                raw_response = r.get("raw_response", {})
                compact_raw_response = {}
                if "error" in raw_response:
                    compact_raw_response["error"] = raw_response["error"]
                for key in (
                    "_output_normalization",
                    "_raw_output_text",
                    "_empty_output_retry",
                ):
                    if key in raw_response:
                        compact_raw_response[key] = raw_response[key]
                if compact_raw_response:
                    r["raw_response"] = compact_raw_response
                else:
                    r.pop("raw_response", None)
        return d

    def save(
        self,
        path: Path,
        include_raw_results: bool = False,
        include_raw_response: bool = False,
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.to_dict(
                    include_raw_results=include_raw_results,
                    include_raw_response=include_raw_response,
                ),
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> "ModelFingerprint":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        fp = cls(
            model_name=d["model_name"],
            family=d["family"],
            n_probes=d.get("n_probes", 0),
            metadata=d.get("metadata", {}),
        )
        sf = d.get("surface", {})
        fp.surface = SurfaceFeatures(
            char_len_hist=sf.get("char_len_hist", {}),
            byte_len_hist=sf.get("byte_len_hist", {}),
            byte_per_char_ratio_mean=sf.get("byte_per_char_ratio_mean", 0.0),
            leading_space_rate=sf.get("leading_space_rate", 0.0),
            leading_newline_rate=sf.get("leading_newline_rate", 0.0),
            empty_output_rate=sf.get("empty_output_rate", 0.0),
        )
        fp.type_feat = TypeFeatures(
            type_distribution=d.get("type_feat", {}).get("type_distribution", {})
        )
        fp.transition = TransitionFeatures(
            transition_matrix=d.get("transition", {}).get("transition_matrix", {})
        )
        fp.raw_results = [
            SingleTokenResult(**r)
            for r in d.get("raw_results", [])
        ]
        if not fp.raw_results:
            sidecar = path.parent.parent / "raw_tokens" / f"{fp.model_name}.jsonl"
            if sidecar.exists():
                with open(sidecar, "r", encoding="utf-8") as f:
                    fp.raw_results = [
                        SingleTokenResult(**json.loads(line))
                        for line in f
                        if line.strip()
                    ]
        return fp


# ── 参考库统计 ──────────────────────────────────────────────────

@dataclass
class BankStatistics:
    """参考库跨家族 BCS 分布统计"""
    cross_family_mean: float = 0.0
    cross_family_std: float = 0.0
    cross_family_p95: float = 0.0
    cross_family_p99: float = 0.0
    cross_family_max: float = 0.0
    n_cross_pairs: int = 0
    n_models: int = 0

    def to_dict(self) -> dict:
        return {
            "cross_family_mean": self.cross_family_mean,
            "cross_family_std": self.cross_family_std,
            "cross_family_p95": self.cross_family_p95,
            "cross_family_p99": self.cross_family_p99,
            "cross_family_max": self.cross_family_max,
            "n_cross_pairs": self.n_cross_pairs,
            "n_models": self.n_models,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BankStatistics":
        return cls(
            cross_family_mean=d.get("cross_family_mean", 0.0),
            cross_family_std=d.get("cross_family_std", 0.0),
            cross_family_p95=d.get("cross_family_p95", 0.0),
            cross_family_p99=d.get("cross_family_p99", 0.0),
            cross_family_max=d.get("cross_family_max", 0.0),
            n_cross_pairs=d.get("n_cross_pairs", 0),
            n_models=d.get("n_models", 0),
        )


# ── 检测结果 ────────────────────────────────────────────────────

class DetectionLabel(str, Enum):
    SAME_SOURCE = "same_source"
    NOT_SAME_SOURCE = "not_same_source"


@dataclass
class DetectionResult:
    """最终检测判定"""
    target_model: str
    label: str
    confidence: float
    top_matches: list[dict] = field(default_factory=list)   # [{model, score, family}]
    stability_variance: float = 0.0
    bootstrap_mean: float = 0.0
    bootstrap_std: float = 0.0
    details: dict = field(default_factory=dict)
    target_fingerprint: Optional[ModelFingerprint] = None
    same_source_of: Optional[str] = None    # 同源指向哪个参考模型
    evidence: dict = field(default_factory=dict)   # 多证据详情
    diagnosis: str = ""                     # 人类可读判定理由

    def to_dict(self) -> dict:
        return {
            "target_model": self.target_model,
            "label": self.label,
            "confidence": self.confidence,
            "top_matches": self.top_matches,
            "stability_variance": self.stability_variance,
            "bootstrap_mean": self.bootstrap_mean,
            "bootstrap_std": self.bootstrap_std,
            "details": self.details,
            "same_source_of": self.same_source_of,
            "evidence": self.evidence,
            "diagnosis": self.diagnosis,
        }
