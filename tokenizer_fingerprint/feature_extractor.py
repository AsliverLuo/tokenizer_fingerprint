"""
feature_extractor.py — 三组特征抽取

第一组：表面长度特征 (Surface Features)
  - char_len_hist, byte_len_hist, byte_per_char_ratio
  - 前导空格率, 前导换行率, 空输出率

第二组：Token 类型特征 (Type Features)
  - 每个 token 输出的类型标签分布

第三组：上下文→输出转移特征 (Transition Features)
  - context_category × output_type 联合概率
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Optional

from .schema import (
    Probe,
    SingleTokenResult,
    TokenType,
    SurfaceFeatures,
    TypeFeatures,
    TransitionFeatures,
    ModelFingerprint,
)


# ── Token 类型分类器 ────────────────────────────────────────────

# 常见代码符号和关键字片段
CODE_PATTERNS = re.compile(
    r"^[\(\)\{\}\[\];:=<>+\-*/&|!~^%@#\\]"
    r"|^(def |class |import |from |return |if |else |for |while "
    r"|var |let |const |function |async |await |try |catch "
    r"|fn |pub |impl |struct |enum |match )"
)

JSON_PATTERNS = re.compile(
    r'^["\{}\[\]:,]'
    r"|^(true|false|null)"
)

URL_PATTERNS = re.compile(
    r"^(https?://|ftp://|mailto:|/[a-zA-Z]|\.\.?/|[a-zA-Z]:\\)"
)


def classify_token(text: str) -> str:
    """
    对单个 token 输出文本进行类型分类。

    优先级从高到低匹配，返回第一个命中的类型。
    """
    if not text:
        return TokenType.EMPTY.value

    # 检查前导符号（在剥离前检查）
    if text.startswith("\n") or text.startswith("\r"):
        return TokenType.NEWLINE_PREFIXED.value
    if text.startswith(" ") or text.startswith("\t"):
        return TokenType.WHITESPACE_PREFIXED.value

    stripped = text.strip()
    if not stripped:
        return TokenType.EMPTY.value

    # Emoji / Symbol
    if all(_is_emoji_or_symbol(c) for c in stripped):
        return TokenType.EMOJI_SYMBOL.value

    # URL-like
    if URL_PATTERNS.match(stripped):
        return TokenType.URL_LIKE.value

    # JSON-like
    if JSON_PATTERNS.match(stripped):
        return TokenType.JSON_LIKE.value

    # Code-like
    if CODE_PATTERNS.match(stripped):
        return TokenType.CODE_LIKE.value

    # Pure digit (must contain at least one actual digit)
    if any(c.isdigit() for c in stripped) and all(c.isdigit() or c in ".,+-eExX" for c in stripped):
        return TokenType.DIGIT.value

    # Pure punctuation
    if all(unicodedata.category(c).startswith("P") for c in stripped):
        return TokenType.PUNCTUATION.value

    # Script detection
    has_cjk = any(_is_cjk(c) for c in stripped)
    has_latin = any(c.isascii() and c.isalpha() for c in stripped)

    if has_cjk and has_latin:
        return TokenType.MIXED_SCRIPT.value
    if has_cjk:
        return TokenType.CJK.value
    if has_latin:
        return TokenType.LATIN.value

    return TokenType.OTHER.value


def _is_cjk(char: str) -> bool:
    """判断字符是否为 CJK 统一表意文字"""
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


def _is_emoji_or_symbol(char: str) -> bool:
    cat = unicodedata.category(char)
    return cat.startswith("So") or cat.startswith("Sk") or ord(char) > 0x1F300


# ── 特征抽取 ────────────────────────────────────────────────────

def extract_surface_features(results: list[SingleTokenResult]) -> SurfaceFeatures:
    """
    第一组：表面长度特征
    """
    if not results:
        return SurfaceFeatures()

    n = len(results)
    char_lens = Counter()
    byte_lens = Counter()
    bpc_ratios = []
    leading_space_count = 0
    leading_newline_count = 0
    empty_count = 0

    for r in results:
        char_lens[r.char_length] += 1
        byte_lens[r.byte_length] += 1

        if r.char_length > 0:
            bpc_ratios.append(r.byte_length / r.char_length)
        if r.has_leading_space:
            leading_space_count += 1
        if r.has_leading_newline:
            leading_newline_count += 1
        if r.is_empty:
            empty_count += 1

    # 归一化 histogram
    char_hist = {str(k): v / n for k, v in sorted(char_lens.items())}
    byte_hist = {str(k): v / n for k, v in sorted(byte_lens.items())}

    return SurfaceFeatures(
        char_len_hist=char_hist,
        byte_len_hist=byte_hist,
        byte_per_char_ratio_mean=(
            sum(bpc_ratios) / len(bpc_ratios) if bpc_ratios else 0.0
        ),
        leading_space_rate=leading_space_count / n,
        leading_newline_rate=leading_newline_count / n,
        empty_output_rate=empty_count / n,
    )


def extract_type_features(results: list[SingleTokenResult]) -> TypeFeatures:
    """
    第二组：Token 类型分布特征
    """
    if not results:
        return TypeFeatures()

    n = len(results)
    type_counts = Counter()

    for r in results:
        # 对每个结果进行分类（如果还没有分类）
        token_type = r.token_type
        if token_type == "other" and r.output_text:
            token_type = classify_token(r.output_text)
        elif r.is_empty:
            token_type = TokenType.EMPTY.value
        type_counts[token_type] += 1

    # 归一化
    type_dist = {k: v / n for k, v in sorted(type_counts.items())}

    return TypeFeatures(type_distribution=type_dist)


def extract_transition_features(
    results: list[SingleTokenResult],
    probes: list[Probe],
) -> TransitionFeatures:
    """
    第三组：上下文→输出类型转移特征

    构建 context_category × output_type 联合概率矩阵。
    """
    if not results or not probes:
        return TransitionFeatures()

    # 建立 probe_id -> category 映射
    probe_cat_map = {p.id: p.category for p in probes}

    # 统计 (context_category, output_type) 联合频率
    transition_counts: dict[str, Counter] = defaultdict(Counter)
    category_totals: Counter = Counter()

    for r in results:
        cat = probe_cat_map.get(r.probe_id, "unknown")
        token_type = r.token_type
        if token_type == "other" and r.output_text:
            token_type = classify_token(r.output_text)
        elif r.is_empty:
            token_type = TokenType.EMPTY.value

        transition_counts[cat][token_type] += 1
        category_totals[cat] += 1

    # 归一化为条件概率 P(output_type | context_category)
    transition_matrix = {}
    for cat, type_counts in transition_counts.items():
        total = category_totals[cat]
        if total > 0:
            for t, c in type_counts.items():
                key = f"{cat}→{t}"
                transition_matrix[key] = c / total

    return TransitionFeatures(transition_matrix=transition_matrix)


def extract_fingerprint(
    model_name: str,
    family: str,
    results: list[SingleTokenResult],
    probes: list[Probe],
) -> ModelFingerprint:
    """
    从查询结果中抽取完整的模型指纹。

    整合三组特征：surface + type + transition
    """
    valid_results = [r for r in results if "error" not in r.raw_response]
    normalization_counts = Counter()
    normalized_output_count = 0
    normalization_config = {}
    empty_retry_configured = 0
    empty_retry_attempt_count = 0
    empty_retry_used_count = 0
    empty_response_count = 0
    recovered_after_empty_count = 0
    final_empty_after_retry_count = 0

    for r in valid_results:
        normalization_meta = r.raw_response.get("_output_normalization", {})
        if normalization_meta:
            normalization_config = normalization_meta.get("config", normalization_config)
            if normalization_meta.get("changed"):
                normalized_output_count += 1
            for name in normalization_meta.get("applied", []):
                normalization_counts[name] += 1

        empty_retry_meta = r.raw_response.get("_empty_output_retry", {})
        if empty_retry_meta:
            empty_retry_configured = max(
                empty_retry_configured,
                int(empty_retry_meta.get("configured_retries", 0) or 0),
            )
            empty_retry_attempt_count += int(
                empty_retry_meta.get("attempt_count", 0) or 0
            )
            empty_retry_used_count += int(
                empty_retry_meta.get("empty_retries_used", 0) or 0
            )
            empty_response_count += int(
                empty_retry_meta.get("empty_response_count", 0) or 0
            )
            if empty_retry_meta.get("recovered_after_empty"):
                recovered_after_empty_count += 1
            if empty_retry_meta.get("final_is_empty"):
                final_empty_after_retry_count += 1

    # 先为所有结果填充 token_type
    for r in valid_results:
        if (r.token_type == "other" or not r.token_type) and r.output_text:
            r.token_type = classify_token(r.output_text)
        elif r.is_empty:
            r.token_type = TokenType.EMPTY.value

    surface = extract_surface_features(valid_results)
    type_feat = extract_type_features(valid_results)
    transition = extract_transition_features(valid_results, probes)

    return ModelFingerprint(
        model_name=model_name,
        family=family,
        surface=surface,
        type_feat=type_feat,
        transition=transition,
        n_probes=len(set(r.probe_id for r in valid_results)),
        raw_results=valid_results,
        metadata={
            "failed_queries": len(results) - len(valid_results),
            "total_queries": len(results),
            "output_normalization": {
                "config": normalization_config,
                "normalized_output_count": normalized_output_count,
                "operation_counts": dict(normalization_counts),
            },
            "empty_output_retry": {
                "configured_retries": empty_retry_configured,
                "results_with_empty_retry": sum(
                    1
                    for r in valid_results
                    if r.raw_response.get("_empty_output_retry")
                ),
                "total_attempt_count_for_retried_results": empty_retry_attempt_count,
                "empty_retries_used": empty_retry_used_count,
                "empty_response_count": empty_response_count,
                "recovered_after_empty_count": recovered_after_empty_count,
                "final_empty_after_retry_count": final_empty_after_retry_count,
            },
            "probe_manifest": {
                p.id: {
                    "text": p.text,
                    "category": p.category,
                    "truncation_ratio": p.truncation_ratio,
                    "source_lang": p.source_lang,
                    "metadata": p.metadata,
                }
                for p in probes
            },
        },
    )
