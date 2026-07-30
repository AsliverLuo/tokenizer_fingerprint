"""
probe_generator.py — Probe Prompt 生成器

设计原则：
- 短 prompt，避免高层语义主导续写
- 边界敏感：半句、半词、半个代码符号
- 多语言多格式覆盖
- 多截断比例：30% / 50% / 70% / 90%
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Optional

from .schema import Probe, ProbeCategory


# ── 内置种子语料 ────────────────────────────────────────────────
# 每类提供边界敏感的 seed texts，实际使用时可从外部语料扩展

SEED_CORPUS = {
    ProbeCategory.CHINESE_NATURAL: [
        "今天北京的天气很好，适合出门散步。",
        "这份研究报告主要讨论了量子计算的最新进展。",
        "随着人工智能技术的发展，越来越多的企业开始关注数据安全问题。",
        "深度学习模型在自然语言处理领域取得了显著的突破。",
        "北京时间今天上午十点，国务院新闻办举行了发布会。",
        "大语言模型的推理能力一直是学术界讨论的焦点。",
        "我们需要在效率和安全之间找到一个平衡点。",
        "昨天的会议上大家讨论了下个季度的产品规划。",
        "中国科学院发布了最新的人工智能发展报告。",
        "这个算法的时间复杂度是对数线性的。",
        "在自然语言理解任务中，预训练模型通常表现优异。",
        "请将以下文本翻译成英文并保持原有格式。",
        "该方法在多个基准测试集上均达到了最优性能。",
        "根据最新的统计数据显示，全球半导体市场持续增长。",
        "我们的系统能够实时处理大规模数据流并生成分析报告。",
    ],
    ProbeCategory.ENGLISH_NATURAL: [
        "The security vulnerability was discovered in the authentication module.",
        "This method achieves state-of-the-art results on multiple benchmarks.",
        "Recent advances in large language models have transformed the field.",
        "The quarterly earnings report exceeded analyst expectations by a wide margin.",
        "We propose a novel framework for distributed training of neural networks.",
        "The experimental results demonstrate significant improvements over baselines.",
        "In this paper we present a comprehensive analysis of tokenization strategies.",
        "The European Central Bank announced its latest interest rate decision.",
        "Climate scientists have published new findings about ocean temperature rises.",
        "The implementation uses a transformer-based architecture with attention mechanisms.",
        "Our approach significantly reduces inference latency while maintaining accuracy.",
        "The study examined the relationship between model size and emergent capabilities.",
        "Several major technology companies reported strong revenue growth this quarter.",
        "The algorithm converges in polynomial time under standard assumptions.",
        "Researchers have identified a novel class of protein structures using AlphaFold.",
    ],
    ProbeCategory.CODE: [
        "def load_config(path: str) -> dict:",
        "for i in range(len(tokens)):",
        "class TokenizerFingerprint:",
        "    def __init__(self, model_name: str):",
        "import torch\nimport numpy as np",
        "async def fetch_completion(prompt: str,",
        "if response.status_code == 200:",
        "    return json.loads(response.text)",
        "try:\n    result = await client.chat.completions.create(",
        "SELECT user_id, COUNT(*) as cnt FROM",
        "const handleSubmit = async (e: React.FormEvent) => {",
        "fn tokenize(input: &str) -> Vec<Token> {",
        "public static void main(String[] args) {",
        "docker run -d --name inference-server -p 8080:",
        "git commit -m 'fix: resolve tokenizer",
    ],
    ProbeCategory.CHINESE_ENGLISH_MIXED: [
        "请输出一个 JSON schema，field name 用 English。",
        "这个 model 的 latency 大概在 50ms 左右。",
        "我们使用 PyTorch 框架来实现这个 attention mechanism。",
        "下面是一段 Python 代码，实现了 binary search 算法。",
        "API 返回的 response 中包含了 token 的 logprobs。",
        "用 transformer 架构实现一个 encoder-decoder 模型。",
        "在 fine-tuning 阶段，我们使用了 LoRA 技术。",
        "这个 bug 出现在 authentication 模块的 middleware 中。",
        "根据 benchmark 结果，这个方法的 accuracy 提升了 3%。",
        "请帮我 review 一下这段 code 的 edge case 处理。",
    ],
    ProbeCategory.NUMBER_DATE_AMOUNT: [
        "2026-04-",
        "$1,234.",
        "3.1415926535",
        "2024年12月31日",
        "¥9,999.",
        "Tel: +86-10-1234",
        "ISBN 978-0-13-468",
        "v2.0.1-beta.",
        "192.168.1.",
        "0xFF3A",
        "1,000,000,000",
        "2025/03/",
        "€2,450.",
        "3.7×10^",
        "12:30:45.",
    ],
    ProbeCategory.JSON_YAML_MARKDOWN: [
        '{"user_id": 1024, "stat',
        "## Experimental",
        '{"model": "gpt-4", "temperature":',
        "```python\ndef",
        "- item: tokenizer\n  version:",
        '| Column A | Column B |',
        '{"results": [{"score":',
        "### 3.2 Method",
        "* **Important**: The",
        '{"config": {"max_tokens":',
        "---\ntitle: Fingerprint\nauthor:",
        "> Note: This is a",
        '{"error": {"code": 429, "message":',
        "1. First, install the",
        "```json\n{",
    ],
    ProbeCategory.URL_PATH_EMAIL: [
        "https://api.openai.com/v1/chat/",
        "mailto:admin@example.",
        "/home/user/.config/",
        "https://huggingface.co/models/",
        "C:\\Users\\dev\\Documents\\",
        "git@github.com:anthropic/",
        "ftp://files.example.com/data/",
        "../../src/tokenizer/",
        "user@company.com",
        "https://arxiv.org/abs/2",
    ],
    ProbeCategory.TRUNCATED_PARTIAL: [
        "The securit",
        "量子计算的最新进",
        "import torc",
        "def __ini",
        "https://ap",
        '{"user_i',
        "这个模型的推理速度非常",
        "async def fe",
        "transforme",
        "SELECT * FRO",
        "北京时间今天上午",
        "The experimental resu",
        "我们提出了一种新",
        "class Tok",
        "pip instal",
    ],
}


def _truncate_text(text: str, ratio: float) -> str:
    """按比例截断文本"""
    cut_point = max(1, int(len(text) * ratio))
    return text[:cut_point]


def _make_id() -> str:
    return uuid.uuid4().hex[:12]


def generate_probes(
    total_count: int = 500,
    category_weights: Optional[dict[str, float]] = None,
    truncation_ratios: Optional[list[float]] = None,
    seed: int = 42,
) -> list[Probe]:
    """
    生成 probe prompt 集合。

    Args:
        total_count: 总 probe 数量
        category_weights: 各类别的权重，如 {"chinese_natural": 0.15, ...}
        truncation_ratios: 截断比例列表，如 [0.3, 0.5, 0.7, 0.9]
        seed: 随机种子

    Returns:
        Probe 列表
    """
    rng = random.Random(seed)

    if category_weights is None:
        category_weights = {
            "chinese_natural": 0.15,
            "english_natural": 0.15,
            "code": 0.15,
            "chinese_english_mixed": 0.10,
            "number_date_amount": 0.10,
            "json_yaml_markdown": 0.10,
            "url_path_email": 0.10,
            "truncated_partial": 0.15,
        }

    if truncation_ratios is None:
        truncation_ratios = [0.3, 0.5, 0.7, 0.9]

    probes: list[Probe] = []

    # 按权重分配每类数量
    categories = list(category_weights.keys())
    weights = [category_weights[c] for c in categories]
    total_weight = sum(weights)
    counts = {
        c: max(1, int(total_count * w / total_weight))
        for c, w in zip(categories, weights)
    }

    # 调整总数
    diff = total_count - sum(counts.values())
    if diff > 0:
        counts[categories[0]] += diff

    for cat_name, count in counts.items():
        cat_enum = ProbeCategory(cat_name)
        seeds = SEED_CORPUS.get(cat_enum, [])
        if not seeds:
            continue

        for i in range(count):
            seed_text = rng.choice(seeds)

            # truncated_partial 类别直接使用原文（已经是截断的）
            if cat_enum == ProbeCategory.TRUNCATED_PARTIAL:
                probe_text = seed_text
                trunc_ratio = 0.0
            else:
                # 随机选一个截断比例
                trunc_ratio = rng.choice(truncation_ratios)
                probe_text = _truncate_text(seed_text, trunc_ratio)

            # 确定 source language
            source_lang = "mixed"
            if cat_enum in (
                ProbeCategory.CHINESE_NATURAL,
            ):
                source_lang = "chinese"
            elif cat_enum in (
                ProbeCategory.ENGLISH_NATURAL,
            ):
                source_lang = "english"
            elif cat_enum == ProbeCategory.CODE:
                source_lang = "code"

            probes.append(
                Probe(
                    id=f"{cat_name}_{_make_id()}",
                    text=probe_text,
                    category=cat_name,
                    truncation_ratio=trunc_ratio,
                    source_lang=source_lang,
                )
            )

    rng.shuffle(probes)
    return probes


def save_probes(probes: list[Probe], path: Path):
    """保存 probe 集合到 JSON"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "id": p.id,
            "text": p.text,
            "category": p.category,
            "truncation_ratio": p.truncation_ratio,
            "source_lang": p.source_lang,
            "metadata": p.metadata,
        }
        for p in probes
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_probes(path: Path) -> list[Probe]:
    """从 JSON 加载 probe 集合"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Probe(
            id=d["id"],
            text=d["text"],
            category=d["category"],
            truncation_ratio=d.get("truncation_ratio", 0.0),
            source_lang=d.get("source_lang", "mixed"),
            metadata=d.get("metadata", {}),
        )
        for d in data
    ]
