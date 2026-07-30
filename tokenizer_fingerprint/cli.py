"""
cli.py — 命令行入口

Commands:
  generate-probes   生成 probe 集合
  build-reference   构建参考库
  detect            检测目标模型
  compare           比较两个模型指纹
  compare-bank      对参考库内模型做离线两两比较
"""

from __future__ import annotations

import asyncio
import copy
import csv
import json
import logging
import os
import sys
from itertools import combinations
from pathlib import Path

import click
import yaml

from .probe_generator import generate_probes, save_probes, load_probes
from .query_engine import APIConfig
from .reference_bank import ReferenceBank
from .detector import (
    TokenizerFingerprintDetector,
    build_reference_fingerprint,
)
from .similarity import compute_similarity, compute_similarity_breakdown
from .schema import ModelFingerprint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_excluded_probe_ids(path: str | None) -> set[str]:
    if not path:
        return set()

    exclude_path = Path(path)
    if not exclude_path.exists():
        raise click.ClickException(f"exclude probe file not found: {exclude_path}")

    data = json.loads(exclude_path.read_text(encoding="utf-8"))
    ids: set[str] = set()

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                ids.add(item)
            elif isinstance(item, dict) and "id" in item:
                ids.add(str(item["id"]))
    elif isinstance(data, dict):
        if "blacklist" in data and isinstance(data["blacklist"], list):
            for item in data["blacklist"]:
                if isinstance(item, str):
                    ids.add(item)
                elif isinstance(item, dict) and "probe_id" in item:
                    ids.add(str(item["probe_id"]))
                elif isinstance(item, dict) and "id" in item:
                    ids.add(str(item["id"]))

    if not ids:
        raise click.ClickException(
            f"no probe ids found in exclude probe file: {exclude_path}"
        )
    return ids


def _filter_probes(probes, excluded_probe_ids: set[str]):
    if not excluded_probe_ids:
        return probes
    return [probe for probe in probes if probe.id not in excluded_probe_ids]


def _filter_fingerprint(fp: ModelFingerprint, excluded_probe_ids: set[str]) -> ModelFingerprint:
    if not excluded_probe_ids:
        return fp

    filtered = copy.deepcopy(fp)
    filtered.raw_results = [
        result for result in filtered.raw_results
        if result.probe_id not in excluded_probe_ids
    ]
    filtered.n_probes = len({result.probe_id for result in filtered.raw_results})
    filtered.metadata = dict(filtered.metadata)
    filtered.metadata["excluded_probe_count"] = len(excluded_probe_ids)
    return filtered


def _filter_bank(bank: ReferenceBank, excluded_probe_ids: set[str]) -> ReferenceBank:
    if not excluded_probe_ids:
        return bank

    filtered_bank = ReferenceBank(bank.base_dir)
    for fp in bank.all_fingerprints():
        filtered_bank.add(_filter_fingerprint(fp, excluded_probe_ids))
    return filtered_bank


@click.group()
def cli():
    """Tokenizer Fingerprint Detection System"""
    pass


# ── generate-probes ─────────────────────────────────────────────

@cli.command("generate-probes")
@click.option("--count", default=500, help="Number of probes to generate")
@click.option("--output", default="probes/default_probes.json", help="Output path")
@click.option("--seed", default=42, help="Random seed")
def cmd_generate_probes(count, output, seed):
    """生成 probe prompt 集合"""
    probes = generate_probes(total_count=count, seed=seed)
    save_probes(probes, Path(output))

    # 统计
    from collections import Counter
    cat_counts = Counter(p.category for p in probes)
    click.echo(f"Generated {len(probes)} probes → {output}")
    for cat, cnt in sorted(cat_counts.items()):
        click.echo(f"  {cat}: {cnt}")


# ── build-reference ─────────────────────────────────────────────

@cli.command("build-reference")
@click.option("--config", required=True, help="Config YAML path")
@click.option("--probes", default=None, help="Probe file path (auto-generate if not provided)")
@click.option("--output", default="reference_bank/", help="Output directory")
@click.option("--concurrency", default=5, help="Concurrent API requests")
@click.option("--target", default=None, help="Reference model name to build (from config)")
def cmd_build_reference(config, probes, output, concurrency, target):
    """构建参考模型指纹库"""
    cfg = _load_config(config)

    # 加载或生成 probes
    if probes:
        probe_path = Path(probes)
        if not probe_path.exists():
            click.echo(f"Error: probe file not found: {probe_path}")
            return
        probe_list = load_probes(probe_path)
    else:
        probe_cfg = cfg.get("probes", {})
        probe_list = generate_probes(
            total_count=probe_cfg.get("total_count", 500),
            category_weights=probe_cfg.get("categories"),
            truncation_ratios=probe_cfg.get("truncation_ratios"),
        )
        probe_path = Path(output) / "probes_used.json"
        save_probes(probe_list, probe_path)
        click.echo(f"Generated {len(probe_list)} probes → {probe_path}")

    output_dir = Path(output)
    if target and (output_dir / "index.json").exists():
        bank = ReferenceBank.load(output_dir)
    else:
        bank = ReferenceBank(output_dir)
    # 遍历参考模型
    ref_models = cfg.get("reference_models", [])
    if target:
        ref_models = [m for m in ref_models if m["name"] == target]
        if not ref_models:
            click.echo(f"Error: reference model not found in config: {target}")
            return

    for model_cfg in ref_models:
        name = model_cfg["name"]
        family = model_cfg["family"]
        provider = model_cfg.get("provider", "openai")
        api_cfg = APIConfig.from_dict(
            model_cfg["api_config"],
            provider,
            defaults=cfg.get("query_protocol"),
        )

        if not api_cfg.api_key:
            click.echo(f"⚠ Skipping {name}: no API key configured")
            continue

        click.echo(f"Building fingerprint for {name} (family={family})...")
        try:
            safe_name = name.replace("/", "_").replace(":", "_")
            raw_results_path = Path(output) / "raw_tokens" / f"{safe_name}.jsonl"
            click.echo(f"  streaming raw tokens → {raw_results_path}")
            fp = asyncio.run(
                build_reference_fingerprint(
                    model_config=api_cfg,
                    model_name=name,
                    family=family,
                    probes=probe_list,
                    concurrency=concurrency,
                    raw_results_path=raw_results_path,
                )
            )
            bank.add(fp)
            click.echo(f"  ✓ {name}: {fp.n_probes} probes, types={list(fp.type_feat.type_distribution.keys())}")
        except Exception as e:
            click.echo(f"  ✗ {name}: {e}")

    bank.save()
    click.echo(f"\nReference bank saved to {output} ({len(bank.list_models())} models)")


# ── detect ──────────────────────────────────────────────────────

@cli.command("detect")
@click.option("--config", required=True, help="Config YAML path")
@click.option("--target", default=None, help="Target model name (from config)")
@click.option("--reference", default="reference_bank/", help="Reference bank directory")
@click.option("--probes", default=None, help="Probe file path")
@click.option("--exclude-probes", default=None, help="JSON file of probe ids to exclude from BCS")
@click.option("--bank-stats", "bank_stats_path", default=None, help="compare-bank JSON for cross-family statistics")
@click.option("--output", default="results/", help="Output directory")
@click.option("--concurrency", default=5, help="Concurrent API requests")
def cmd_detect(config, target, reference, probes, exclude_probes, bank_stats_path, output, concurrency):
    """检测目标模型"""
    cfg = _load_config(config)
    excluded_probe_ids = _load_excluded_probe_ids(exclude_probes)

    # 加载参考库（可选 bank_compare 统计量）
    bank = ReferenceBank.load(Path(reference), bank_compare_path=bank_stats_path)
    if not bank.list_models():
        click.echo("Error: Reference bank is empty. Run build-reference first.")
        return
    bank = _filter_bank(bank, excluded_probe_ids)

    # 加载 probes
    if probes:
        probe_path = Path(probes)
        if not probe_path.exists():
            click.echo(f"Error: probe file not found: {probe_path}")
            return
        probe_list = load_probes(probe_path)
    else:
        # 尝试从参考库目录加载
        ref_probes = Path(reference) / "probes_used.json"
        if ref_probes.exists():
            probe_list = load_probes(ref_probes)
        else:
            probe_list = generate_probes(total_count=500)
    original_probe_count = len(probe_list)
    probe_list = _filter_probes(probe_list, excluded_probe_ids)
    if excluded_probe_ids:
        click.echo(
            f"Using probe blacklist: excluded {original_probe_count - len(probe_list)} "
            f"probes, remaining {len(probe_list)}"
        )
    if not probe_list:
        click.echo("Error: no probes remain after applying exclude-probes.")
        return

    # 配置
    scoring_cfg = cfg.get("scoring", {})
    weights = scoring_cfg.get("weights")
    thresholds = scoring_cfg.get("thresholds")

    detector = TokenizerFingerprintDetector(bank, weights=weights, thresholds=thresholds)

    # 获取目标模型配置
    target_models = cfg.get("target_models", [])
    if target:
        target_models = [m for m in target_models if m["name"] == target]

    if not target_models:
        click.echo("Error: No target models configured or specified.")
        return

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_cfg in target_models:
        name = model_cfg["name"]
        provider = model_cfg.get("provider", "openai")
        api_cfg = APIConfig.from_dict(
            model_cfg["api_config"],
            provider,
            defaults=cfg.get("query_protocol"),
        )

        if not api_cfg.api_key:
            click.echo(f"⚠ Skipping {name}: no API key configured")
            continue

        click.echo(f"\n{'='*60}")
        click.echo(f"Detecting: {name}")
        click.echo(f"{'='*60}")

        try:
            safe_name = name.replace("/", "_").replace(":", "_")
            raw_results_path = output_dir / "raw_tokens" / f"{safe_name}.jsonl"
            click.echo(f"Streaming raw tokens to {raw_results_path}")
            result = detector.detect(
                target_config=api_cfg,
                target_name=name,
                probes=probe_list,
                concurrency=concurrency,
                raw_results_path=raw_results_path,
            )

            # 输出结果
            _print_result(result)

            # 保存结果
            result_path = output_dir / f"{safe_name}_result.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            click.echo(f"\nResult saved to {result_path}")

            if result.target_fingerprint:
                fp_path = output_dir / f"{safe_name}_fingerprint.json"
                result.target_fingerprint.save(fp_path, include_raw_results=True)
                click.echo(f"Fingerprint saved to {fp_path}")

        except Exception as e:
            click.echo(f"Error detecting {name}: {e}")
            import traceback
            traceback.print_exc()


def _print_result(result):
    """格式化输出检测结果"""
    label_descriptions = {
        "same_source": "是，同源",
        "not_same_source": "否，非同源",
    }

    click.echo(f"\n  判定: {label_descriptions.get(result.label, result.label)}")
    click.echo(f"  置信度: {result.confidence:.4f}")

    if result.same_source_of:
        top_match = result.top_matches[0] if result.top_matches else None
        if top_match:
            family = top_match.get("family", "?")
            click.echo(f"  同源模型: {result.same_source_of} (family={family})")

    if result.diagnosis:
        click.echo(f"  诊断: {result.diagnosis}")

    if result.evidence:
        ev = result.evidence
        if ev.get("z_score_vs_cross_family", 0) != 0:
            click.echo(f"  证据: z={ev['z_score_vs_cross_family']}, margin_sigma={ev['margin_sigma']}, family={ev.get('family_consistency', '?')}")


# ── compare ─────────────────────────────────────────────────────

@cli.command("compare")
@click.argument("fp1_path")
@click.argument("fp2_path")
@click.option("--exclude-probes", default=None, help="JSON file of probe ids to exclude from BCS")
def cmd_compare(fp1_path, fp2_path, exclude_probes):
    """比较两个模型指纹的相似度"""
    excluded_probe_ids = _load_excluded_probe_ids(exclude_probes)
    fp1 = _filter_fingerprint(ModelFingerprint.load(Path(fp1_path)), excluded_probe_ids)
    fp2 = _filter_fingerprint(ModelFingerprint.load(Path(fp2_path)), excluded_probe_ids)

    score = compute_similarity(fp1, fp2)
    breakdown = compute_similarity_breakdown(fp1, fp2)

    click.echo(f"\n比较: {fp1.model_name} vs {fp2.model_name}")
    click.echo(f"  家族: {fp1.family} vs {fp2.family}")
    click.echo(f"\n  BCS 边界一致性分数: {score:.4f}")
    click.echo(f"\n  分项:")
    click.echo(f"    共同 probe 数: {breakdown['common_probe_count']}")
    click.echo(f"    边界签名完全一致率: {breakdown['boundary_exact_match_rate']:.4f}")
    click.echo(f"    字符长度一致率: {breakdown['char_length_match_rate']:.4f}")
    click.echo(f"    字节长度一致率: {breakdown['byte_length_match_rate']:.4f}")
    click.echo(f"    Token 类型一致率: {breakdown['token_type_match_rate']:.4f}")
    click.echo(f"    前缀边界一致率: {breakdown['prefix_match_rate']:.4f}")
    click.echo(f"    空输出状态一致率: {breakdown['empty_match_rate']:.4f}")


# ── compare-bank ────────────────────────────────────────────────

@cli.command("compare-bank")
@click.option("--reference", default="reference_bank/", help="Reference bank directory")
@click.option("--output", default="bank_compare.json", help="Output JSON path")
@click.option("--csv-output", default=None, help="Optional output CSV path")
@click.option("--top-k", default=5, help="Print top-k nearest neighbors per model")
@click.option("--exclude-probes", default=None, help="JSON file of probe ids to exclude from BCS")
def cmd_compare_bank(reference, output, csv_output, top_k, exclude_probes):
    """只在参考库内部做离线两两 BCS 对比，不调用模型 API。"""
    excluded_probe_ids = _load_excluded_probe_ids(exclude_probes)
    bank = ReferenceBank.load(Path(reference))
    bank = _filter_bank(bank, excluded_probe_ids)
    fps = bank.all_fingerprints()
    if len(fps) < 2:
        click.echo("Error: reference bank must contain at least two fingerprints.")
        return
    same_source_threshold = 0.50
    margin_threshold = 0.08

    pairs = []
    neighbors: dict[str, list[dict]] = {fp.model_name: [] for fp in fps}

    for fp1, fp2 in combinations(fps, 2):
        score = compute_similarity(fp1, fp2)
        breakdown = compute_similarity_breakdown(fp1, fp2)
        row = {
            "model_a": fp1.model_name,
            "family_a": fp1.family,
            "model_b": fp2.model_name,
            "family_b": fp2.family,
            "score": score,
            "bcs": score,
            "common_probe_count": breakdown["common_probe_count"],
            "char_length_match_rate": breakdown["char_length_match_rate"],
            "byte_length_match_rate": breakdown["byte_length_match_rate"],
            "token_type_match_rate": breakdown["token_type_match_rate"],
            "prefix_match_rate": breakdown["prefix_match_rate"],
            "empty_match_rate": breakdown["empty_match_rate"],
        }
        pairs.append(row)
        neighbors[fp1.model_name].append({
            "model": fp2.model_name,
            "family": fp2.family,
            "score": score,
        })
        neighbors[fp2.model_name].append({
            "model": fp1.model_name,
            "family": fp1.family,
            "score": score,
        })

    for values in neighbors.values():
        values.sort(key=lambda x: x["score"], reverse=True)

    result = {
        "reference": str(reference),
        "excluded_probe_count": len(excluded_probe_ids),
        "n_models": len(fps),
        "n_pairs": len(pairs),
        "models": [
            {
                "model": fp.model_name,
                "family": fp.family,
                "n_probes": fp.n_probes,
                "top1_model": (neighbors[fp.model_name][0]["model"] if neighbors[fp.model_name] else None),
                "top1_score": (neighbors[fp.model_name][0]["score"] if neighbors[fp.model_name] else 0.0),
                "top1_minus_top2": (
                    neighbors[fp.model_name][0]["score"] - neighbors[fp.model_name][1]["score"]
                    if len(neighbors[fp.model_name]) > 1 else 0.0
                ),
                "same_source": (
                    len(neighbors[fp.model_name]) > 1
                    and neighbors[fp.model_name][0]["score"] >= same_source_threshold
                    and (neighbors[fp.model_name][0]["score"] - neighbors[fp.model_name][1]["score"]) >= margin_threshold
                ),
            }
            for fp in fps
        ],
        "pairs": sorted(pairs, key=lambda x: x["score"], reverse=True),
        "nearest_neighbors": {
            model: values[:top_k]
            for model, values in neighbors.items()
        },
        "same_source_rule": {
            "top1_score_threshold": same_source_threshold,
            "top1_minus_top2_threshold": margin_threshold,
        },
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if csv_output:
        csv_path = Path(csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
            writer.writeheader()
            writer.writerows(result["pairs"])

    click.echo(
        f"Compared {len(fps)} models, {len(pairs)} pairs. "
        f"JSON saved to {output_path}"
    )
    if csv_output:
        click.echo(f"CSV saved to {csv_output}")
    if excluded_probe_ids:
        click.echo(f"Excluded probes: {len(excluded_probe_ids)}")

    click.echo("\nTop nearest neighbors:")
    for fp in fps:
        click.echo(f"\n{fp.model_name} (family={fp.family})")
        for item in neighbors[fp.model_name][:top_k]:
            click.echo(
                f"  {item['model']} (family={item['family']}) "
                f"score={item['score']:.4f}"
            )

def main():
    cli()


if __name__ == "__main__":
    main()
