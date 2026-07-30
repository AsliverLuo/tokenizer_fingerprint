#!/usr/bin/env python3
"""Build family-level tokenizer baseline registry from local tokenizer files.

The script consumes an explicit manifest, compares tokenizer vocabularies and
sample encodings, clusters equivalent/near-equivalent tokenizer lineages within
each family, and selects a representative baseline for each lineage.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml


TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "config.json",
]

COMPARABLE_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
}

DEFAULT_ENCODING_SAMPLES = [
    "Hello world",
    "The tokenizer reveals ",
    "A leading-space boundary",
    "你好，世界",
    "中文 English mixed 123",
    "def foo(x): return x + 1",
    '{"model": "qwen", "temperature":',
    "https://example.com/path?a=1",
]

ALLOWED_ROLES = {"base", "pretrain", "instruct", "chat", "other"}
ALLOWED_SOURCES = {"hf", "tiktoken"}
ROLE_PRIORITY = {
    "base": 0,
    "pretrain": 0,
    "instruct": 1,
    "chat": 2,
    "other": 3,
}


@dataclass(frozen=True)
class Thresholds:
    near_token_jaccard: float = 0.995
    near_id_consistency: float = 0.995
    near_encoding_rate: float = 0.99


@dataclass(frozen=True)
class ManifestEntry:
    name: str
    family: str
    model_id: str
    source: str = "hf"
    local_dir: str = ""
    encoding: str = ""
    version: str = ""
    role: str = "other"
    preferred_rank: int = 1000
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build tokenizer family baseline registry from local files."
    )
    parser.add_argument(
        "--manifest",
        default="tokenizer_baselines/models_manifest.yaml",
        help="YAML manifest describing tokenizer models.",
    )
    parser.add_argument(
        "--output-dir",
        default="tokenizer_baselines",
        help="Directory for registry, pairwise comparison, and report outputs.",
    )
    parser.add_argument(
        "--encoding-samples",
        default="probes/default_probes.json",
        help="JSON file containing sample strings or probe objects with text fields.",
    )
    parser.add_argument(
        "--max-encoding-samples",
        type=int,
        default=128,
        help="Maximum unique sample texts to encode for behavior comparison.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer.",
    )
    parser.add_argument(
        "--near-token-jaccard",
        type=float,
        default=Thresholds.near_token_jaccard,
        help="Token-set Jaccard threshold for near-equivalent tokenizer lineages.",
    )
    parser.add_argument(
        "--near-id-consistency",
        type=float,
        default=Thresholds.near_id_consistency,
        help="Shared-token id consistency threshold for near-equivalent tokenizer lineages.",
    )
    parser.add_argument(
        "--near-encoding-rate",
        type=float,
        default=Thresholds.near_encoding_rate,
        help="Sample encoding match-rate threshold for near-equivalent tokenizer lineages.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = Thresholds(
        near_token_jaccard=args.near_token_jaccard,
        near_id_consistency=args.near_id_consistency,
        near_encoding_rate=args.near_encoding_rate,
    )

    entries = load_manifest(Path(args.manifest))
    samples = load_encoding_samples(Path(args.encoding_samples), args.max_encoding_samples)
    models = load_tokenizer_models(entries, samples, trust_remote_code=args.trust_remote_code)
    model_order = [entry.name for entry in entries]
    pairs = compare_pairs(models, model_order, samples, thresholds)
    registry = build_baseline_registry(
        models=models,
        model_order=model_order,
        pair_rows=pairs,
        thresholds=thresholds,
        manifest_path=Path(args.manifest),
        encoding_sample_count=len(samples),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "baseline_registry.json", registry)
    write_json(
        output_dir / "pairwise_tokenizer_compare.json",
        {
            "manifest": str(Path(args.manifest)),
            "thresholds": asdict(thresholds),
            "n_models": len(model_order),
            "n_pairs": len(pairs),
            "pairs": pairs,
        },
    )
    write_csv(output_dir / "pairwise_tokenizer_compare.csv", pairs)
    write_report(output_dir / "TOKENIZER_BASELINE_REPORT.md", registry, pairs)

    print(f"Wrote {output_dir / 'baseline_registry.json'}")
    print(f"Wrote {output_dir / 'pairwise_tokenizer_compare.json'}")
    print(f"Wrote {output_dir / 'pairwise_tokenizer_compare.csv'}")
    print(f"Wrote {output_dir / 'TOKENIZER_BASELINE_REPORT.md'}")
    return 0


def load_manifest(path: Path) -> list[ManifestEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("Manifest must contain a non-empty 'models' list.")

    entries: list[ManifestEntry] = []
    for idx, item in enumerate(raw_models, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest model entry #{idx} must be a mapping.")
        source = str(item.get("source", "hf"))
        if source not in ALLOWED_SOURCES:
            raise ValueError(
                f"Manifest model {item.get('name', idx)} has source={source!r}; "
                f"allowed sources are {sorted(ALLOWED_SOURCES)}"
            )
        required = ["name", "family", "model_id"]
        if source == "hf":
            required.append("local_dir")
        else:
            required.append("encoding")
        missing = [key for key in required if not item.get(key)]
        if missing:
            raise ValueError(f"Manifest model entry #{idx} missing required fields: {missing}")
        role = str(item.get("role", "other"))
        if role not in ALLOWED_ROLES:
            raise ValueError(
                f"Manifest model {item['name']} has role={role!r}; "
                f"allowed roles are {sorted(ALLOWED_ROLES)}"
            )
        entries.append(
            ManifestEntry(
                name=str(item["name"]),
                family=str(item["family"]),
                model_id=str(item["model_id"]),
                source=source,
                local_dir=str(item.get("local_dir", "")),
                encoding=str(item.get("encoding", "")),
                version=str(item.get("version", "")),
                role=role,
                preferred_rank=int(item.get("preferred_rank", 1000)),
                notes=str(item.get("notes", "")),
            )
        )

    duplicate_names = duplicates(entry.name for entry in entries)
    duplicate_model_ids = duplicates(entry.model_id for entry in entries)
    if duplicate_names:
        raise ValueError(f"Duplicate manifest names: {duplicate_names}")
    if duplicate_model_ids:
        raise ValueError(f"Duplicate manifest model_id values: {duplicate_model_ids}")

    for entry in entries:
        if entry.source == "hf":
            local_dir = Path(entry.local_dir)
            if not local_dir.exists():
                raise FileNotFoundError(f"Tokenizer directory not found for {entry.name}: {local_dir}")
            files = {path.name for path in local_dir.iterdir() if path.is_file()}
            if not (files & {"tokenizer.json", "tokenizer.model", "vocab.json"}):
                raise FileNotFoundError(
                    f"Tokenizer directory for {entry.name} lacks tokenizer.json/tokenizer.model/vocab.json: "
                    f"{local_dir}"
                )

    return entries


def duplicates(values: Any) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def load_encoding_samples(path: Path, max_samples: int) -> list[str]:
    samples: list[str] = []
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    samples.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    samples.append(item["text"])
        else:
            raise ValueError(f"Encoding sample file must contain a JSON list: {path}")
    else:
        samples = list(DEFAULT_ENCODING_SAMPLES)

    if not samples:
        samples = list(DEFAULT_ENCODING_SAMPLES)
    return unique_texts(samples, max_samples)


def unique_texts(values: list[str], max_items: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= max_items:
            break
    return result


def load_tokenizer_models(
    entries: list[ManifestEntry],
    samples: list[str],
    trust_remote_code: bool = False,
) -> dict[str, dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required. Install with: pip install transformers tokenizers sentencepiece"
        ) from exc

    models: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if entry.source == "tiktoken":
            models[entry.name] = load_tiktoken_model(entry, samples)
            continue
        tokenizer = AutoTokenizer.from_pretrained(
            entry.local_dir,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        vocab = tokenizer.get_vocab()
        special_tokens_map = json_safe(dict(tokenizer.special_tokens_map))
        encoding_samples = {
            text: tokenizer.encode(text, add_special_tokens=False)
            for text in samples
        }
        file_hashes = hash_tokenizer_files(Path(entry.local_dir))
        models[entry.name] = {
            "name": entry.name,
            "family": entry.family,
            "model_id": entry.model_id,
            "source": entry.source,
            "local_dir": entry.local_dir,
            "encoding": entry.encoding,
            "version": entry.version,
            "role": entry.role,
            "preferred_rank": entry.preferred_rank,
            "notes": entry.notes,
            "vocab": vocab,
            "vocab_size": len(vocab),
            "vocab_sha256": sha256_json(vocab),
            "special_tokens_map": special_tokens_map,
            "special_tokens_map_sha256": sha256_json(special_tokens_map),
            "file_hashes": file_hashes,
            "comparable_file_hashes": comparable_file_hashes(file_hashes),
            "encoding_samples": encoding_samples,
        }
    return models


def load_tiktoken_model(entry: ManifestEntry, samples: list[str]) -> dict[str, Any]:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("tiktoken is required for source=tiktoken manifest entries.") from exc

    encoding = tiktoken.get_encoding(entry.encoding)
    vocab = {
        encode_tiktoken_token(token_bytes): int(token_id)
        for token_bytes, token_id in encoding._mergeable_ranks.items()
    }
    special_tokens_map = {
        str(token): int(token_id)
        for token, token_id in encoding._special_tokens.items()
    }
    for token, token_id in special_tokens_map.items():
        vocab[f"special:{token}"] = token_id
    encoding_samples = {
        text: encoding.encode(text, disallowed_special=())
        for text in samples
    }
    synthetic_file_hashes = {
        f"{entry.encoding}.tiktoken": sha256_json(
            {
                "encoding": entry.encoding,
                "n_vocab": encoding.n_vocab,
                "mergeable_ranks": vocab,
                "special_tokens": special_tokens_map,
            }
        )
    }
    return {
        "name": entry.name,
        "family": entry.family,
        "model_id": entry.model_id,
        "source": entry.source,
        "local_dir": entry.local_dir,
        "encoding": entry.encoding,
        "version": entry.version,
        "role": entry.role,
        "preferred_rank": entry.preferred_rank,
        "notes": entry.notes,
        "vocab": vocab,
        "vocab_size": len(vocab),
        "vocab_sha256": sha256_json(vocab),
        "special_tokens_map": special_tokens_map,
        "special_tokens_map_sha256": sha256_json(special_tokens_map),
        "file_hashes": synthetic_file_hashes,
        "comparable_file_hashes": synthetic_file_hashes,
        "encoding_samples": encoding_samples,
    }


def encode_tiktoken_token(token_bytes: bytes) -> str:
    return "b64:" + base64.b64encode(token_bytes).decode("ascii")


def compare_pairs(
    models: dict[str, dict[str, Any]],
    model_order: list[str],
    samples: list[str],
    thresholds: Thresholds,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name_a, name_b in combinations(model_order, 2):
        row = compare_model_pair(models[name_a], models[name_b], samples, thresholds)
        rows.append(row)
    return rows


def compare_model_pair(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    samples: list[str],
    thresholds: Thresholds,
) -> dict[str, Any]:
    vocab_a = model_a["vocab"]
    vocab_b = model_b["vocab"]
    set_a = set(vocab_a)
    set_b = set(vocab_b)
    shared = set_a & set_b
    union = set_a | set_b
    id_match_count = sum(1 for token in shared if vocab_a[token] == vocab_b[token])
    id_mismatch_count = len(shared) - id_match_count
    encoding_same_count = sum(
        1
        for text in samples
        if model_a["encoding_samples"][text] == model_b["encoding_samples"][text]
    )
    encoding_sample_count = len(samples)
    token_jaccard = len(shared) / len(union) if union else 0.0
    id_consistency = id_match_count / len(shared) if shared else 0.0
    encoding_same_rate = (
        encoding_same_count / encoding_sample_count
        if encoding_sample_count
        else 0.0
    )
    exact_token_to_id_same = vocab_a == vocab_b
    same_special_tokens = model_a["special_tokens_map"] == model_b["special_tokens_map"]
    same_tokenizer_file_hashes = (
        model_a["comparable_file_hashes"] == model_b["comparable_file_hashes"]
    )
    similarity_score = tokenizer_similarity_score(
        token_jaccard=token_jaccard,
        id_consistency=id_consistency,
        encoding_same_rate=encoding_same_rate,
        same_special_tokens=same_special_tokens,
    )
    equivalence_label = classify_equivalence(
        exact_token_to_id_same=exact_token_to_id_same,
        encoding_same_rate=encoding_same_rate,
        token_jaccard=token_jaccard,
        id_consistency=id_consistency,
        thresholds=thresholds,
    )
    return {
        "model_a": model_a["name"],
        "family_a": model_a["family"],
        "model_b": model_b["name"],
        "family_b": model_b["family"],
        "same_family": model_a["family"] == model_b["family"],
        "exact_token_to_id_same": exact_token_to_id_same,
        "same_token_set": set_a == set_b,
        "same_vocab_size": len(vocab_a) == len(vocab_b),
        "same_special_tokens": same_special_tokens,
        "same_tokenizer_file_hashes": same_tokenizer_file_hashes,
        "vocab_size_a": len(vocab_a),
        "vocab_size_b": len(vocab_b),
        "shared_token_count": len(shared),
        "union_token_count": len(union),
        "token_jaccard": token_jaccard,
        "id_match_count": id_match_count,
        "id_mismatch_count": id_mismatch_count,
        "id_consistency": id_consistency,
        "only_a_count": len(set_a - set_b),
        "only_b_count": len(set_b - set_a),
        "encoding_same_count": encoding_same_count,
        "encoding_sample_count": encoding_sample_count,
        "encoding_same_rate": encoding_same_rate,
        "all_sample_encodings_same": encoding_same_count == encoding_sample_count,
        "tokenizer_similarity_score": similarity_score,
        "equivalence_label": equivalence_label,
    }


def tokenizer_similarity_score(
    token_jaccard: float,
    id_consistency: float,
    encoding_same_rate: float,
    same_special_tokens: bool,
) -> float:
    return (
        0.45 * token_jaccard
        + 0.35 * id_consistency
        + 0.15 * encoding_same_rate
        + 0.05 * (1.0 if same_special_tokens else 0.0)
    )


def classify_equivalence(
    exact_token_to_id_same: bool,
    encoding_same_rate: float,
    token_jaccard: float,
    id_consistency: float,
    thresholds: Thresholds,
) -> str:
    if exact_token_to_id_same and encoding_same_rate == 1.0:
        return "hard_equivalent"
    if (
        token_jaccard >= thresholds.near_token_jaccard
        and id_consistency >= thresholds.near_id_consistency
        and encoding_same_rate >= thresholds.near_encoding_rate
    ):
        return "near_equivalent"
    return "distinct"


def build_baseline_registry(
    models: dict[str, dict[str, Any]],
    model_order: list[str],
    pair_rows: list[dict[str, Any]],
    thresholds: Thresholds,
    manifest_path: Path,
    encoding_sample_count: int,
) -> dict[str, Any]:
    pair_lookup = {frozenset((row["model_a"], row["model_b"])): row for row in pair_rows}
    families: dict[str, list[str]] = defaultdict(list)
    for name in model_order:
        families[models[name]["family"]].append(name)

    family_entries: dict[str, dict[str, Any]] = {}
    baselines: list[dict[str, Any]] = []
    for family in sorted(families):
        family_models = families[family]
        components = cluster_family_models(family_models, pair_lookup)
        lineages = []
        for members in components:
            representative = select_representative(members, models, pair_lookup)
            lineages.append(
                {
                    "members": members,
                    "representative": representative,
                    "metrics": lineage_metrics(members, pair_lookup),
                }
            )

        primary_lineage = min(
            lineages,
            key=lambda lineage: (
                -len(lineage["members"]),
                role_priority(models[lineage["representative"]]),
                models[lineage["representative"]]["preferred_rank"],
                lineage["representative"],
            ),
        )
        ordered_lineages = [primary_lineage] + sorted(
            [lineage for lineage in lineages if lineage is not primary_lineage],
            key=lambda lineage: (
                -len(lineage["members"]),
                role_priority(models[lineage["representative"]]),
                models[lineage["representative"]]["preferred_rank"],
                lineage["representative"],
            ),
        )

        family_slug = slugify(family)
        baseline_ids: list[str] = []
        variant_idx = 1
        primary_baseline_id = ""
        for lineage in ordered_lineages:
            is_primary = lineage is primary_lineage
            baseline_id = (
                f"{family_slug}__primary"
                if is_primary
                else f"{family_slug}__variant_{variant_idx:02d}"
            )
            if not is_primary:
                variant_idx += 1
            baseline_ids.append(baseline_id)
            if is_primary:
                primary_baseline_id = baseline_id

            representative = lineage["representative"]
            rep_data = models[representative]
            baseline = {
                "baseline_id": baseline_id,
                "family": family,
                "kind": "primary" if is_primary else "variant",
                "representative_model": representative,
                "representative_model_id": rep_data["model_id"],
                "representative_source": rep_data["source"],
                "representative_local_dir": rep_data["local_dir"],
                "representative_encoding": rep_data["encoding"],
                "member_models": lineage["members"],
                "selection_reason": selection_reason(
                    is_primary=is_primary,
                    lineage=lineage,
                    family_model_count=len(family_models),
                    rep_data=rep_data,
                ),
                "tokenizer_hashes": rep_data["file_hashes"],
                "vocab_size": rep_data["vocab_size"],
                "vocab_sha256": rep_data["vocab_sha256"],
                "special_tokens_map_sha256": rep_data["special_tokens_map_sha256"],
                "lineage_metrics": lineage["metrics"],
            }
            baselines.append(baseline)

        family_entries[family] = {
            "primary_baseline_id": primary_baseline_id,
            "baseline_ids": baseline_ids,
            "model_count": len(family_models),
            "lineage_count": len(ordered_lineages),
        }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest": str(manifest_path),
        "thresholds": asdict(thresholds),
        "encoding_sample_count": encoding_sample_count,
        "n_models": len(model_order),
        "n_families": len(family_entries),
        "n_baselines": len(baselines),
        "families": family_entries,
        "baselines": baselines,
        "models": {
            name: public_model_summary(models[name])
            for name in model_order
        },
    }


def cluster_family_models(
    family_models: list[str],
    pair_lookup: dict[frozenset[str], dict[str, Any]],
) -> list[list[str]]:
    parent = {name: name for name in family_models}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(a: str, b: str) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for model_a, model_b in combinations(family_models, 2):
        row = pair_lookup.get(frozenset((model_a, model_b)))
        if row and row["equivalence_label"] != "distinct":
            union(model_a, model_b)

    groups: dict[str, list[str]] = defaultdict(list)
    for name in family_models:
        groups[find(name)].append(name)
    return sorted((sorted(values) for values in groups.values()), key=lambda values: values[0])


def select_representative(
    members: list[str],
    models: dict[str, dict[str, Any]],
    pair_lookup: dict[frozenset[str], dict[str, Any]],
) -> str:
    scored = []
    for name in members:
        if len(members) == 1:
            avg_similarity = 1.0
        else:
            total = 0.0
            for other in members:
                if other == name:
                    continue
                row = pair_lookup[frozenset((name, other))]
                total += float(row["tokenizer_similarity_score"])
            avg_similarity = total / (len(members) - 1)
        data = models[name]
        scored.append(
            (
                -avg_similarity,
                role_priority(data),
                int(data["preferred_rank"]),
                name,
            )
        )
    scored.sort()
    return scored[0][3]


def lineage_metrics(
    members: list[str],
    pair_lookup: dict[frozenset[str], dict[str, Any]],
) -> dict[str, Any]:
    if len(members) == 1:
        return {
            "member_count": 1,
            "internal_pair_count": 0,
            "mean_internal_similarity": 1.0,
            "min_internal_similarity": 1.0,
            "max_internal_similarity": 1.0,
            "relation_counts": {},
        }

    scores = []
    relation_counts: Counter[str] = Counter()
    for model_a, model_b in combinations(members, 2):
        row = pair_lookup[frozenset((model_a, model_b))]
        scores.append(float(row["tokenizer_similarity_score"]))
        relation_counts[str(row["equivalence_label"])] += 1
    return {
        "member_count": len(members),
        "internal_pair_count": len(scores),
        "mean_internal_similarity": sum(scores) / len(scores),
        "min_internal_similarity": min(scores),
        "max_internal_similarity": max(scores),
        "relation_counts": dict(sorted(relation_counts.items())),
    }


def selection_reason(
    is_primary: bool,
    lineage: dict[str, Any],
    family_model_count: int,
    rep_data: dict[str, Any],
) -> str:
    role = rep_data["role"]
    rank = rep_data["preferred_rank"]
    prefix = "family primary" if is_primary else "family variant"
    return (
        f"selected as {prefix}: lineage has {len(lineage['members'])}/"
        f"{family_model_count} family models; representative is the lineage medoid; "
        f"tie-breakers role={role}, preferred_rank={rank}, name={rep_data['name']}"
    )


def public_model_summary(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": model["name"],
        "family": model["family"],
        "model_id": model["model_id"],
        "source": model["source"],
        "local_dir": model["local_dir"],
        "encoding": model["encoding"],
        "version": model["version"],
        "role": model["role"],
        "preferred_rank": model["preferred_rank"],
        "notes": model["notes"],
        "vocab_size": model["vocab_size"],
        "vocab_sha256": model["vocab_sha256"],
        "special_tokens_map_sha256": model["special_tokens_map_sha256"],
        "file_hashes": model["file_hashes"],
    }


def role_priority(model: dict[str, Any]) -> int:
    return ROLE_PRIORITY.get(str(model.get("role", "other")), ROLE_PRIORITY["other"])


def hash_tokenizer_files(model_dir: Path) -> dict[str, str]:
    hashes = {}
    for filename in TOKENIZER_FILES:
        path = model_dir / filename
        if path.exists() and path.is_file():
            hashes[filename] = sha256_bytes(path.read_bytes())
    return hashes


def comparable_file_hashes(file_hashes: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in file_hashes.items()
        if key in COMPARABLE_TOKENIZER_FILES
    }


def write_report(path: Path, registry: dict[str, Any], pairs: list[dict[str, Any]]) -> None:
    lines = [
        "# Tokenizer Baseline Registry Report",
        "",
        "## Summary",
        "",
        f"- Models: {registry['n_models']}",
        f"- Families: {registry['n_families']}",
        f"- Baselines: {registry['n_baselines']}",
        f"- Encoding samples: {registry['encoding_sample_count']}",
        "",
        "Representative selection uses tokenizer-lineage medoids, then role/base priority, "
        "preferred_rank, and model name as tie-breakers.",
        "",
        "## Family Baselines",
        "",
        "| Family | Baseline | Kind | Representative | Members | Mean internal sim |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for baseline in registry["baselines"]:
        metrics = baseline["lineage_metrics"]
        lines.append(
            "| {family} | {baseline_id} | {kind} | {rep} | {members} | {mean:.4f} |".format(
                family=baseline["family"],
                baseline_id=baseline["baseline_id"],
                kind=baseline["kind"],
                rep=baseline["representative_model"],
                members=len(baseline["member_models"]),
                mean=float(metrics["mean_internal_similarity"]),
            )
        )

    lines.extend([
        "",
        "## Lineage Details",
        "",
    ])
    for baseline in registry["baselines"]:
        lines.extend(
            [
                f"### {baseline['baseline_id']}",
                "",
                f"- Family: `{baseline['family']}`",
                f"- Kind: `{baseline['kind']}`",
                f"- Representative: `{baseline['representative_model']}`",
                f"- Model ID: `{baseline['representative_model_id']}`",
                f"- Source: `{baseline['representative_source']}`",
                f"- Encoding: `{baseline['representative_encoding'] or 'n/a'}`",
                f"- Local dir: `{baseline['representative_local_dir'] or 'n/a'}`",
                f"- Members: {', '.join(f'`{name}`' for name in baseline['member_models'])}",
                f"- Reason: {baseline['selection_reason']}",
                "",
            ]
        )

    relation_counts = Counter(row["equivalence_label"] for row in pairs)
    lines.extend([
        "## Pairwise Relation Counts",
        "",
        "| Relation | Count |",
        "| --- | ---: |",
    ])
    for relation, count in sorted(relation_counts.items()):
        lines.append(f"| {relation} | {count} |")

    same_family_pairs = [row for row in pairs if row["same_family"]]
    distinct_same_family = [
        row for row in same_family_pairs
        if row["equivalence_label"] == "distinct"
    ]
    if distinct_same_family:
        lines.extend([
            "",
            "## Same-family Distinct Lineages",
            "",
            "| Model A | Model B | Token Jaccard | ID consistency | Encoding rate |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for row in sorted(
            distinct_same_family,
            key=lambda item: (item["family_a"], item["model_a"], item["model_b"]),
        ):
            lines.append(
                "| {a} | {b} | {j:.4f} | {idc:.4f} | {enc:.4f} |".format(
                    a=row["model_a"],
                    b=row["model_b"],
                    j=float(row["token_jaccard"]),
                    idc=float(row["id_consistency"]),
                    enc=float(row["encoding_same_rate"]),
                )
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256_bytes(data)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return str(value)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "family"


if __name__ == "__main__":
    raise SystemExit(main())
