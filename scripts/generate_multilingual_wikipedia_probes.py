#!/usr/bin/env python3
"""Generate multilingual Wikipedia truncation probes.

The output matches the repository probe JSON format and can be passed to
`tokenizer_fingerprint.cli detect --probes ...`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import regex


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

DEFAULT_LANGUAGES = ("de", "fr", "ar", "zh")


@dataclass(frozen=True)
class LanguageSpec:
    lang: str
    source_lang: str
    category: str
    script_name: str
    span_pattern: str
    min_span_len: int
    max_span_len: int
    min_prefix_len: int
    ratios: tuple[float, ...]
    context_words: int = 6
    context_chars: int = 40
    min_chars: int = 4
    max_chars: int = 140
    min_fragment_chars: int = 12
    sentence_split_pattern: str = r"(?<=[.!?;:])\s+"


LANGUAGE_SPECS: dict[str, LanguageSpec] = {
    "de": LanguageSpec(
        lang="de",
        source_lang="german",
        category="german_partial",
        script_name="Latin",
        span_pattern=r"(?V1)(?<![\p{Latin}\p{M}])[\p{Latin}\p{M}]+(?:[’'-][\p{Latin}\p{M}]+)*(?![\p{Latin}\p{M}])",
        min_span_len=8,
        max_span_len=40,
        min_prefix_len=4,
        ratios=(0.55, 0.65, 0.75),
    ),
    "fr": LanguageSpec(
        lang="fr",
        source_lang="french",
        category="french_partial",
        script_name="Latin",
        span_pattern=r"(?V1)(?<![\p{Latin}\p{M}])[\p{Latin}\p{M}]+(?:[’'-][\p{Latin}\p{M}]+)*(?![\p{Latin}\p{M}])",
        min_span_len=8,
        max_span_len=40,
        min_prefix_len=4,
        ratios=(0.55, 0.65, 0.75),
    ),
    "ar": LanguageSpec(
        lang="ar",
        source_lang="arabic",
        category="arabic_partial",
        script_name="Arabic",
        span_pattern=r"(?V1)(?<![\p{Arabic}\p{M}])[\p{Arabic}\p{M}]+(?![\p{Arabic}\p{M}])",
        min_span_len=5,
        max_span_len=30,
        min_prefix_len=3,
        ratios=(0.55, 0.70, 0.80),
        sentence_split_pattern=r"(?<=[.!?؟؛;:])\s+",
    ),
    "zh": LanguageSpec(
        lang="zh",
        source_lang="chinese",
        category="chinese_partial",
        script_name="Han",
        span_pattern=r"(?V1)\p{Han}+",
        min_span_len=3,
        max_span_len=16,
        min_prefix_len=1,
        ratios=(0.45, 0.60, 0.75),
        context_words=0,
        context_chars=40,
        min_chars=4,
        min_fragment_chars=8,
        sentence_split_pattern=r"(?<=[。！？；：.!?;:])\s*",
    ),
}


@dataclass
class Candidate:
    text: str
    lang: str
    source_lang: str
    category: str
    source_dataset: str
    dataset_config: str
    split: str
    source_row_index: int
    article_id: str
    title: str
    url: str
    source_span: str
    truncation_ratio: float
    cut_length: int
    span_start_in_fragment: int


class ProgressBar:
    def __init__(self, label: str, enabled: bool = True, width: int = 32):
        self.label = label
        self.enabled = enabled
        self.width = width
        self.started_at = time.monotonic()
        self.last_draw_at = 0.0

    def update(
        self,
        rows: int,
        candidates: int,
        unique: int,
        selected: int,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_draw_at < 0.25:
            return
        self.last_draw_at = now
        elapsed = max(1e-6, now - self.started_at)
        rate = rows / elapsed
        bar = "#" * self.width
        sys.stderr.write(
            f"\r{self.label} [{bar}] rows={rows} candidates={candidates} "
            f"unique={unique} selected={selected} {rate:7.1f} rows/s"
        )
        sys.stderr.flush()

    def close(self, rows: int, candidates: int, unique: int, selected: int) -> None:
        if not self.enabled:
            return
        self.update(rows, candidates, unique, selected, force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()


def stable_id(category: str, text: str) -> str:
    digest = hashlib.sha1(f"{category}\0{text}".encode("utf-8")).hexdigest()
    return f"{category}_{digest[:12]}"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = regex.sub(r"[\p{Cc}\p{Cf}&&[^\n\t]]+", " ", text)
    text = regex.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()


def split_fragments(text: str, spec: LanguageSpec) -> Iterator[str]:
    text = normalize_text(text)
    if not text:
        return
    for paragraph in regex.split(r"\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for fragment in regex.split(spec.sentence_split_pattern, paragraph):
            fragment = fragment.strip(" \t")
            if is_usable_fragment(fragment, spec):
                yield fragment


def is_usable_fragment(fragment: str, spec: LanguageSpec) -> bool:
    if len(fragment) < spec.min_fragment_chars:
        return False
    if fragment.count("http://") + fragment.count("https://") + fragment.count("www.") > 1:
        return False
    if regex.search(r"[<>{}\[\]|]{3,}", fragment):
        return False
    script_chars = count_script_chars(fragment, spec.script_name)
    if script_chars < spec.min_span_len:
        return False
    punctuation = sum(1 for char in fragment if unicodedata.category(char).startswith("P"))
    return punctuation / max(1, len(fragment)) <= 0.35


def count_script_chars(text: str, script_name: str) -> int:
    return len(regex.findall(rf"(?V1)\p{{{script_name}}}", text))


def ends_with_script(text: str, script_name: str) -> bool:
    if not text:
        return False
    return bool(regex.fullmatch(rf"(?V1)\p{{{script_name}}}", text[-1]))


def safe_prefix(span: str, cut_length: int, spec: LanguageSpec) -> str:
    prefix = span[:cut_length].rstrip("-'’")
    while prefix and unicodedata.combining(prefix[-1]):
        prefix = prefix[:-1]
    if count_script_chars(prefix, spec.script_name) < spec.min_prefix_len:
        return ""
    if not ends_with_script(prefix, spec.script_name):
        return ""
    return prefix


def left_context_start(fragment: str, span_start: int, spec: LanguageSpec) -> int:
    if spec.context_words <= 0:
        return max(0, span_start - spec.context_chars)
    prior_words = list(regex.finditer(r"\S+", fragment[:span_start]))
    if len(prior_words) <= spec.context_words:
        return 0
    return prior_words[-spec.context_words].start()


def iter_candidates_from_row(
    row: dict[str, Any],
    *,
    row_index: int,
    spec: LanguageSpec,
    source_dataset: str,
    dataset_config: str,
    split: str,
    max_candidates_per_row: int,
) -> Iterator[Candidate]:
    text = row.get("text") or ""
    article_id = str(row.get("id") or "")
    title = str(row.get("title") or "")
    url = str(row.get("url") or "")
    span_re = regex.compile(spec.span_pattern)
    yielded = 0

    for fragment in split_fragments(text, spec):
        for span, span_start in iter_source_spans(fragment, spec, span_re):
            span_script_len = count_script_chars(span, spec.script_name)
            if not spec.min_span_len <= span_script_len <= spec.max_span_len:
                continue

            seen_cuts: set[int] = set()
            for ratio in spec.ratios:
                cut_length = int(len(span) * ratio)
                cut_length = max(spec.min_prefix_len, min(len(span) - 1, cut_length))
                if cut_length in seen_cuts or cut_length >= len(span):
                    continue
                seen_cuts.add(cut_length)

                prefix = safe_prefix(span, cut_length, spec)
                if not prefix:
                    continue
                start = left_context_start(fragment, span_start, spec)
                probe_text = (fragment[start:span_start] + prefix).strip()
                if not spec.min_chars <= len(probe_text) <= spec.max_chars:
                    continue
                if not ends_with_script(probe_text, spec.script_name):
                    continue

                yield Candidate(
                    text=probe_text,
                    lang=spec.lang,
                    source_lang=spec.source_lang,
                    category=spec.category,
                    source_dataset=source_dataset,
                    dataset_config=dataset_config,
                    split=split,
                    source_row_index=row_index,
                    article_id=article_id,
                    title=title,
                    url=url,
                    source_span=span,
                    truncation_ratio=ratio,
                    cut_length=cut_length,
                    span_start_in_fragment=span_start,
                )
                yielded += 1
                if max_candidates_per_row > 0 and yielded >= max_candidates_per_row:
                    return


def iter_source_spans(
    fragment: str,
    spec: LanguageSpec,
    span_re: regex.Pattern,
) -> Iterator[tuple[str, int]]:
    for match in span_re.finditer(fragment):
        span = match.group(0)
        if spec.lang != "zh" or len(span) <= spec.max_span_len:
            yield span, match.start()
            continue

        stride = max(1, spec.max_span_len // 2)
        stop = max(1, len(span) - spec.min_span_len + 1)
        for offset in range(0, stop, stride):
            window = span[offset : offset + spec.max_span_len]
            if len(window) >= spec.min_span_len:
                yield window, match.start() + offset


def iter_wikipedia_rows(
    *,
    dataset: str,
    dataset_config: str,
    split: str,
    cache_dir: Path,
    max_rows: int,
) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "datasets is required to stream Hugging Face Wikipedia data. "
            "Install it with: python -m pip install -r requirements.txt"
        ) from exc

    stream = load_dataset(
        dataset,
        dataset_config,
        split=split,
        streaming=True,
        cache_dir=str(cache_dir),
    )
    for row_index, row in enumerate(stream):
        if row_index >= max_rows:
            return
        yield row_index, dict(row)


def iter_language_candidates(
    *,
    spec: LanguageSpec,
    dataset: str,
    dataset_config: str,
    split: str,
    cache_dir: Path,
    max_rows: int,
    max_candidates_per_row: int,
) -> Iterator[Candidate]:
    for row_index, row in iter_wikipedia_rows(
        dataset=dataset,
        dataset_config=dataset_config,
        split=split,
        cache_dir=cache_dir,
        max_rows=max_rows,
    ):
        yield from iter_candidates_from_row(
            row,
            row_index=row_index,
            spec=spec,
            source_dataset=dataset,
            dataset_config=dataset_config,
            split=split,
            max_candidates_per_row=max_candidates_per_row,
        )


def reservoir_sample(
    candidates: Iterable[Candidate],
    *,
    count: int,
    seed: int,
    progress_label: str,
    progress: bool,
    early_stop_unique_count: int | None,
) -> tuple[list[Candidate], dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[Candidate] = []
    seen_texts: set[str] = set()
    stats: dict[str, Any] = {
        "candidate_count": 0,
        "duplicate_text_count": 0,
        "unique_candidate_count": 0,
        "rows_scanned": 0,
        "early_stopped": False,
    }
    bar = ProgressBar(progress_label, enabled=progress)

    for candidate in candidates:
        stats["candidate_count"] += 1
        stats["rows_scanned"] = max(stats["rows_scanned"], candidate.source_row_index + 1)
        if candidate.text in seen_texts:
            stats["duplicate_text_count"] += 1
            continue
        seen_texts.add(candidate.text)
        unique_index = len(seen_texts)
        stats["unique_candidate_count"] = unique_index

        if len(selected) < count:
            selected.append(candidate)
        else:
            replace_at = rng.randrange(unique_index)
            if replace_at < count:
                selected[replace_at] = candidate

        if stats["candidate_count"] % 1024 == 0:
            bar.update(
                stats["rows_scanned"],
                stats["candidate_count"],
                stats["unique_candidate_count"],
                len(selected),
            )
        if early_stop_unique_count and unique_index >= early_stop_unique_count:
            stats["early_stopped"] = True
            break

    rng.shuffle(selected)
    bar.close(
        stats["rows_scanned"],
        stats["candidate_count"],
        stats["unique_candidate_count"],
        len(selected),
    )
    return selected, stats


def to_probe(candidate: Candidate) -> dict[str, Any]:
    return {
        "id": stable_id(candidate.category, candidate.text),
        "text": candidate.text,
        "category": candidate.category,
        "truncation_ratio": candidate.truncation_ratio,
        "source_lang": candidate.source_lang,
        "metadata": {
            "probe_type": "wikipedia_multilingual_internal_truncation",
            "source_dataset": candidate.source_dataset,
            "dataset_config": candidate.dataset_config,
            "split": candidate.split,
            "source_row_index": candidate.source_row_index,
            "article_id": candidate.article_id,
            "title": candidate.title,
            "url": candidate.url,
            "source_span": candidate.source_span,
            "cut_length": candidate.cut_length,
            "source_span_length": len(candidate.source_span),
            "span_start_in_fragment": candidate.span_start_in_fragment,
        },
    }


def output_path(output_dir: Path, dump: str, lang: str, count: int, suffix: str) -> Path:
    return output_dir / f"wikipedia_{dump}_{lang}_truncated_{count}_{suffix}"


def generate_language(
    *,
    spec: LanguageSpec,
    dataset: str,
    dump: str,
    split: str,
    cache_dir: Path,
    output_dir: Path,
    count: int,
    max_rows: int,
    max_candidates_per_row: int,
    seed: int,
    early_stop_unique_factor: float,
    progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_config = f"{dump}.{spec.lang}"
    early_stop_unique_count = (
        int(count * early_stop_unique_factor)
        if early_stop_unique_factor > 0
        else None
    )
    selected, stats = reservoir_sample(
        iter_language_candidates(
            spec=spec,
            dataset=dataset,
            dataset_config=dataset_config,
            split=split,
            cache_dir=cache_dir,
            max_rows=max_rows,
            max_candidates_per_row=max_candidates_per_row,
        ),
        count=count,
        seed=seed,
        progress_label=f"Wikipedia {spec.lang}",
        progress=progress,
        early_stop_unique_count=early_stop_unique_count,
    )
    if len(selected) < count:
        raise SystemExit(
            f"{spec.lang}: generated only {len(selected)} unique probes; "
            f"increase --max-rows-per-language or relax extraction thresholds."
        )

    probes = [to_probe(candidate) for candidate in selected]
    probe_output = output_path(output_dir, dump, spec.lang, count, "probes.json")
    manifest_output = output_path(output_dir, dump, spec.lang, count, "manifest.json")
    write_json(probe_output, probes)

    manifest = {
        "output": str(probe_output),
        "source_dataset": dataset,
        "dataset_config": dataset_config,
        "split": split,
        "cache_dir": str(cache_dir),
        "language": spec.lang,
        "source_lang": spec.source_lang,
        "category": spec.category,
        "requested_count": count,
        "generated_count": len(probes),
        "seed": seed,
        "ratios": list(spec.ratios),
        "min_span_len": spec.min_span_len,
        "max_span_len": spec.max_span_len,
        "min_prefix_len": spec.min_prefix_len,
        "context_words": spec.context_words,
        "context_chars": spec.context_chars,
        "min_chars": spec.min_chars,
        "max_chars": spec.max_chars,
        "max_rows": max_rows,
        "max_candidates_per_row": max_candidates_per_row,
        "early_stop_unique_factor": early_stop_unique_factor,
        "stats": stats,
        "examples": probes[:10],
    }
    write_json(manifest_output, manifest)
    print(f"Wrote {len(probes)} {spec.source_lang} probes to {probe_output}")
    print(f"Wrote manifest to {manifest_output}")
    return probes, manifest


def ensure_unique_ids(probes: list[dict[str, Any]], output_name: str) -> None:
    ids = [probe["id"] for probe in probes]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{output_name}: duplicate probe ids detected")


def write_combined_outputs(
    *,
    output: Path,
    manifest_output: Path,
    probes: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    label: str,
) -> None:
    ensure_unique_ids(probes, str(output))
    write_json(output, probes)
    manifest = {
        "output": str(output),
        "label": label,
        "generated_count": len(probes),
        "inputs": [
            {
                "output": item["output"],
                "language": item.get("language", "english"),
                "source_lang": item.get("source_lang", "english"),
                "generated_count": item["generated_count"],
            }
            for item in manifests
        ],
    }
    write_json(manifest_output, manifest)
    print(f"Wrote {len(probes)} combined probes to {output}")
    print(f"Wrote combined manifest to {manifest_output}")


def load_existing_probes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing existing English probe file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected a list of probes in {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate de/fr/ar/zh Wikipedia truncation probes."
    )
    parser.add_argument("--dataset", default="wikimedia/wikipedia")
    parser.add_argument("--dump", default="20231101")
    parser.add_argument("--split", default="train")
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES))
    parser.add_argument("--count-per-language", type=int, default=100000)
    parser.add_argument("--max-rows-per-language", type=int, default=200000)
    parser.add_argument(
        "--max-candidates-per-row",
        type=int,
        default=50,
        help="Limit candidates contributed by each Wikipedia article; set <=0 to disable.",
    )
    parser.add_argument(
        "--early-stop-unique-factor",
        type=float,
        default=3.0,
        help="Stop each language after count*factor unique candidates; set <=0 to scan max rows.",
    )
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--cache-dir", default="raw_probe/.cache/huggingface")
    parser.add_argument("--output-dir", default="probes")
    parser.add_argument("--combined-output", default=None)
    parser.add_argument("--combined-manifest", default=None)
    parser.add_argument("--english-probes", default="probes/wikitext_truncated_100000_probes.json")
    parser.add_argument("--english-combined-output", default=None)
    parser.add_argument("--english-combined-manifest", default=None)
    parser.add_argument("--skip-english-combined", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    unknown = sorted(set(args.languages) - set(LANGUAGE_SPECS))
    if unknown:
        raise SystemExit(f"Unsupported languages: {', '.join(unknown)}")
    if args.count_per_language <= 0:
        raise SystemExit("--count-per-language must be positive")

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    all_multilingual: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []

    for offset, lang in enumerate(args.languages):
        spec = LANGUAGE_SPECS[lang]
        probes, manifest = generate_language(
            spec=spec,
            dataset=args.dataset,
            dump=args.dump,
            split=args.split,
            cache_dir=cache_dir,
            output_dir=output_dir,
            count=args.count_per_language,
            max_rows=args.max_rows_per_language,
            max_candidates_per_row=args.max_candidates_per_row,
            seed=args.seed + offset,
            early_stop_unique_factor=args.early_stop_unique_factor,
            progress=not args.no_progress,
        )
        all_multilingual.extend(probes)
        manifests.append(manifest)

    total = len(all_multilingual)
    combined_output = (
        Path(args.combined_output)
        if args.combined_output
        else output_dir / f"wikipedia_{args.dump}_multilingual_{total}_probes.json"
    )
    combined_manifest = (
        Path(args.combined_manifest)
        if args.combined_manifest
        else combined_output.with_name(combined_output.stem.replace("_probes", "_manifest") + ".json")
    )
    write_combined_outputs(
        output=combined_output,
        manifest_output=combined_manifest,
        probes=all_multilingual,
        manifests=manifests,
        label="multilingual_wikipedia",
    )

    if not args.skip_english_combined:
        english_path = Path(args.english_probes)
        english_probes = load_existing_probes(english_path)
        english_manifest = {
            "output": str(english_path),
            "language": "en",
            "source_lang": "english",
            "generated_count": len(english_probes),
        }
        combined_with_english = english_probes + all_multilingual
        english_combined_output = (
            Path(args.english_combined_output)
            if args.english_combined_output
            else output_dir / f"wikitext_plus_wikipedia_multilingual_{len(combined_with_english)}_probes.json"
        )
        english_combined_manifest = (
            Path(args.english_combined_manifest)
            if args.english_combined_manifest
            else english_combined_output.with_name(
                english_combined_output.stem.replace("_probes", "_manifest") + ".json"
            )
        )
        write_combined_outputs(
            output=english_combined_output,
            manifest_output=english_combined_manifest,
            probes=combined_with_english,
            manifests=[english_manifest, *manifests],
            label="english_wikitext_plus_multilingual_wikipedia",
        )

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
