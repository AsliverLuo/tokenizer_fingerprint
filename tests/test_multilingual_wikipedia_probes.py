import re

from scripts.generate_multilingual_wikipedia_probes import (
    LANGUAGE_SPECS,
    iter_candidates_from_row,
    stable_id,
    to_probe,
)


def _first_probe(lang, text):
    spec = LANGUAGE_SPECS[lang]
    row = {
        "id": "article-1",
        "title": "Example",
        "url": "https://example.invalid/wiki/Example",
        "text": text,
    }
    candidate = next(
        iter_candidates_from_row(
            row,
            row_index=7,
            spec=spec,
            source_dataset="wikimedia/wikipedia",
            dataset_config=f"20231101.{lang}",
            split="train",
            max_candidates_per_row=50,
        )
    )
    return to_probe(candidate)


def test_latin_language_probe_generation():
    german = _first_probe(
        "de",
        "Die industrielle Revolution beeinflusste zahlreiche Gemeinschaften in Europa.",
    )
    french = _first_probe(
        "fr",
        "La responsabilite internationale influence plusieurs administrations europeennes.",
    )

    assert german["category"] == "german_partial"
    assert german["source_lang"] == "german"
    assert re.search(r"[A-Za-z]$", german["text"])
    assert german["metadata"]["dataset_config"] == "20231101.de"

    assert french["category"] == "french_partial"
    assert french["source_lang"] == "french"
    assert re.search(r"[A-Za-z]$", french["text"])
    assert french["metadata"]["dataset_config"] == "20231101.fr"


def test_arabic_probe_generation():
    probe = _first_probe(
        "ar",
        "تطورت الحضارة العربية في العديد من المناطق التاريخية والثقافية.",
    )

    assert probe["category"] == "arabic_partial"
    assert probe["source_lang"] == "arabic"
    assert probe["metadata"]["dataset_config"] == "20231101.ar"
    assert "\u0600" <= probe["text"][-1] <= "\u06ff"


def test_chinese_probe_generation():
    probe = _first_probe(
        "zh",
        "中华人民共和国的历史文化影响了许多地区的发展进程。",
    )

    assert probe["category"] == "chinese_partial"
    assert probe["source_lang"] == "chinese"
    assert probe["metadata"]["dataset_config"] == "20231101.zh"
    assert "\u4e00" <= probe["text"][-1] <= "\u9fff"


def test_stable_id_is_deterministic():
    assert stable_id("german_partial", "Die industri") == stable_id(
        "german_partial",
        "Die industri",
    )
