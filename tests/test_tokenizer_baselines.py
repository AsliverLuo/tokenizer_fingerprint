import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_tokenizer_baselines import (
    Thresholds,
    build_baseline_registry,
    classify_equivalence,
    compare_pairs,
)


def test_classify_equivalence_hard_and_near():
    thresholds = Thresholds(
        near_token_jaccard=0.60,
        near_id_consistency=0.60,
        near_encoding_rate=0.50,
    )

    hard = classify_equivalence(
        exact_token_to_id_same=True,
        encoding_same_rate=1.0,
        token_jaccard=1.0,
        id_consistency=1.0,
        thresholds=thresholds,
    )
    assert hard == "hard_equivalent"

    near = classify_equivalence(
        exact_token_to_id_same=False,
        encoding_same_rate=1.0,
        token_jaccard=1.0,
        id_consistency=2 / 3,
        thresholds=thresholds,
    )
    assert near == "near_equivalent"

    distinct = classify_equivalence(
        exact_token_to_id_same=False,
        encoding_same_rate=0.0,
        token_jaccard=0.20,
        id_consistency=0.20,
        thresholds=thresholds,
    )
    assert distinct == "distinct"


def test_baseline_registry_selects_primary_and_variant():
    samples = ["sample_a", "sample_b"]
    thresholds = Thresholds(
        near_token_jaccard=0.95,
        near_id_consistency=0.95,
        near_encoding_rate=0.95,
    )
    models = {
        "Fam-Base": _model(
            "Fam-Base",
            family="fam",
            role="base",
            preferred_rank=10,
            vocab={"a": 1, "b": 2, "c": 3},
            encodings={"sample_a": [1], "sample_b": [2]},
        ),
        "Fam-Instruct": _model(
            "Fam-Instruct",
            family="fam",
            role="instruct",
            preferred_rank=1,
            vocab={"a": 1, "b": 2, "c": 3},
            encodings={"sample_a": [1], "sample_b": [2]},
        ),
        "Fam-NewBase": _model(
            "Fam-NewBase",
            family="fam",
            role="base",
            preferred_rank=5,
            vocab={"x": 10, "y": 11, "z": 12},
            encodings={"sample_a": [10], "sample_b": [11]},
        ),
    }
    model_order = ["Fam-Base", "Fam-Instruct", "Fam-NewBase"]
    pairs = compare_pairs(models, model_order, samples, thresholds)
    registry = build_baseline_registry(
        models=models,
        model_order=model_order,
        pair_rows=pairs,
        thresholds=thresholds,
        manifest_path=Path("manifest.yaml"),
        encoding_sample_count=len(samples),
    )

    family = registry["families"]["fam"]
    assert family["lineage_count"] == 2
    assert family["primary_baseline_id"] == "fam__primary"

    baselines = {item["baseline_id"]: item for item in registry["baselines"]}
    primary = baselines["fam__primary"]
    variant = baselines["fam__variant_01"]

    assert primary["representative_model"] == "Fam-Base"
    assert primary["member_models"] == ["Fam-Base", "Fam-Instruct"]
    assert primary["kind"] == "primary"
    assert variant["representative_model"] == "Fam-NewBase"
    assert variant["kind"] == "variant"


def test_family_primary_uses_preferred_rank_on_equal_lineage_size():
    samples = ["sample"]
    thresholds = Thresholds(
        near_token_jaccard=0.95,
        near_id_consistency=0.95,
        near_encoding_rate=0.95,
    )
    models = {
        "Old-Base": _model(
            "Old-Base",
            family="fam",
            role="base",
            preferred_rank=20,
            vocab={"old": 1},
            encodings={"sample": [1]},
        ),
        "New-Base": _model(
            "New-Base",
            family="fam",
            role="base",
            preferred_rank=10,
            vocab={"new": 2},
            encodings={"sample": [2]},
        ),
    }
    model_order = ["Old-Base", "New-Base"]
    pairs = compare_pairs(models, model_order, samples, thresholds)
    registry = build_baseline_registry(
        models=models,
        model_order=model_order,
        pair_rows=pairs,
        thresholds=thresholds,
        manifest_path=Path("manifest.yaml"),
        encoding_sample_count=len(samples),
    )

    baselines = {item["baseline_id"]: item for item in registry["baselines"]}
    assert baselines["fam__primary"]["representative_model"] == "New-Base"
    assert baselines["fam__variant_01"]["representative_model"] == "Old-Base"


def _model(name, family, role, preferred_rank, vocab, encodings):
    return {
        "name": name,
        "family": family,
        "model_id": name,
        "source": "hf",
        "local_dir": f"hub/tokenizer_only/{name}",
        "encoding": "",
        "version": "",
        "role": role,
        "preferred_rank": preferred_rank,
        "notes": "",
        "vocab": vocab,
        "vocab_size": len(vocab),
        "vocab_sha256": "vocab",
        "special_tokens_map": {"eos_token": "</s>"},
        "special_tokens_map_sha256": "special",
        "file_hashes": {"tokenizer.json": name},
        "comparable_file_hashes": {"tokenizer.json": name},
        "encoding_samples": encodings,
    }
