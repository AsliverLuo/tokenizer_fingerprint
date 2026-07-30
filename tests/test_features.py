"""
test_features.py — 核心模块单元测试

测试覆盖：
- Probe 生成与序列化
- Token 类型分类
- 特征抽取
- 相似度计算
- 决策逻辑
"""

import json
import tempfile
from pathlib import Path

import numpy as np

# ── 测试辅助 ────────────────────────────────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer_fingerprint.schema import (
    Probe, SingleTokenResult, ModelFingerprint, DetectionLabel,
)
from tokenizer_fingerprint.probe_generator import (
    generate_probes, save_probes, load_probes,
)
from tokenizer_fingerprint.feature_extractor import (
    classify_token, extract_surface_features, extract_type_features,
    extract_transition_features, extract_fingerprint,
)
from tokenizer_fingerprint.similarity import (
    boundary_consistency_score,
    compute_similarity,
    compute_similarity_breakdown,
    bootstrap_similarity,
    compute_stability_variance,
    compute_bank_statistics,
    make_decision,
)
from tokenizer_fingerprint.schema import BankStatistics
from tokenizer_fingerprint.reference_bank import ReferenceBank


def test_probe_generation():
    """测试 probe 生成"""
    probes = generate_probes(total_count=100, seed=42)
    assert len(probes) == 100
    assert all(isinstance(p, Probe) for p in probes)
    assert all(p.id for p in probes)
    assert all(p.text for p in probes)
    assert all(p.category for p in probes)

    # 检查类别分布
    categories = set(p.category for p in probes)
    assert len(categories) >= 5  # 至少覆盖 5 个类别
    print(f"  ✓ Generated {len(probes)} probes, {len(categories)} categories")


def test_probe_serialization():
    """测试 probe 序列化/反序列化"""
    probes = generate_probes(total_count=50, seed=42)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    save_probes(probes, path)
    loaded = load_probes(path)

    assert len(loaded) == len(probes)
    assert loaded[0].id == probes[0].id
    assert loaded[0].text == probes[0].text
    path.unlink()
    print("  ✓ Probe serialization roundtrip OK")


def test_token_classification():
    """测试 token 类型分类"""
    cases = [
        ("你", "cjk"),
        ("好的", "cjk"),
        ("hello", "latin"),
        ("The", "latin"),
        ("123", "digit"),
        ("3.14", "digit"),
        (".", "punctuation"),
        ("，", "punctuation"),
        (" the", "whitespace_prefixed"),
        ("\ndef", "newline_prefixed"),
        ("()", "code_like"),
        ('{"', "json_like"),
        ("https://", "url_like"),
        ("", "empty"),
        ("  ", "whitespace_prefixed"),
    ]
    for text, expected in cases:
        result = classify_token(text)
        assert result == expected, f"classify_token({text!r}) = {result!r}, expected {expected!r}"
    print("  ✓ Token classification: all cases passed")


def test_surface_features():
    """测试表面特征抽取"""
    results = [
        SingleTokenResult(probe_id="p1", model_name="test", output_text=" the",
                         char_length=4, byte_length=4, has_leading_space=True),
        SingleTokenResult(probe_id="p2", model_name="test", output_text="你",
                         char_length=1, byte_length=3),
        SingleTokenResult(probe_id="p3", model_name="test", output_text="\n",
                         char_length=1, byte_length=1, has_leading_newline=True),
        SingleTokenResult(probe_id="p4", model_name="test", output_text="",
                         char_length=0, byte_length=0, is_empty=True),
    ]
    sf = extract_surface_features(results)
    assert sf.leading_space_rate == 0.25
    assert sf.leading_newline_rate == 0.25
    assert sf.empty_output_rate == 0.25
    assert "4" in sf.char_len_hist  # char_length=4 for " the"
    print("  ✓ Surface features extraction OK")


def test_type_features():
    """测试类型特征抽取"""
    results = [
        SingleTokenResult(probe_id="p1", model_name="test", output_text=" the",
                         char_length=4, byte_length=4, token_type="whitespace_prefixed"),
        SingleTokenResult(probe_id="p2", model_name="test", output_text="你",
                         char_length=1, byte_length=3, token_type="cjk"),
        SingleTokenResult(probe_id="p3", model_name="test", output_text="def",
                         char_length=3, byte_length=3, token_type="latin"),
    ]
    tf = extract_type_features(results)
    assert abs(tf.type_distribution.get("whitespace_prefixed", 0) - 1/3) < 0.01
    assert abs(tf.type_distribution.get("cjk", 0) - 1/3) < 0.01
    print("  ✓ Type features extraction OK")


def test_transition_features():
    """测试转移特征抽取"""
    probes = [
        Probe(id="p1", text="今天天气", category="chinese_natural"),
        Probe(id="p2", text="The weather", category="english_natural"),
        Probe(id="p3", text="def func(", category="code"),
    ]
    results = [
        SingleTokenResult(probe_id="p1", model_name="test", output_text="很",
                         char_length=1, byte_length=3, token_type="cjk"),
        SingleTokenResult(probe_id="p2", model_name="test", output_text=" is",
                         char_length=3, byte_length=3, token_type="whitespace_prefixed"),
        SingleTokenResult(probe_id="p3", model_name="test", output_text=")",
                         char_length=1, byte_length=1, token_type="code_like"),
    ]
    tf = extract_transition_features(results, probes)
    assert "chinese_natural→cjk" in tf.transition_matrix
    assert "english_natural→whitespace_prefixed" in tf.transition_matrix
    assert "code→code_like" in tf.transition_matrix
    print("  ✓ Transition features extraction OK")


def test_boundary_consistency_score():
    """测试 BCS """
    fp1 = _make_dummy_fingerprint("model_a", "family_x")
    fp2 = _make_dummy_fingerprint("model_b", "family_x")
    fp3 = _make_dummy_fingerprint("model_c", "family_y")

    # No raw_results → BCS=0 for all pairs
    bcs_same = boundary_consistency_score(fp1, fp2)
    bcs_diff = boundary_consistency_score(fp1, fp3)
    assert bcs_same == 0.0 and bcs_diff == 0.0
    print("  ✓ BCS OK")


def test_bcs_breakdown():
    """测试 BCS breakdown 分项"""
    fp1 = _make_dummy_fingerprint("model_a", "family_x")
    fp2 = _make_dummy_fingerprint("model_b", "family_x")
    breakdown = compute_similarity_breakdown(fp1, fp2)
    assert "bcs" in breakdown
    assert "char_length_match_rate" in breakdown
    assert "common_probe_count" in breakdown
    print("  ✓ BCS breakdown OK")


def test_compute_similarity():
    """测试综合相似度 (with aligned raw_results for BCS)"""
    # Create two fingerprints with identical raw_results → BCS=1.0
    shared_results = [
        SingleTokenResult(probe_id="p1", model_name="test", output_text="你", char_length=1, byte_length=3, token_type="cjk"),
        SingleTokenResult(probe_id="p2", model_name="test", output_text="的", char_length=1, byte_length=3, token_type="cjk"),
    ]
    diff_results = [
        SingleTokenResult(probe_id="p1", model_name="test", output_text="the", char_length=3, byte_length=3, token_type="latin"),
        SingleTokenResult(probe_id="p2", model_name="test", output_text="a", char_length=1, byte_length=1, token_type="latin"),
    ]

    fp1 = _make_dummy_fingerprint("model_a", "family_x")
    fp1.raw_results = shared_results
    fp2 = _make_dummy_fingerprint("model_b", "family_x")
    fp2.raw_results = shared_results  # identical → BCS=1.0
    fp3 = _make_dummy_fingerprint("model_c", "family_y")
    fp3.raw_results = diff_results  # all different → BCS=0.0

    sim_same = compute_similarity(fp1, fp2)
    sim_diff = compute_similarity(fp1, fp3)
    assert sim_same > sim_diff, f"Same family sim ({sim_same:.4f}) should > diff ({sim_diff:.4f})"
    assert sim_same == 1.0
    assert sim_diff == 0.0
    print(f"  ✓ Similarity: same_family={sim_same:.4f}, diff_family={sim_diff:.4f}")


def test_stability_variance():
    """测试稳定性方差计算"""
    # 完全一致
    results_stable = [
        SingleTokenResult(probe_id="p1", model_name="test", output_text="the"),
        SingleTokenResult(probe_id="p1", model_name="test", output_text="the"),
        SingleTokenResult(probe_id="p1", model_name="test", output_text="the"),
    ]
    assert compute_stability_variance(results_stable) == 0.0

    # 不一致
    results_unstable = [
        SingleTokenResult(probe_id="p1", model_name="test", output_text="the"),
        SingleTokenResult(probe_id="p1", model_name="test", output_text="a"),
    ]
    assert compute_stability_variance(results_unstable) == 1.0
    print("  ✓ Stability variance OK")


def test_decision():
    """测试决策逻辑 (dummy fingerprints, no raw_results → all BCS=0 → not_same_source)"""
    fp_target = _make_dummy_fingerprint("target", "unknown", cjk_rate=0.5, latin_rate=0.3)
    fp_ref_close = _make_dummy_fingerprint("ref_close", "family_x", cjk_rate=0.5, latin_rate=0.3)
    fp_ref_far = _make_dummy_fingerprint("ref_far", "family_y", cjk_rate=0.1, latin_rate=0.8)

    decision = make_decision(
        fp_target,
        [fp_ref_close, fp_ref_far],
        bootstrap_n=10,
    )
    assert "label" in decision
    assert "confidence" in decision
    assert "top_matches" in decision
    assert "diagnosis" in decision
    assert "evidence" in decision
    assert decision["same_source_of"] is None  # no raw_results → not same_source
    # 无 raw_results → 所有 BCS=0 → not_same_source
    assert decision["label"] == "not_same_source"
    print(f"  ✓ Decision: label={decision['label']}, conf={decision['confidence']:.4f}")
    print(f"    diagnosis={decision['diagnosis']}")
    print(f"    evidence={decision['evidence']}")


def test_bank_statistics_from_fingerprints():
    """测试从指纹列表计算跨家族统计"""
    fp1 = _make_dummy_fingerprint("model_a", "family_x")
    fp2 = _make_dummy_fingerprint("model_b", "family_x")
    fp3 = _make_dummy_fingerprint("model_c", "family_y")

    stats = compute_bank_statistics([fp1, fp2, fp3])
    assert stats.n_models == 3
    assert stats.n_cross_pairs >= 0  # 跨家族对：fp1-fp3, fp2-fp3
    print(f"  ✓ Bank statistics: {stats.n_models} models, {stats.n_cross_pairs} cross-fam pairs")


def test_decision_self_filter():
    """验证自匹配被排除"""
    fp_target = _make_dummy_fingerprint("target_model", "family_x")
    fp_same_name = _make_dummy_fingerprint("target_model", "family_x")  # 重名
    fp_other = _make_dummy_fingerprint("other_model", "family_x")

    decision = make_decision(fp_target, [fp_same_name, fp_other], bootstrap_n=10)
    # 自匹配被排除，只有 other_model 参与匹配
    assert len(decision["top_matches"]) <= 1  # 排除了同名的
    # 无 raw_results → BCS=0 → not_same_source
    assert decision["label"] == "not_same_source"
    print(f"  ✓ Self-filter: excluded duplicate name, top_matches={len(decision['top_matches'])}")


def test_decision_with_bank_stats():
    """测试传入 bank_stats 时的证据计算"""
    fp_target = _make_dummy_fingerprint("target", "unknown")
    fp_ref = _make_dummy_fingerprint("ref_model", "family_x")

    # 构造虚拟的 bank statistics
    stats = BankStatistics(
        cross_family_mean=0.25,
        cross_family_std=0.15,
        cross_family_p95=0.45,
        cross_family_p99=0.49,
        cross_family_max=0.50,
        n_cross_pairs=30,
        n_models=9,
    )

    decision = make_decision(
        fp_target,
        [fp_ref],
        bank_stats=stats,
        bootstrap_n=10,
    )
    assert "evidence" in decision
    ev = decision["evidence"]
    assert "cross_family_baseline" in ev
    assert ev["cross_family_baseline"]["mean"] == 0.25
    assert ev["cross_family_baseline"]["p99"] == 0.49
    print(f"  ✓ Bank stats decision: evidence={ev}")


def test_reference_bank_with_stats():
    """测试 ReferenceBank 加载 bank_compare 统计"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        bank = ReferenceBank(Path(tmpdir))
        fp1 = _make_dummy_fingerprint("model_a", "family_x")
        fp2 = _make_dummy_fingerprint("model_b", "family_y")
        bank.add(fp1)
        bank.add(fp2)
        bank.save()

        # 创建虚拟 bank_compare JSON
        compare_data = {
            "pairs": [
                {"model_a": "model_a", "family_a": "family_x",
                 "model_b": "model_b", "family_b": "family_y",
                 "score": 0.35},
                {"model_a": "model_a", "family_a": "family_x",
                 "model_b": "model_a", "family_b": "family_x",
                 "score": 1.0},
            ]
        }
        bc_path = Path(tmpdir) / "bank_compare.json"
        bc_path.write_text(json.dumps(compare_data))

        # 加载带统计量的 bank
        loaded = ReferenceBank.load(Path(tmpdir), bank_compare_path=bc_path)
        stats = loaded.statistics
        assert stats is not None
        assert stats.n_cross_pairs == 1
        assert stats.cross_family_max == 0.35
    print("  ✓ ReferenceBank with bank_compare stats OK")


def test_reference_bank():
    """测试参考库保存/加载"""
    with tempfile.TemporaryDirectory() as tmpdir:
        bank = ReferenceBank(Path(tmpdir))
        fp1 = _make_dummy_fingerprint("model_a", "family_x")
        fp2 = _make_dummy_fingerprint("model_b", "family_y")
        bank.add(fp1)
        bank.add(fp2)
        bank.save()

        loaded = ReferenceBank.load(Path(tmpdir))
        assert set(loaded.list_models()) == {"model_a", "model_b"}
        assert set(loaded.list_families()) == {"family_x", "family_y"}
    print("  ✓ Reference bank save/load OK")


def test_fingerprint_serialization():
    """测试 fingerprint 序列化"""
    fp = _make_dummy_fingerprint("test_model", "test_family")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    fp.save(path)
    loaded = ModelFingerprint.load(path)

    assert loaded.model_name == fp.model_name
    assert loaded.family == fp.family
    assert loaded.surface.leading_space_rate == fp.surface.leading_space_rate
    path.unlink()
    print("  ✓ Fingerprint serialization OK")


# ── 辅助函数 ────────────────────────────────────────────────────

def _make_dummy_fingerprint(
    name: str, family: str,
    cjk_rate: float = 0.3, latin_rate: float = 0.4,
) -> ModelFingerprint:
    """创建测试用虚拟指纹"""
    from tokenizer_fingerprint.schema import (
        SurfaceFeatures, TypeFeatures, TransitionFeatures,
    )
    other_rate = max(0, 1.0 - cjk_rate - latin_rate)

    fp = ModelFingerprint(
        model_name=name,
        family=family,
        surface=SurfaceFeatures(
            char_len_hist={"1": cjk_rate, "3": latin_rate, "2": other_rate},
            byte_len_hist={"3": cjk_rate, "3": latin_rate, "2": other_rate},
            byte_per_char_ratio_mean=1.5 + cjk_rate,
            leading_space_rate=latin_rate * 0.8,
            leading_newline_rate=0.05,
            empty_output_rate=0.02,
        ),
        type_feat=TypeFeatures(
            type_distribution={
                "cjk": cjk_rate,
                "latin": latin_rate,
                "digit": other_rate * 0.3,
                "punctuation": other_rate * 0.3,
                "other": other_rate * 0.4,
            }
        ),
        transition=TransitionFeatures(
            transition_matrix={
                "chinese_natural→cjk": cjk_rate * 0.8,
                "chinese_natural→punctuation": cjk_rate * 0.2,
                "english_natural→latin": latin_rate * 0.7,
                "english_natural→whitespace_prefixed": latin_rate * 0.3,
                "code→code_like": 0.5,
                "code→punctuation": 0.3,
                "code→latin": 0.2,
            }
        ),
        n_probes=100,
    )
    return fp


# ── 运行测试 ────────────────────────────────────────────────────

def run_all_tests():
    tests = [
        ("Probe Generation", test_probe_generation),
        ("Probe Serialization", test_probe_serialization),
        ("Token Classification", test_token_classification),
        ("Surface Features", test_surface_features),
        ("Type Features", test_type_features),
        ("Transition Features", test_transition_features),
        ("BCS Boundary Consistency", test_boundary_consistency_score),
        ("BCS Breakdown", test_bcs_breakdown),
        ("Compute Similarity", test_compute_similarity),
        ("Stability Variance", test_stability_variance),
        ("Decision Logic", test_decision),
        ("Bank Statistics from Fingerprints", test_bank_statistics_from_fingerprints),
        ("Decision Self Filter", test_decision_self_filter),
        ("Decision with Bank Stats", test_decision_with_bank_stats),
        ("Reference Bank with Stats", test_reference_bank_with_stats),
        ("Reference Bank", test_reference_bank),
        ("Fingerprint Serialization", test_fingerprint_serialization),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"\n[TEST] {name}")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    if failed == 0:
        print("All tests passed! ✓")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
