"""Tests for synthetic data generation: cluster.py"""

import json
import pytest
import numpy as np
from src.codeflow.synthetic.cluster import (
    kcenter_greedy,
    embed_examples,
    select_coreset,
    select_by_feature_diversity,
    _cosine_distance,
    _euclidean_distance,
)


class TestDistance:
    def test_cosine_identical(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert abs(_cosine_distance(a, b)) < 1e-6

    def test_cosine_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(_cosine_distance(a, b) - 2.0) < 1e-6

    def test_euclidean_same(self):
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 0.0])
        assert _euclidean_distance(a, b) < 1e-6

    def test_euclidean_different(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert abs(_euclidean_distance(a, b) - 5.0) < 1e-6


class TestKCenterGreedy:
    def test_small_dataset(self):
        emb = np.random.randn(10, 64).astype(np.float32)
        # Normalize to spread points on unit sphere for better cosine diversity
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        indices = kcenter_greedy(emb, target_size=5, seed=42)
        assert len(indices) == 5
        assert len(set(indices)) == 5  # All unique

    def test_target_larger_than_dataset(self):
        emb = np.random.randn(3, 64).astype(np.float32)
        indices = kcenter_greedy(emb, target_size=10, seed=42)
        assert len(indices) == 3

    def test_selection_is_diverse(self):
        # Use euclidean metric with well-separated clusters
        rng = np.random.RandomState(42)
        c1 = rng.randn(10, 64) + 0.0
        c2 = rng.randn(10, 64) + 50.0
        c3 = rng.randn(10, 64) + 100.0
        emb = np.vstack([c1, c2, c3]).astype(np.float32)

        indices = kcenter_greedy(emb, target_size=3, metric="euclidean", seed=42)
        # Should pick one from each cluster (idx ranges: 0-9, 10-19, 20-29)
        clusters = [idx // 10 for idx in indices]
        assert len(set(clusters)) == 3

    def test_deterministic_with_seed(self):
        emb = np.random.randn(20, 64).astype(np.float32)
        idx1 = kcenter_greedy(emb, target_size=5, seed=42)
        idx2 = kcenter_greedy(emb, target_size=5, seed=42)
        assert np.array_equal(idx1, idx2)

    def test_euclidean_metric(self):
        emb = np.random.randn(10, 32).astype(np.float32)
        indices = kcenter_greedy(emb, target_size=3, metric="euclidean", seed=42)
        assert len(indices) == 3


def _bag_of_chars(text):
    v = np.zeros(64)
    for c in text:
        v[ord(c) % 64] += 1
    return v / (np.linalg.norm(v) or 1)


class TestEmbedExamples:
    def test_embeds_each_example(self):
        examples = [
            {"instruction": "task 1", "input": "code 1"},
            {"instruction": "task 2", "input": "code 2"},
        ]
        assert embed_examples(examples, _bag_of_chars, embedding_dim=64).shape == (2, 64)

    def test_empty_handled(self):
        assert embed_examples([], _bag_of_chars, embedding_dim=64).shape == (0, 64)

    def test_requires_an_embedding_model(self):
        with pytest.raises(ValueError):
            embed_examples([{"instruction": "x"}], None)


class TestSelectCoreset:
    def test_small_dataset_unchanged(self):
        examples = [{"instruction": f"task {i}", "input": "code"} for i in range(5)]
        selected = select_coreset(examples, _bag_of_chars, target_size=10)
        assert len(selected) == 5

    def test_large_dataset_reduced(self):
        examples = [{"instruction": f"task {i:04d}", "input": "code"} for i in range(100)]
        selected = select_coreset(examples, _bag_of_chars, target_size=20)
        assert len(selected) == 20


class TestSelectByFeatureDiversity:
    def test_basic_selection(self):
        examples = [
            {"instruction": "t1", "features": ["macros", "atoms"]},
            {"instruction": "t2", "features": ["macros"]},
            {"instruction": "t3", "features": ["atoms"]},
            {"instruction": "t4", "features": ["protocols"]},
            {"instruction": "t5", "features": ["transducers"]},
        ]
        selected = select_by_feature_diversity(examples, target_size=3, seed=42)
        assert len(selected) == 3

    def test_target_larger_than_available(self):
        examples = [{"instruction": "t1", "features": ["macros"]}]
        selected = select_by_feature_diversity(examples, target_size=10)
        assert len(selected) <= 1

    def test_string_features_accepted(self):
        examples = [
            {"instruction": "t1", "feature": "macros"},
            {"instruction": "t2", "feature": "atoms"},
        ]
        selected = select_by_feature_diversity(examples, target_size=2, seed=42)
        assert 1 <= len(selected) <= 2


class TestFeatureDiversityNeverShrinks:
    """Training-example dicts carry no 'features' key; selection used to
    return [] and wipe the output whenever there were more than target."""

    def test_examples_without_features_fill_to_target(self):
        examples = [{"instruction": f"t{i}", "output": "o"} for i in range(10)]
        selected = select_by_feature_diversity(examples, target_size=4, seed=1)
        assert len(selected) == 4

    def test_fewer_features_than_target_fills_to_target(self):
        examples = [{"instruction": f"t{i}", "features": ["macros" if i % 2 else "atoms"]}
                    for i in range(10)]
        selected = select_by_feature_diversity(examples, target_size=6, seed=1)
        assert len(selected) == 6
        assert len({e["instruction"] for e in selected}) == 6
        # Both features still covered.
        assert {f for e in selected for f in e["features"]} == {"macros", "atoms"}

    def test_custom_extractor(self):
        feats = {f"t{i}": ["rare"] if i == 7 else [] for i in range(10)}
        examples = [{"instruction": f"t{i}"} for i in range(10)]
        selected = select_by_feature_diversity(
            examples, extract_fn=lambda ex: feats[ex["instruction"]],
            target_size=2, seed=3)
        assert len(selected) == 2
        assert any(e["instruction"] == "t7" for e in selected)

    def test_dict_features_do_not_crash(self):
        examples = [{"instruction": f"t{i}", "features": [{"name": f"f{i % 3}"}]}
                    for i in range(6)]
        selected = select_by_feature_diversity(examples, target_size=3, seed=1)
        assert len(selected) == 3

    def test_deterministic(self):
        examples = [{"instruction": f"t{i}"} for i in range(20)]
        a = select_by_feature_diversity(examples, target_size=5, seed=9)
        b = select_by_feature_diversity(examples, target_size=5, seed=9)
        assert a == b
