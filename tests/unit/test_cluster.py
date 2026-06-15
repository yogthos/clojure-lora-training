"""Tests for synthetic data generation: cluster.py"""

import json
import pytest
import numpy as np
from src.synthetic.cluster import (
    kcenter_greedy,
    embed_examples,
    select_coreset,
    select_by_feature_diversity,
    _embed_text,
    _cosine_distance,
    _euclidean_distance,
)


class TestEmbedText:
    def test_returns_fixed_dimension(self):
        emb = _embed_text("test text", dim=768)
        assert len(emb) == 768
        assert isinstance(emb, np.ndarray)

    def test_deterministic(self):
        emb1 = _embed_text("same text")
        emb2 = _embed_text("same text")
        assert np.allclose(emb1, emb2)

    def test_different_texts_different_embeddings(self):
        emb1 = _embed_text("hello world")
        emb2 = _embed_text("completely different content")
        assert not np.allclose(emb1, emb2)

    def test_normalized(self):
        emb = _embed_text("test", dim=64)
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-6


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


class TestEmbedExamples:
    def test_requires_content_field(self):
        examples = [
            {"instruction": "task 1", "input": "code 1"},
            {"instruction": "task 2", "input": "code 2"},
        ]
        embeddings = embed_examples(examples, embedding_dim=64)
        assert embeddings.shape == (2, 64)

    def test_empty_handled(self):
        embeddings = embed_examples([], embedding_dim=64)
        assert embeddings.shape == (0, 64)


class TestSelectCoreset:
    def test_small_dataset_unchanged(self):
        examples = [{"instruction": f"task {i}", "input": "code"} for i in range(5)]
        selected = select_coreset(examples, target_size=10)
        assert len(selected) == 5

    def test_large_dataset_reduced(self):
        examples = [{"instruction": f"task {i:04d}", "input": "code"} for i in range(100)]
        selected = select_coreset(examples, target_size=20)
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
