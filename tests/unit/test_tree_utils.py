"""Tests for synthetic data generation: tree_utils.py"""

import pytest
from src.synthetic.construct_tree import (
    build_baseline_tree,
    assign_features_to_tree,
    FeatureTreeNode,
)
from src.synthetic.tree_utils import (
    sample_nodes,
    sample_features,
    compute_feature_frequency,
    compute_node_coverage,
    find_sparse_nodes,
    compute_tree_diversity,
    generate_training_split,
)


class TestSampleNodes:
    def test_samples_from_baseline_tree(self):
        tree = build_baseline_tree()
        nodes = sample_nodes(tree, count=5)
        assert 1 <= len(nodes) <= 5
        assert all(isinstance(n, FeatureTreeNode) for n in nodes)

    def test_min_depth_filter(self):
        tree = build_baseline_tree()
        # Only root nodes are at depth 0
        nodes = sample_nodes(tree, count=10, min_depth=0)
        root_nodes = [n for n in nodes if n.depth == 0]
        assert len(root_nodes) > 0

        # Only subcategories
        nodes = sample_nodes(tree, count=10, min_depth=1)
        assert all(n.depth >= 1 for n in nodes)

    def test_uniform_strategy(self):
        tree = build_baseline_tree()
        nodes = sample_nodes(tree, count=5, strategy="uniform")
        assert len(nodes) == 5

    def test_balanced_strategy(self):
        tree = build_baseline_tree()
        nodes = sample_nodes(tree, count=5, strategy="balanced")
        assert len(nodes) <= 5  # May be fewer if categories < count

    def test_all_strategies(self):
        tree = build_baseline_tree()
        for strategy in ("proportional", "uniform", "balanced"):
            nodes = sample_nodes(tree, count=3, strategy=strategy)
            assert len(nodes) <= 3

    def test_count_larger_than_eligible(self):
        tree = build_baseline_tree()
        nodes = sample_nodes(tree, count=1000)
        # Should return all eligible nodes
        eligible = [n for n in tree.nodes.values() if n.depth >= 1]
        assert len(nodes) == len(eligible)


class TestSampleFeatures:
    def test_samples_from_populated_tree(self):
        tree = build_baseline_tree()
        features = [
            {"feature_type": "macros", "name": "m1", "description": "d1"},
            {"feature_type": "macros", "name": "m2", "description": "d2"},
            {"feature_type": "atoms", "name": "a1", "description": "d3"},
        ]
        tree = assign_features_to_tree(features, tree)
        sampled = sample_features(tree, count=2)
        assert len(sampled) <= 2
        assert all(isinstance(f, dict) for f in sampled)
        assert all("name" in f for f in sampled)


class TestFrequency:
    def test_compute_from_multiple_trees(self):
        tree1 = build_baseline_tree()
        tree2 = build_baseline_tree()
        features1 = [
            {"feature_type": "macros", "name": "m1", "description": "d1"},
            {"feature_type": "atoms", "name": "a1", "description": "d2"},
        ]
        features2 = [
            {"feature_type": "macros", "name": "m2", "description": "d3"},
            {"feature_type": "protocols", "name": "p1", "description": "d4"},
        ]
        tree1 = assign_features_to_tree(features1, tree1)
        tree2 = assign_features_to_tree(features2, tree2)

        freq = compute_feature_frequency([tree1, tree2])
        assert freq["macros"] == 2
        assert freq["atoms"] == 1
        assert freq["protocols"] == 1


class TestNodeCoverage:
    def test_counts_per_node(self):
        tree = build_baseline_tree()
        features = [
            {"feature_type": "macros", "name": "m1", "description": "d1"},
            {"feature_type": "macros", "name": "m2", "description": "d2"},
        ]
        tree = assign_features_to_tree(features, tree)
        coverage = compute_node_coverage(tree)
        assert any(v >= 2 for v in coverage.values())


class TestSparseNodes:
    def test_finds_nodes_below_threshold(self):
        tree = build_baseline_tree()
        # Fresh tree has no features, many nodes are sparse
        sparse = find_sparse_nodes(tree, threshold=5)
        assert len(sparse) > 0

    def test_after_population(self):
        tree = build_baseline_tree()
        features = [
            {"feature_type": "macros", "name": f"m{i}", "description": f"d{i}"}
            for i in range(10)
        ]
        tree = assign_features_to_tree(features, tree)
        # macros should no longer be sparse
        sparse = find_sparse_nodes(tree, threshold=5)
        assert "metaprogramming/macros" not in sparse


class TestDiversity:
    def test_empty_tree(self):
        tree = build_baseline_tree()
        div = compute_tree_diversity(tree)
        assert 0.0 <= div <= 1.0

    def test_populated_tree(self):
        tree = build_baseline_tree()
        features = [
            {"feature_type": "macros", "name": "m1", "description": "d1"},
            {"feature_type": "atoms", "name": "a1", "description": "d2"},
            {"feature_type": "protocols", "name": "p1", "description": "d3"},
        ]
        tree = assign_features_to_tree(features, tree)
        div = compute_tree_diversity(tree)
        assert div > 0.0


class TestTrainingSplit:
    def test_generates_allocation(self):
        tree = build_baseline_tree()
        alloc = generate_training_split(tree, total_examples=100, min_per_node=2)
        assert len(alloc) > 0
        assert all(v >= 2 for v in alloc.values())

    def test_total_equals_requested(self):
        tree = build_baseline_tree()
        alloc = generate_training_split(tree, total_examples=100, min_per_node=2)
        total = sum(alloc.values())
        assert abs(total - 100) <= len(alloc)  # within rounding error


class TestSampleFeaturesEdgeCases:
    def test_empty_tree(self):
        from src.synthetic.construct_tree import FeatureTree
        tree = FeatureTree()
        features = sample_features(tree, count=5)
        assert features == []

    def test_min_depth_too_high(self):
        tree = build_baseline_tree()
        features = sample_features(tree, count=5, min_depth=10)
        assert features == []
