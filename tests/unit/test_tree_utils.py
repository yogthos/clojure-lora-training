"""Tests for synthetic data generation: tree_utils.py"""

import pytest
from src.codeflow.synthetic.construct_tree import (
    build_baseline_tree,
    assign_features_to_tree,
    FeatureTreeNode,
)
from src.codeflow.synthetic.tree_utils import (
    sample_nodes,
    sample_features,
    compute_feature_frequency,
    compute_node_coverage,
    find_sparse_nodes,
    compute_tree_diversity,
    generate_training_split,
    reweight_frequencies,
    allocate_by_reweighted_frequency,
)


class TestReweightFrequencies:
    """EpiCoder Eq 1: p'_i = exp(log p_i / t) / sum_j exp(log p_j / t)."""

    def test_sums_to_one(self):
        p = reweight_frequencies([3, 1, 1], t=1.5)
        assert abs(sum(p) - 1.0) < 1e-9

    def test_t1_reproduces_normalized_frequency(self):
        # t=1 leaves the distribution unchanged.
        p = reweight_frequencies([3, 1, 1], t=1.0)
        assert p == pytest.approx([0.6, 0.2, 0.2])

    def test_high_t_flattens_toward_uniform(self):
        # A large temperature smooths the distribution: the dominant feature
        # loses share, the rare ones gain — the whole point of reweighting.
        base = reweight_frequencies([8, 1, 1], t=1.0)
        hot = reweight_frequencies([8, 1, 1], t=5.0)
        assert hot[0] < base[0]          # dominant downweighted
        assert hot[1] > base[1]          # rare upweighted
        # closer to uniform (1/3 each)
        assert abs(hot[0] - 1 / 3) < abs(base[0] - 1 / 3)

    def test_low_t_sharpens(self):
        cold = reweight_frequencies([8, 1, 1], t=0.5)
        base = reweight_frequencies([8, 1, 1], t=1.0)
        assert cold[0] > base[0]         # dominant gets even more

    def test_very_high_t_approaches_uniform(self):
        p = reweight_frequencies([100, 1, 1], t=1e6)
        assert p == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-3)

    def test_zero_frequency_stays_zero(self):
        p = reweight_frequencies([4, 0, 4], t=2.0)
        assert p[1] == 0.0
        assert p[0] == pytest.approx(p[2])

    def test_all_zero_is_uniform(self):
        p = reweight_frequencies([0, 0, 0, 0], t=1.0)
        assert p == pytest.approx([0.25, 0.25, 0.25, 0.25])

    def test_empty(self):
        assert reweight_frequencies([], t=1.0) == []

    def test_single(self):
        assert reweight_frequencies([5], t=2.0) == pytest.approx([1.0])

    def test_nonpositive_t_raises(self):
        with pytest.raises(ValueError):
            reweight_frequencies([1, 2], t=0.0)


class TestAllocateByReweightedFrequency:
    def test_sums_to_total(self):
        counts = allocate_by_reweighted_frequency([10, 1, 1], total=20, t=1.5)
        assert sum(counts) == 20
        assert len(counts) == 3

    def test_high_t_more_even_than_low(self):
        cold = allocate_by_reweighted_frequency([20, 1, 1], total=30, t=0.5)
        hot = allocate_by_reweighted_frequency([20, 1, 1], total=30, t=8.0)
        # The dominant bucket gets a smaller share at high temperature.
        assert hot[0] < cold[0]
        assert hot[1] >= cold[1]

    def test_min_each_floor_is_respected(self):
        counts = allocate_by_reweighted_frequency(
            [100, 0, 0], total=10, t=1.0, min_each=1
        )
        assert all(c >= 1 for c in counts)
        assert sum(counts) == 10

    def test_zero_total(self):
        assert allocate_by_reweighted_frequency([1, 2, 3], total=0, t=1.0) == [0, 0, 0]

    def test_empty(self):
        assert allocate_by_reweighted_frequency([], total=5, t=1.0) == []


class TestSampleNodes:
    def test_samples_from_baseline_tree(self):
        tree = build_baseline_tree()
        nodes = sample_nodes(tree, count=5)
        assert 1 <= len(nodes) <= 5
        assert all(isinstance(n, FeatureTreeNode) for n in nodes)

    def test_min_depth_filter(self):
        tree = build_baseline_tree()
        # min_depth=0 includes root nodes, min_depth=1 excludes them
        eligible_all = [n for n in tree.nodes.values() if n.depth >= 0]
        eligible_deep = [n for n in tree.nodes.values() if n.depth >= 1]
        assert len(eligible_all) > len(eligible_deep)
        # When sampling all eligible nodes, depth 0 ones must be present
        all_nodes = sample_nodes(tree, count=len(eligible_all), min_depth=0)
        root_nodes = [n for n in all_nodes if n.depth == 0]
        assert len(root_nodes) > 0

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
        from src.codeflow.synthetic.construct_tree import FeatureTree
        tree = FeatureTree()
        features = sample_features(tree, count=5)
        assert features == []

    def test_min_depth_too_high(self):
        tree = build_baseline_tree()
        features = sample_features(tree, count=5, min_depth=10)
        assert features == []
