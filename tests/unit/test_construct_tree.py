"""Tests for synthetic data generation: construct_tree.py"""

import json
import pytest
from src.synthetic.construct_tree import (
    FeatureTreeNode,
    FeatureTree,
    build_baseline_tree,
    assign_features_to_tree,
    tree_to_json,
    tree_from_json,
    get_tree_statistics,
)


class TestBuildBaselineTree:
    def test_creates_root_categories(self):
        tree = build_baseline_tree()
        assert tree.root_count >= 10
        assert "metaprogramming" in tree.nodes
        assert "concurrency" in tree.nodes
        assert "polymorphism" in tree.nodes
        assert "repl-development" in tree.nodes

    def test_root_nodes_have_children(self):
        tree = build_baseline_tree()
        meta = tree.nodes["metaprogramming"]
        assert len(meta.children) > 0
        assert meta.depth == 0
        assert meta.node_type == "category"

    def test_subcategories_have_parent(self):
        tree = build_baseline_tree()
        sub = tree.nodes.get("metaprogramming/macros")
        assert sub is not None
        assert sub.parent == "metaprogramming"
        assert sub.depth == 1
        assert sub.node_type == "subcategory"


class TestAssignFeatures:
    def test_assigns_macro_feature(self):
        tree = build_baseline_tree()
        features = [
            {"feature_type": "macros", "name": "my-macro", "description": "A macro"}
        ]
        tree = assign_features_to_tree(features, tree)
        sub = tree.nodes.get("metaprogramming/macros")
        assert sub is not None
        assert len(sub.features) >= 1

    def test_assigns_protocol_feature(self):
        tree = build_baseline_tree()
        features = [
            {"feature_type": "protocols", "name": "Storage", "description": "Storage protocol"}
        ]
        tree = assign_features_to_tree(features, tree)
        sub = tree.nodes.get("polymorphism/protocols")
        assert sub is not None
        assert len(sub.features) >= 1

    def test_assigns_concurrency_feature(self):
        tree = build_baseline_tree()
        features = [
            {"feature_type": "atoms", "name": "state", "description": "atom state"},
            {"feature_type": "core-async", "name": "processor", "description": "go-loop processor"},
        ]
        tree = assign_features_to_tree(features, tree)
        atoms_node = tree.nodes.get("concurrency/atoms-refs")
        async_node = tree.nodes.get("concurrency/core-async")
        assert atoms_node is not None
        assert async_node is not None

    def test_unknown_feature_goes_to_default(self):
        tree = build_baseline_tree()
        features = [
            {"feature_type": "something-entirely-new", "name": "weird", "description": "unknown"}
        ]
        tree = assign_features_to_tree(features, tree)
        # Should go to default bucket (data-transformation)
        # Verify some node got the feature
        total = sum(len(n.features) for n in tree.nodes.values())
        assert total == 1


class TestTreeSerialization:
    def test_roundtrip(self):
        tree = build_baseline_tree()
        features = [
            {"feature_type": "macros", "name": "my-macro", "description": "A macro"},
            {"feature_type": "atoms", "name": "state", "description": "atom state"},
        ]
        tree = assign_features_to_tree(features, tree)

        data = tree_to_json(tree)
        restored = tree_from_json(data)

        assert restored.name == tree.name
        assert len(restored.nodes) == len(tree.nodes)
        assert "metaprogramming/macros" in restored.nodes
        assert len(restored.nodes["metaprogramming/macros"].features) >= 1

    def test_json_contains_required_fields(self):
        tree = build_baseline_tree()
        data = tree_to_json(tree)
        assert "name" in data
        assert "description" in data
        assert "nodes" in data
        node_data = list(data["nodes"].values())[0]
        assert "label" in node_data
        assert "depth" in node_data
        assert "children" in node_data
        assert "features" in node_data


class TestTreeStatistics:
    def test_stats_intact_tree(self):
        tree = build_baseline_tree()
        stats = get_tree_statistics(tree)
        assert stats["total_nodes"] > 10
        assert stats["total_root_categories"] >= 10
        assert "categories" in stats

    def test_stats_after_assignment(self):
        tree = build_baseline_tree()
        features = [
            {"feature_type": "macros", "name": "m1", "description": "d1"},
            {"feature_type": "protocols", "name": "p1", "description": "d2"},
            {"feature_type": "atoms", "name": "a1", "description": "d3"},
        ]
        tree = assign_features_to_tree(features, tree)
        stats = get_tree_statistics(tree)
        assert stats["total_features"] == 3
