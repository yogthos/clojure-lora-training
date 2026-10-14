"""Tests for synthetic data generation: feature_evol.py"""

import pytest
from src.codeflow.synthetic.construct_tree import (
    build_baseline_tree,
    assign_features_to_tree,
    FeatureTree,
)
from src.codeflow.synthetic.feature_evol import (
    EvolConfig,
    evolve_breadth,
    evolve_depth,
    evolve_detail,
    merge_evolved_trees,
    sample_feature_names,
)


class _FakeLLM:
    """Returns a fixed JSON payload for require_json calls."""

    def __init__(self, payload):
        import json
        self._payload = json.dumps(payload)

    def call(self, system_prompt, user_prompt, temperature=None,
             max_tokens=None, require_json=False, reject_truncated=False):
        return self._payload


class TestEvolveAddsNodes:
    """Regression for the key-mismatch no-op: the LLM returns 'name'-keyed
    categories, but the code read 'category' and silently dropped them all."""

    def test_breadth_adds_name_keyed_categories(self):
        tree = build_baseline_tree()
        before = len(tree.nodes)
        llm = _FakeLLM([
            {"name": "web-development", "description": "Ring/Reitit handlers"},
            {"name": "state-management", "description": "atoms and watchers"},
        ])
        evolved = evolve_breadth(tree, llm, EvolConfig())
        assert len(evolved.nodes) > before
        assert "web-development" in evolved.nodes
        assert "state-management" in evolved.nodes

    def test_depth_adds_name_keyed_subcategories(self):
        tree = build_baseline_tree()
        before = len(tree.nodes)
        # Each existing root gets a new subcategory keyed 'name'.
        llm = _FakeLLM([
            {"new_subcategories": [{"name": "transducers-advanced",
                                    "description": "stateful xforms"}]}
        ])
        evolved = evolve_depth(tree, llm, EvolConfig())
        assert len(evolved.nodes) > before

    def test_breadth_skips_existing_and_blank(self):
        tree = build_baseline_tree()
        first_root = next(n for n, node in tree.nodes.items() if node.depth == 0)
        before = len(tree.nodes)
        llm = _FakeLLM([
            {"name": first_root},          # already exists -> skip
            {"description": "no name"},     # blank id -> skip
        ])
        evolved = evolve_breadth(tree, llm, EvolConfig())
        assert len(evolved.nodes) == before


class TestEvolConfig:
    def test_defaults(self):
        config = EvolConfig()
        assert config.max_breadth_nodes > 0
        assert config.max_depth_nodes > 0
        assert config.max_detail_features > 0

    def test_custom(self):
        config = EvolConfig(
            max_breadth_nodes=5,
            max_depth_nodes=10,
            llm_temperature=0.9,
        )
        assert config.max_breadth_nodes == 5
        assert config.llm_temperature == 0.9


class TestMergeEvolvedTrees:
    def test_merge_single_tree(self):
        tree = build_baseline_tree()
        features = [
            {"feature_type": "macros", "name": "m1", "description": "d1"},
        ]
        tree = assign_features_to_tree(features, tree)
        merged = merge_evolved_trees([tree])
        assert len(merged.nodes) == len(tree.nodes)

    def test_merge_deduplicates_features(self):
        tree1 = build_baseline_tree()
        tree2 = build_baseline_tree()

        features1 = [
            {"feature_type": "macros", "name": "my-macro", "description": "shared macro"},
        ]
        features2 = [
            {"feature_type": "macros", "name": "my-macro", "description": "shared macro"},
            {"feature_type": "macros", "name": "other-macro", "description": "unique macro"},
        ]

        tree1 = assign_features_to_tree(features1, tree1)
        tree2 = assign_features_to_tree(features2, tree2)

        merged = merge_evolved_trees([tree1, tree2])
        macro_node = merged.nodes.get("metaprogramming/macros")
        assert macro_node is not None
        # my-macro should only appear once despite being in both trees
        names = [f.get("name") for f in macro_node.features]
        assert names.count("my-macro") == 1

    def test_merge_adds_new_nodes(self):
        tree1 = build_baseline_tree()
        tree2 = build_baseline_tree()

        # Add a synthetic node to tree2
        tree2.nodes["custom-test/experimental"] = build_baseline_tree().nodes["metaprogramming/macros"]
        tree2.nodes["custom-test/experimental"].name = "custom-test/experimental"
        tree2.nodes["custom-test/experimental"].parent = "custom-test"

        merged = merge_evolved_trees([tree1, tree2])
        assert "custom-test/experimental" in merged.nodes

    def test_merge_empty(self):
        merged = merge_evolved_trees([])
        assert len(merged.nodes) > 0  # Returns baseline


class TestSampleFeatureNames:
    def test_samples_from_populated_tree(self):
        tree = build_baseline_tree()
        features = [
            {"feature_type": "macros", "name": f"macro-{i}", "description": f"d{i}"}
            for i in range(10)
        ] + [
            {"feature_type": "atoms", "name": f"atom-{i}", "description": f"d{i}"}
            for i in range(10)
        ]
        tree = assign_features_to_tree(features, tree)

        names = sample_feature_names(tree, count=5)
        assert 1 <= len(names) <= 5
        assert all(isinstance(n, str) for n in names)

    def test_samples_at_most_available(self):
        tree = build_baseline_tree()
        features = [
            {"feature_type": "macros", "name": "only-macro", "description": "d1"},
        ]
        tree = assign_features_to_tree(features, tree)

        names = sample_feature_names(tree, count=50)
        assert len(names) <= 1

    def test_min_depth_filter(self):
        tree = build_baseline_tree()
        features = [
            {"feature_type": "macros", "name": "m1", "description": "d1"},
        ]
        tree = assign_features_to_tree(features, tree)

        # min_depth=2 should yield nothing since features are at depth 1
        names = sample_feature_names(tree, count=10, min_depth=2)
        assert len(names) == 0


import json as _json

from src.codeflow.synthetic.construct_tree import FeatureTreeNode
from src.codeflow.synthetic.prompts import BREADTH_SYSTEM, DEPTH_SYSTEM, DETAIL_SYSTEM


class _RecordingLLM:
    """Returns a fixed object reply and records which prompts it was sent."""

    def __init__(self, payload=None, exc=None):
        self.payload, self.exc = payload, exc
        self.prompts = []

    def call(self, system_prompt, user_prompt, **kw):
        self.prompts.append(user_prompt)
        if self.exc:
            raise self.exc
        return _json.dumps(self.payload)


class TestEvolutionShapes:
    """Prompts and parsing must agree on one reply shape per evolution type."""

    def test_breadth_object_reply_adds_root_with_children(self):
        tree = build_baseline_tree()
        llm = _RecordingLLM({"categories": [
            {"name": "observability", "description": "logging and metrics",
             "subcategories": [{"name": "structured-logging", "description": "maps as logs"}]},
        ]})
        evolved = evolve_breadth(tree, llm, EvolConfig())
        assert evolved.nodes["observability"].children == ["observability/structured-logging"]
        assert evolved.nodes["observability/structured-logging"].depth == 1

    def test_breadth_prompt_asks_for_categories_with_subcategories(self):
        assert '"categories"' in BREADTH_SYSTEM
        assert '"subcategories"' in BREADTH_SYSTEM
        assert "parent_features" not in BREADTH_SYSTEM

    def test_depth_prompt_asks_for_subcategories_object(self):
        assert '"subcategories"' in DEPTH_SYSTEM
        assert "JSON array" not in DEPTH_SYSTEM

    def test_detail_object_reply_adds_named_features(self):
        tree = build_baseline_tree()
        for name, node in tree.nodes.items():
            if node.depth > 0:
                node.features = [{"name": f"x{i}"} for i in range(5)]  # not sparse
        target = tree.nodes["metaprogramming/macros"]
        target.features = []
        llm = _RecordingLLM({"features": [
            {"feature_type": "macros", "name": "with-retry", "description": "retry macro"},
            {"description": "nameless"},
        ]})
        evolve_detail(tree, llm, EvolConfig())
        assert [f["name"] for f in target.features] == ["with-retry"]

    def test_detail_prompt_asks_for_features_not_namespace_spec(self):
        assert '"features"' in DETAIL_SYSTEM
        assert '"namespace"' not in DETAIL_SYSTEM


class TestDepthCoversEvolvedRoots:
    """evolve_depth only ever looked at root_nodes[:5] — always baseline roots."""

    def test_evolved_root_without_children_gets_subcategories(self):
        tree = build_baseline_tree()
        tree.nodes["observability"] = FeatureTreeNode(
            name="observability", label="Observability", depth=0)
        tree.root_count += 1
        llm = _RecordingLLM({"subcategories": [{"name": "tracing", "description": "spans"}]})
        evolved = evolve_depth(tree, llm, EvolConfig())
        assert "observability/tracing" in evolved.nodes
        assert evolved.nodes["observability"].children == ["observability/tracing"]


class TestEvolutionLLMFailures:
    @pytest.mark.parametrize("fn", [evolve_breadth, evolve_depth, evolve_detail])
    def test_auth_error_propagates(self, fn):
        from src.llm.provider import LLMAuthError
        with pytest.raises(LLMAuthError):
            fn(build_baseline_tree(), _RecordingLLM(exc=LLMAuthError("401")), EvolConfig())

    @pytest.mark.parametrize("fn", [evolve_breadth, evolve_depth, evolve_detail])
    def test_llm_error_leaves_tree_unchanged(self, fn):
        from src.llm.provider import LLMError
        tree = build_baseline_tree()
        before = len(tree.nodes)
        out = fn(tree, _RecordingLLM(exc=LLMError("503")), EvolConfig())
        assert len(out.nodes) == before
        assert sum(len(n.features) for n in out.nodes.values()) == 0
