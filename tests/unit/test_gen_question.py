"""Tests for synthetic data generation: gen_question.py"""

import pytest
from src.codeflow.synthetic.construct_tree import (
    build_baseline_tree,
    assign_features_to_tree,
    FeatureTreeNode,
)
from src.codeflow.synthetic.gen_question import (
    GeneratedTask,
    _build_task_prompt,
    _generate_fallback_tasks,
    _split_budget,
    generate_tasks_from_node,
    generate_tasks_from_tree,
    format_task_for_training,
    filter_tasks_by_difficulty,
)
from src.codeflow.synthetic.construct_tree import FeatureTree


class TestGeneratedTask:
    def test_from_dict(self):
        d = {
            "type": "bug-fix",
            "instruction": "Fix nil handling in parser",
            "feature_used": "nil-punning",
            "difficulty": "intermediate",
            "context_files": ["src/parser.clj"],
            "expected_changes": "Add nil checks",
        }
        task = GeneratedTask.from_dict(d)
        assert task.task_type == "bug-fix"
        assert task.instruction == "Fix nil handling in parser"
        assert task.feature_used == "nil-punning"
        assert task.difficulty == "intermediate"
        assert task.context_files == ["src/parser.clj"]

    def test_to_dict(self):
        task = GeneratedTask(
            task_type="refactor",
            instruction="Refactor to use transducers",
            feature_used="transducers",
            difficulty="advanced",
            context_files=["src/pipeline.clj"],
            expected_changes="Replace map/filter chains with transducers",
        )
        d = task.to_dict()
        assert d["type"] == "refactor"
        assert d["instruction"] == "Refactor to use transducers"
        assert d["difficulty"] == "advanced"

    def test_defaults(self):
        task = GeneratedTask.from_dict({})
        assert task.task_type == "add-feature"
        assert task.instruction == ""
        assert task.difficulty == "intermediate"


class TestFallbackTasks:
    def test_generates_tasks(self):
        node = FeatureTreeNode(
            name="metaprogramming/macros",
            label="Macros",
            description="Custom syntax extensions",
            parent="metaprogramming",
            depth=1,
        )
        tasks = _generate_fallback_tasks(node, count=3)
        assert len(tasks) == 3
        assert all(isinstance(t, GeneratedTask) for t in tasks)
        assert all(t.instruction for t in tasks)
        assert all(t.feature_used for t in tasks)

    def test_respects_count(self):
        node = FeatureTreeNode(
            name="concurrency/atoms-refs",
            label="Atoms & Refs",
            description="Mutable state",
            parent="concurrency",
            depth=1,
        )
        tasks = _generate_fallback_tasks(node, count=1)
        assert len(tasks) == 1


class TestTaskPrompt:
    def test_builds_prompt(self):
        node = FeatureTreeNode(
            name="metaprogramming/macros",
            label="Macros",
            description="Custom syntax extensions",
            parent="metaprogramming",
            depth=1,
            features=[
                {"feature_type": "macros", "name": "defmacro", "description": "Define macro"},
                {"feature_type": "macros", "name": "syntax-quote", "description": "Syntax quoting"},
            ],
        )
        prompt = _build_task_prompt(node.features, node, count=2)
        assert "Macros" in prompt
        assert "defmacro" in prompt
        assert "syntax-quote" in prompt
        assert len(prompt) > 100


class TestFormatTask:
    def test_formats_simple_task(self):
        task = GeneratedTask(
            task_type="bug-fix",
            instruction="Fix nil handling in parse-args",
            feature_used="nil-punning",
            context_files=["src/parser.clj", "src/utils.clj"],
        )
        formatted = format_task_for_training(task)
        assert "Fix nil handling" in formatted
        assert "src/parser.clj" in formatted
        assert "src/utils.clj" in formatted

    def test_formats_task_without_files(self):
        task = GeneratedTask(
            task_type="add-feature",
            instruction="Add validation",
            feature_used="spec",
        )
        formatted = format_task_for_training(task)
        assert "Add validation" in formatted


class TestFilterTasks:
    def test_filters_by_difficulty(self):
        tasks = [
            GeneratedTask(task_type="bug-fix", instruction="t1", feature_used="f1", difficulty="beginner"),
            GeneratedTask(task_type="bug-fix", instruction="t2", feature_used="f2", difficulty="intermediate"),
            GeneratedTask(task_type="bug-fix", instruction="t3", feature_used="f3", difficulty="advanced"),
        ]
        filtered = filter_tasks_by_difficulty(tasks, ["beginner", "intermediate"])
        assert len(filtered) == 2
        assert all(t.difficulty in ("beginner", "intermediate") for t in filtered)


class TestSplitBudget:
    def test_even_split(self):
        assert _split_budget(10, 2) == [5, 5]

    def test_uneven_split_front_loaded(self):
        assert _split_budget(10, 3) == [4, 3, 3]
        assert sum(_split_budget(10, 3)) == 10

    def test_zero_parts(self):
        assert _split_budget(10, 0) == []


def _two_node_tree(dense_features: int, sparse_features: int) -> FeatureTree:
    """A tree with one dense and one sparse subcategory node (depth 1)."""
    tree = FeatureTree(name="t")
    feat = lambda i: {"feature_type": "x", "name": f"f{i}", "description": "d"}
    tree.nodes["dense"] = FeatureTreeNode(
        name="dense", depth=1, node_type="subcategory",
        features=[feat(i) for i in range(dense_features)],
    )
    tree.nodes["sparse"] = FeatureTreeNode(
        name="sparse", depth=1, node_type="subcategory",
        features=[feat(i) for i in range(sparse_features)],
    )
    return tree


class TestGenerateTasksTemperature:
    """Reweighting controls how the task budget spreads across nodes."""

    def _allocation(self, monkeypatch, tree, **kwargs):
        # Stub the per-node generator to just record how many tasks each node
        # was asked for, so we observe the allocation without calling an LLM.
        calls = {}

        def fake_node_gen(node, llm, count):
            calls[node.name] = calls.get(node.name, 0) + count
            return [
                GeneratedTask(task_type="feature", instruction=f"{node.name}-{i}",
                              feature_used="f")
                for i in range(count)
            ]

        monkeypatch.setattr(
            "src.codeflow.synthetic.gen_question.generate_tasks_from_node",
            fake_node_gen,
        )
        generate_tasks_from_tree(tree, llm=None, tasks_per_node=0, **kwargs)
        return calls

    def test_high_temperature_boosts_sparse_node(self, monkeypatch):
        tree = _two_node_tree(dense_features=30, sparse_features=1)
        cold = self._allocation(monkeypatch, tree, max_total=40, temperature=0.5)
        hot = self._allocation(monkeypatch, tree, max_total=40, temperature=8.0)
        # The rare node gets a larger share of the budget when smoothed.
        assert hot.get("sparse", 0) > cold.get("sparse", 0)
        assert hot.get("dense", 0) < cold.get("dense", 0)

    def test_empty_tree_returns_no_tasks(self, monkeypatch):
        tree = FeatureTree(name="empty")
        assert self._allocation(monkeypatch, tree, max_total=10) == {}

    def test_multiple_temperatures_mix(self, monkeypatch):
        tree = _two_node_tree(dense_features=30, sparse_features=1)
        mixed = self._allocation(
            monkeypatch, tree, max_total=40, temperatures=[0.5, 8.0]
        )
        # Both nodes get work when mixing a sharp and a smooth temperature.
        assert mixed.get("dense", 0) > 0
        assert mixed.get("sparse", 0) > 0
