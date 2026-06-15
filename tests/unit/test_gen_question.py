"""Tests for synthetic data generation: gen_question.py"""

import pytest
from src.synthetic.construct_tree import (
    build_baseline_tree,
    assign_features_to_tree,
    FeatureTreeNode,
)
from src.synthetic.gen_question import (
    GeneratedTask,
    _build_task_prompt,
    _generate_fallback_tasks,
    generate_tasks_from_node,
    format_task_for_training,
    filter_tasks_by_difficulty,
)


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
