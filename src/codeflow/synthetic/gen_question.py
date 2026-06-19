"""Generate Clojure coding tasks from feature tree nodes.

Adapted from EpiCoder's gen/gen_question.py. Walks the feature taxonomy
tree and generates realistic Clojure coding tasks (instructions) that
would require using specific features or patterns.

Each task simulates a real development scenario: a user asks for help
with a Clojure problem, and the coding agent responds with REPL-driven
exploration and code changes.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ...llm.provider import LLMProvider
from .construct_tree import FeatureTree, FeatureTreeNode


# System prompt for question generation — Clojure-specific
from .prompts import QUESTION_SYSTEM as _QUESTION_SYSTEM


@dataclass
class GeneratedTask:
    """A synthetic coding task for training."""
    task_type: str  # bug-fix, refactor, add-feature, optimize, repl-explore
    instruction: str
    feature_used: str
    difficulty: str = "intermediate"
    context_files: List[str] = field(default_factory=list)
    expected_changes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "GeneratedTask":
        return cls(
            task_type=d.get("type", "add-feature"),
            instruction=d.get("instruction", ""),
            feature_used=d.get("feature_used", ""),
            difficulty=d.get("difficulty", "intermediate"),
            context_files=d.get("context_files", []),
            expected_changes=d.get("expected_changes", ""),
        )

    def to_dict(self) -> dict:
        return {
            "type": self.task_type,
            "instruction": self.instruction,
            "feature_used": self.feature_used,
            "difficulty": self.difficulty,
            "context_files": self.context_files,
            "expected_changes": self.expected_changes,
        }


# Task type templates for Clojure scenarios
_TASK_TEMPLATES = {
    "bug-fix": [
        "A user reports that {feature_desc}. Current code at {file} doesn't handle {edge_case}. "
        "Investigate the issue using the REPL, find the root cause, and fix it.",

        "The function {feature_name} in {file} has unexpected behavior when called with {input_type}. "
        "Use the REPL to reproduce, diagnose, and fix the problem.",

        "After a dependency update, {feature_name} broke. The error is {error_desc}. "
        "Explore the updated API in the REPL and adapt the code.",
    ],
    "refactor": [
        "Refactor {feature_name} in {file} to use {pattern}. "
        "Use the REPL to test the refactored version interactively before applying changes.",

        "The code in {file} duplicates {feature_name} logic across several functions. "
        "Extract shared behavior using {pattern}. Verify equivalence via REPL evaluation.",

        "Update {file} to follow Clojure idioms for {feature_desc}. "
        "Explore alternatives in the REPL before settling on a final approach.",
    ],
    "add-feature": [
        "Add {feature_desc} to the codebase. Start by exploring the existing code in the REPL "
        "to understand the current architecture, then implement the new feature interactively.",

        "We need {feature_desc} in {file}. This should use {feature_name}. "
        "Prototype the implementation in the REPL, test with sample inputs, then apply.",

        "Extend the system with {feature_desc}. The feature should leverage {feature_name}. "
        "Use the REPL to incrementally build and test the implementation.",
    ],
    "optimize": [
        "Profile {feature_name} in {file} using the REPL. Identify performance bottlenecks "
        "and optimize using {pattern}. Verify correctness after optimization.",

        "The {feature_desc} is too slow for large inputs. Use the REPL to measure, "
        "then apply {pattern} to improve performance while maintaining correctness.",

        "Optimize memory usage in {feature_name} by switching to {pattern}. "
        "Use the REPL to compare before/after behavior and memory profiles.",
    ],
    "repl-explore": [
        "Explore how {feature_name} works in the REPL. Understand its behavior with "
        "different inputs, edge cases, and interactions. Document findings and suggest improvements.",

        "Investigate the {feature_desc} by evaluating expressions in the REPL. "
        "Identify any design issues and propose refactoring using {feature_name}.",

        "Use the REPL to understand the data flow through {feature_desc}. "
        "Map out the transformation pipeline and identify opportunities for using {pattern}.",
    ],
}


def _build_task_prompt(
    features: List[dict],
    node: FeatureTreeNode,
    count: int = 3,
) -> str:
    """Build a prompt for generating tasks from a feature tree node.

    Uses template-based seeding for more consistent generation.
    """
    feature_names = [
        f.get("name", node.label) for f in features[:5]
    ] if features else [node.label]

    # Pick diverse templates
    templates = []
    for task_type, type_templates in _TASK_TEMPLATES.items():
        if len(templates) >= count:
            break
        templates.append((task_type, random.choice(type_templates)))

    seed_tasks = []
    for task_type, tmpl in templates:
        feat_name = random.choice(feature_names)
        seed = tmpl.format(
            feature_desc=node.description or node.label,
            feature_name=feat_name,
            file=random.choice(["src/core.clj", f"src/{node.label.lower().replace(' ', '_')}.clj"]),
            pattern="Clojure idioms appropriate for this context",
            edge_case=random.choice(["nil inputs", "empty collections", "large datasets", "concurrent access", "malformed data"]),
            input_type=random.choice(["nil", "empty map", "large vector", "lazy seq", "nested data"]),
            error_desc=random.choice(["NullPointerException", "ClassCastException", "stack overflow", "arity mismatch", "spec validation failure"]),
        )
        seed_tasks.append({"type": task_type, "seed": seed})

    prompt = (
        f"Feature category: {node.label}\n"
        f"Description: {node.description}\n"
        f"Available features: {json.dumps(feature_names)}\n\n"
        f"Seed ideas (expand and make realistic):\n"
        f"{json.dumps(seed_tasks, indent=2)}\n\n"
        f"Generate {count} realistic Clojure coding tasks based on these seeds. "
        f"Make each task specific, concrete, and suitable for REPL-driven development. "
        f"Output as JSON array."
    )
    return prompt


def generate_tasks_from_node(
    node: FeatureTreeNode,
    llm: LLMProvider,
    count: int = 3,
) -> List[GeneratedTask]:
    """Generate coding tasks for a single taxonomy node.

    Args:
        node: The feature tree node to generate tasks for.
        llm: LLM provider.
        count: Number of tasks to generate.

    Returns:
        List of GeneratedTask objects.
    """
    features = node.features
    prompt = _build_task_prompt(features, node, count)

    try:
        result = llm.call(
            system_prompt=_QUESTION_SYSTEM,
            user_prompt=prompt,
            temperature=0.8,
            max_tokens=4096,
            require_json=True,
        )
        items = json.loads(result) if isinstance(result, str) else result
        if isinstance(items, dict):
            items = [items]
    except Exception:
        return _generate_fallback_tasks(node, count)

    tasks = []
    for item in items:
        if not isinstance(item, dict):
            continue
        task = GeneratedTask.from_dict(item)
        if task.instruction:
            tasks.append(task)

    return tasks[:count]


def _generate_fallback_tasks(
    node: FeatureTreeNode,
    count: int,
) -> List[GeneratedTask]:
    """Generate basic fallback tasks without LLM."""
    tasks = []
    task_types = list(_TASK_TEMPLATES.keys())

    for i in range(min(count, len(task_types))):
        ttype = task_types[i % len(task_types)]
        feat_desc = node.description or node.label

        tasks.append(GeneratedTask(
            task_type=ttype,
            instruction=(
                f"In {node.label}, implement improvements using Clojure idioms. "
                f"Use the REPL to explore, test, and iterate before applying changes. "
                f"The feature area is: {feat_desc}."
            ),
            feature_used=node.label,
            difficulty="intermediate",
            context_files=[f"src/{node.label.lower().replace(' ', '_')}.clj"],
            expected_changes=f"Changes to {node.label} functionality using Clojure patterns",
        ))

    return tasks


def generate_tasks_from_tree(
    tree: FeatureTree,
    llm: LLMProvider,
    tasks_per_node: int = 2,
    max_total: int = 50,
    min_depth: int = 1,
) -> List[GeneratedTask]:
    """Generate tasks from across the feature taxonomy tree.

    Walks nodes at min_depth or deeper, generating tasks proportional
    to each node's feature count.

    Args:
        tree: The populated feature taxonomy.
        llm: LLM provider.
        tasks_per_node: Base tasks per node (scaled by feature count).
        max_total: Maximum total tasks to generate.
        min_depth: Minimum node depth to generate from.

    Returns:
        List of GeneratedTask objects, shuffled for diversity.
    """
    all_tasks = []

    # Collect eligible nodes, weighted by feature count
    nodes_with_features = [
        (name, node) for name, node in tree.nodes.items()
        if node.depth >= min_depth
    ]
    nodes_with_features.sort(
        key=lambda x: len(x[1].features), reverse=True
    )

    for node_name, node in nodes_with_features:
        if len(all_tasks) >= max_total:
            break

        # Scale: more features → more tasks
        feature_count = max(len(node.features), 1)
        n = min(tasks_per_node * max(1, feature_count // 3), max_total - len(all_tasks))
        if n <= 0:
            continue

        tasks = generate_tasks_from_node(node, llm, count=n)
        all_tasks.extend(tasks)

    random.shuffle(all_tasks)
    return all_tasks[:max_total]


def filter_tasks_by_difficulty(
    tasks: List[GeneratedTask],
    difficulties: List[str],
) -> List[GeneratedTask]:
    """Filter tasks to only those matching the given difficulty levels."""
    return [t for t in tasks if t.difficulty in difficulties]


def format_task_for_training(task: GeneratedTask) -> str:
    """Format a task as a training instruction string.

    Mimics a user asking for help in a Clojure development context.
    """
    parts = [task.instruction]

    if task.context_files:
        parts.append(f"\nRelevant files: {', '.join(task.context_files)}")

    return " ".join(parts)
