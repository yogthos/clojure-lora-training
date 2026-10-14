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

from ...llm.provider import LLMAuthError, LLMError, LLMProvider
from .construct_tree import FeatureTree, FeatureTreeNode
from .llm_json import parse_json_list


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


# Most tasks requested from one call. A reply cut off at max_tokens is broken
# JSON, and tasks carry inline code, so large counts are split into batches.
_MAX_TASKS_PER_CALL = 4


# Task type templates for Clojure scenarios. These seed the LLM, which writes
# the actual instruction. The answer starts from an empty REPL with no project,
# so seeds never point at files: existing code is shown inline in the task, and
# new behavior is described with literal example data.
_TASK_TEMPLATES = {
    "bug-fix": [
        "A function built around {feature_name} returns wrong results for {edge_case}. "
        "Show the buggy function inline and a call that misbehaves, and ask for it to be "
        "reproduced in the REPL, diagnosed, and fixed.",

        "Code using {feature_name} throws {error_desc} when given {input_type}. "
        "Show the offending function inline, state the expected behavior, and ask for a "
        "REPL-driven fix.",
    ],
    "refactor": [
        "Show a working but clumsy function (inline) for {feature_desc}. Ask for it to be "
        "refactored to use {feature_name} idiomatically, checking equivalence on sample "
        "data in the REPL.",

        "Show two or three small functions (inline) that duplicate logic around "
        "{feature_name}. Ask for the shared behavior to be extracted, verifying in the "
        "REPL that results are unchanged.",
    ],
    "add-feature": [
        "Describe a small, concrete piece of functionality for {feature_desc}, with "
        "example inputs and expected outputs, to be built with {feature_name}. Ask for "
        "it to be prototyped and tested in the REPL.",

        "Give the data shape as a literal example and the behavior needed for "
        "{feature_desc}. Ask for an implementation using {feature_name}, built up "
        "incrementally in the REPL.",
    ],
    "optimize": [
        "Show a correct but slow function (inline) related to {feature_desc}. Ask for it "
        "to be made faster for large inputs using {feature_name}, timing both versions in "
        "the REPL and checking the results match.",

        "Show a function (inline) that does redundant work with {feature_name}. Ask for "
        "the waste to be removed while keeping results identical, comparing old and new "
        "in the REPL.",
    ],
    "repl-explore": [
        "Ask for an interactive exploration of how {feature_name} behaves with different "
        "inputs, including {edge_case}, ending with a small helper that captures the findings.",

        "Show a short snippet (inline) using {feature_name} and ask for its data flow to "
        "be traced in the REPL step by step, then for a clearer version to be written.",
    ],
}


def _pick_task_types(count: int) -> List[str]:
    """Pick ``count`` task types, covering every type before repeating any.

    Taking types in dict order meant small counts only ever produced the first
    few (bug-fix, refactor), never optimize or repl-explore.
    """
    types: List[str] = []
    while len(types) < count:
        types.extend(random.sample(list(_TASK_TEMPLATES), len(_TASK_TEMPLATES)))
    return types[:count]


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

    seed_tasks = []
    for task_type in _pick_task_types(count):
        seed = random.choice(_TASK_TEMPLATES[task_type]).format(
            feature_desc=node.description or node.label,
            feature_name=random.choice(feature_names),
            edge_case=random.choice(["nil inputs", "empty collections", "large datasets", "concurrent access", "malformed data"]),
            input_type=random.choice(["nil", "an empty map", "a large vector", "a lazy seq", "nested data"]),
            error_desc=random.choice(["a NullPointerException", "a ClassCastException", "a stack overflow", "an arity exception", "an IllegalArgumentException"]),
        )
        seed_tasks.append({"type": task_type, "seed": seed})

    prompt = (
        f"Feature category: {node.label}\n"
        f"Description: {node.description}\n"
        f"Available features: {json.dumps(feature_names)}\n\n"
        f"Seed ideas (expand and make realistic):\n"
        f"{json.dumps(seed_tasks, indent=2)}\n\n"
        f"Generate {count} realistic Clojure coding tasks based on these seeds, one per seed. "
        f"Make each task specific, concrete, self-contained, and suitable for REPL-driven "
        f"development. Output a JSON object with the tasks under a \"tasks\" key."
    )
    return prompt


def generate_tasks_from_node(
    node: FeatureTreeNode,
    llm: LLMProvider,
    count: int = 3,
) -> List[GeneratedTask]:
    """Generate coding tasks for a single taxonomy node.

    Requests at most _MAX_TASKS_PER_CALL tasks per call. A failed or
    unparseable batch is dropped — no canned tasks are substituted, since a
    generic task isn't about this node and pollutes the training data.

    Args:
        node: The feature tree node to generate tasks for.
        llm: LLM provider.
        count: Number of tasks to generate.

    Returns:
        List of GeneratedTask objects.
    """
    tasks: List[GeneratedTask] = []
    for start in range(0, count, _MAX_TASKS_PER_CALL):
        n = min(_MAX_TASKS_PER_CALL, count - start)
        prompt = _build_task_prompt(node.features, node, n)
        try:
            result = llm.call(
                system_prompt=_QUESTION_SYSTEM,
                user_prompt=prompt,
                temperature=0.8,
                max_tokens=4096,
                require_json=True,
                reject_truncated=True,
            )
            items = parse_json_list(result, "tasks")
        except LLMAuthError:
            raise
        except (LLMError, ValueError):
            continue

        batch = [GeneratedTask.from_dict(item) for item in items]
        tasks.extend([
            t for t in batch if isinstance(t.instruction, str) and t.instruction.strip()
        ][:n])

    return tasks[:count]


def generate_tasks_from_tree(
    tree: FeatureTree,
    llm: LLMProvider,
    tasks_per_node: int = 2,
    max_total: int = 50,
    min_depth: int = 1,
    temperature: float = 1.0,
    temperatures: Optional[List[float]] = None,
) -> List[GeneratedTask]:
    """Generate tasks from across the feature taxonomy tree.

    Walks nodes at min_depth or deeper and allocates the task budget across
    them by feature frequency, reweighted by temperature (EpiCoder Eq 1). At
    temperature 1 this is plain proportional sampling; higher temperatures
    smooth the allocation so underrepresented features get more tasks instead
    of being crowded out by high-frequency-but-easy ones.

    Args:
        tree: The populated feature taxonomy.
        llm: LLM provider.
        tasks_per_node: Per-node floor — reserve this many tasks for each node
            before distributing the remainder by reweighted frequency (subject
            to the budget).
        max_total: Total task budget.
        min_depth: Minimum node depth to generate from.
        temperature: Reweighting temperature for the frequency distribution.
        temperatures: Optional list of temperatures to mix. The budget is split
            evenly across them and each slice allocated at its own temperature,
            widening the range of feature distributions (per the paper). Takes
            precedence over ``temperature`` when given.

    Returns:
        List of GeneratedTask objects, shuffled for diversity.
    """
    from .tree_utils import allocate_by_reweighted_frequency

    nodes = [
        node for _, node in tree.nodes.items()
        if node.depth >= min_depth and node.features
    ]
    if not nodes:
        return []

    freqs = [len(node.features) for node in nodes]
    temps = temperatures if temperatures else [temperature]

    # Split the budget across temperatures, then allocate each slice by its own
    # reweighted distribution and sum the per-node counts.
    per_temp = _split_budget(max_total, len(temps))
    totals = [0] * len(nodes)
    for t, budget in zip(temps, per_temp):
        counts = allocate_by_reweighted_frequency(
            freqs, budget, t, min_each=tasks_per_node
        )
        for i, c in enumerate(counts):
            totals[i] += c

    all_tasks: List[GeneratedTask] = []
    for node, n in zip(nodes, totals):
        if len(all_tasks) >= max_total:
            break
        if n <= 0:
            continue
        n = min(n, max_total - len(all_tasks))
        all_tasks.extend(generate_tasks_from_node(node, llm, count=n))

    random.shuffle(all_tasks)
    return all_tasks[:max_total]


def _split_budget(total: int, parts: int) -> List[int]:
    """Split an integer budget into ``parts`` near-equal chunks summing to total."""
    if parts <= 0:
        return []
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


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
