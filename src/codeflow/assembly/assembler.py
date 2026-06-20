"""Dataset assembler: merge git-mined and synthetic data, deduplicate, balance.

Produces a merged JSONL dataset ready for LLaMA-Factory training.
"""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from ...shared import compute_dedup_key, load_jsonl

# ── Classification keyword sets ──────────────────────────────────────────

_BUG_KEYWORDS = [
    "bug", "fix", "error", "crash", "null", "broken", "incorrect",
    "wrong", "fails", "failure", "failing", "exception", "stack trace",
    "regression", "resolve", "patch", "hotfix",
]

_REFACTOR_KEYWORDS = [
    "refactor", "cleanup", "clean up", "restructure", "reorganize",
    "simplify", "extract", "rename", "move", "rename", "dry",
    "technical debt", "tech debt", "redundant", "consolidate",
    "split", "merge", "decompose", "composition", "extract function",
]

_FEATURE_KEYWORDS = [
    "add", "implement", "create", "new", "feature", "support",
    "introduce", "build", "enable", "endpoint", "route", "handler",
    "component", "api", "service",
]

_OPTIMIZE_KEYWORDS = [
    "optimize", "performance", "perf", "faster", "slow", "speed",
    "memory", "reduce", "lazy", "cache", "memoize", "throughput",
    "bottleneck", "profile", "transducer", "reduce",
]

_TEST_KEYWORDS = [
    "test", "spec", "assert", "coverage", "expect", "generative",
    "property-based", "check",
]

_DOCS_KEYWORDS = [
    "doc", "readme", "comment", "docstring", "documentation",
]


def classify_example(example: dict) -> str:
    """Classify a training example by change type.

    Examines both the instruction and output fields for keyword signals.
    Returns one of: bug-fix, refactor, add-feature, optimize, test, docs,
    or refactor (default).
    """
    text = " ".join([
        example.get("instruction", ""),
        example.get("output", ""),
        example.get("input", ""),
    ]).lower()

    scores = {
        "bug-fix": _count_keywords(text, _BUG_KEYWORDS),
        "refactor": _count_keywords(text, _REFACTOR_KEYWORDS),
        "add-feature": _count_keywords(text, _FEATURE_KEYWORDS),
        "optimize": _count_keywords(text, _OPTIMIZE_KEYWORDS),
        "test": _count_keywords(text, _TEST_KEYWORDS),
        "docs": _count_keywords(text, _DOCS_KEYWORDS),
    }

    # Use the highest-scoring category
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "refactor"


def _count_keywords(text: str, keywords: List[str]) -> int:
    """Count how many keywords appear in the text."""
    return sum(1 for kw in keywords if kw in text)


def deduplicate(records: List[dict]) -> List[dict]:
    """Remove duplicate records, keeping the first occurrence."""
    seen: set[str] = set()
    result = []
    for rec in records:
        key = compute_dedup_key(rec)
        if key not in seen:
            seen.add(key)
            result.append(rec)
    return result


def _example_chars(record: dict) -> int:
    """Total character length of the trainable fields of an example.

    Counts instruction + input + output. The system prompt is a constant
    preamble, so it is excluded — what matters here is the per-example payload
    that has to fit inside the training context window.
    """
    return (
        len(record.get("instruction", ""))
        + len(record.get("input", ""))
        + len(record.get("output", ""))
    )


def filter_by_length(
    records: List[dict],
    max_chars: Optional[int] = None,
) -> List[dict]:
    """Drop records whose instruction+input+output exceeds max_chars.

    Git-mined examples embed full before-state files as input, so a few
    multi-file commits balloon to hundreds of KB — far past any training
    context window, where they would just be truncated. This removes them
    before balancing so the cap operates on the trainable pool.

    If max_chars is None, all records are kept.
    """
    if max_chars is None:
        return records
    return [r for r in records if _example_chars(r) <= max_chars]


def balance_by_type(
    records: List[dict],
    max_per_type: Optional[int] = None,
    seed: int = 42,
) -> List[dict]:
    """Balance the dataset so no single change type dominates.

    If max_per_type is None, it is set to the count of the smallest type,
    effectively capping all types at the minority class size.

    Records beyond the cap are randomly sampled (not just truncated).
    """
    if not records:
        return []

    # Group by type
    by_type: Dict[str, List[dict]] = {}
    for rec in records:
        t = classify_example(rec)
        by_type.setdefault(t, []).append(rec)

    if max_per_type is None:
        max_per_type = min(len(v) for v in by_type.values())

    rng = random.Random(seed)
    balanced = []
    for t, group in sorted(by_type.items()):
        if len(group) <= max_per_type:
            balanced.extend(group)
        else:
            balanced.extend(rng.sample(group, max_per_type))

    # Shuffle so types are interleaved
    rng.shuffle(balanced)
    return balanced


def assemble_dataset(
    git_paths: List[Path],
    synth_paths: List[Path],
    output_path: Path,
    max_per_type: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> List[dict]:
    """Assemble a complete training dataset from git-mined and synthetic sources.

    Steps:
    1. Load all git-mined records
    2. Load all synthetic records
    3. Merge into one list
    4. Deduplicate (keep first occurrence — typically git-mined wins)
    5. Drop records too large for the training context window
    6. Balance by change type
    7. Write to output_path as JSONL
    8. Return the assembled records

    Args:
        git_paths: Directories or files containing git-mined JSONL data.
        synth_paths: Directories or files containing synthetic JSONL data.
        output_path: Where to write the merged JSONL.
        max_per_type: Max records per change type for balancing.
        max_chars: Drop records whose instruction+input+output exceeds this
            many characters (None disables the filter).

    Returns:
        List of assembled records.
    """
    records = []

    # Tag origin so the pipeline can report the git/synthetic mix. Git is loaded
    # first, so on a cross-source duplicate the git record wins dedup. The tag is
    # transient — format standardization drops it from the final training JSONL.
    for p in git_paths:
        for rec in load_jsonl(p):
            rec.setdefault("source", "git")
            records.append(rec)

    for p in synth_paths:
        for rec in load_jsonl(p):
            rec.setdefault("source", "synthetic")
            records.append(rec)

    records = deduplicate(records)
    records = filter_by_length(records, max_chars=max_chars)

    # Balance only the abundant git-mined pool by change type. The synthetic
    # workflow set is small, curated, and execution-verified to a target count,
    # and trains a distinct objective (the full agent loop) — so it is kept in
    # full rather than diluted by random sampling inside saturated buckets.
    git_records = [r for r in records if r.get("source") != "synthetic"]
    synth_records = [r for r in records if r.get("source") == "synthetic"]
    git_records = balance_by_type(git_records, max_per_type=max_per_type)
    records = git_records + synth_records
    random.Random(42).shuffle(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records
