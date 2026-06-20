"""Shared constants and utilities for the Clojure LoRA trainer.

This module is the single source of truth for:
- The system prompt (used by both git_mining and synthetic)
- JSONL I/O (used by assembly, synthetic, and scripts)
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Union

# ── Shared constants ───────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a Clojure coding agent using nREPL-driven development. "
    "The project has a running nREPL server available. "
    "Develop interactively: evaluate forms in the REPL to explore and test, "
    "inspect results, refine your approach, then apply final changes to files. "
    "Output format: start with ;; eval: blocks showing REPL evaluations and "
    "their results, followed by ;; apply: with unified diff patches."
)

# System prompt for the code-flow transition task: git-mined examples teach how
# a real change is applied to existing code. There is no REPL trace in git
# history, so these examples target patch generation directly.
_TRANSITION_SYSTEM_PROMPT = (
    "You are a Clojure coding agent. You are given the current state of one or "
    "more Clojure source files and a description of a change to make. Apply the "
    "change by producing a unified diff patch in git format. Output only the "
    "unified diff."
)

# ── JSONL I/O ──────────────────────────────────────────────────────────────

PathOrStr = Union[Path, str]


def load_jsonl(path: PathOrStr) -> List[dict]:
    """Load JSONL records from a file or directory of .jsonl files."""
    p = Path(path)
    if p.is_dir():
        records: List[dict] = []
        for fp in sorted(p.glob("*.jsonl")):
            records.extend(_read_file(fp))
        return records
    elif p.is_file():
        return _read_file(p)
    return []


def write_jsonl(records: List[dict], path: PathOrStr) -> None:
    """Write records to a JSONL file, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_records(path: PathOrStr) -> int:
    """Count non-blank lines in a JSONL file."""
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def compute_dedup_key(example: dict) -> str:
    """SHA-256 hash of instruction + output for deduplication."""
    instruction = example.get("instruction", "")
    output = example.get("output", "")
    combined = f"{instruction.strip()}\n{output.strip()}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _read_file(path: Path) -> List[dict]:
    """Load JSONL from a single file, skipping blank/malformed lines."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
