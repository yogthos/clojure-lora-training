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

# System prompt for the full agent workflow: given a vague user request, the
# model must work out the goal, lay out files, sketch a plan, then build the
# code iteratively in the REPL — write, run, observe, fix, repeat — and emit a
# diff. The planning lives in the OUTPUT so the model learns to produce it.
_WORKFLOW_SYSTEM_PROMPT = (
    "You are a Clojure coding agent that develops via nREPL-driven development. "
    "Given a user request, work out the goal, decide which files you need, and "
    "sketch a short plan of the functions to build in dependency order. Then "
    "implement them one at a time in the REPL: write a function, evaluate it, "
    "run it on sample data, inspect the result, and if it is wrong or throws, "
    "fix it and re-run until it works — then move to the next function. "
    "Structure your response as: ';; Goal:' (the end state), ';; Files:' (each "
    "file and its purpose), ';; Plan (build order):' (the functions in order), "
    "';; nREPL session:' (the iterative eval/result steps, including any "
    "failures and how you recover from them), and ';; apply:' with a unified "
    "diff of the final working code."
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

# Paraphrase pools. Using one identical system string across every example
# invites attention collapse — the model keys on the fixed prompt rather than
# the task. We assign an equivalent paraphrase per example. Index 0 is the
# canonical phrasing above. The WORKFLOW variants keep the literal section
# markers (';; Goal:' etc.) verbatim, since the output format depends on them;
# only the surrounding prose varies.
TRANSITION_SYSTEM_PROMPTS = [
    _TRANSITION_SYSTEM_PROMPT,
    "You are a Clojure coding agent. Given one or more Clojure source files and "
    "a description of the change to make, apply it as a unified diff patch in "
    "git format. Respond with the diff only.",
    "You are a Clojure coding assistant. You receive the current contents of "
    "some Clojure files together with a requested change. Produce a git-format "
    "unified diff that makes the change, and output nothing but the diff.",
    "Act as a Clojure coding agent. From the given Clojure source files and a "
    "change description, generate the unified diff (git format) that applies the "
    "change. Return only the diff.",
    "You are a Clojure patch-generation agent. Read the provided Clojure source "
    "and the requested change, then emit a git-style unified diff that performs "
    "it. Your entire response must be the diff.",
]

_WORKFLOW_STRUCTURE = (
    "Structure your response as: ';; Goal:' (the end state), ';; Files:' (each "
    "file and its purpose), ';; Plan (build order):' (the functions in order), "
    "';; nREPL session:' (the iterative eval/result steps, including any "
    "failures and how you recover from them), and ';; apply:' with a unified "
    "diff of the final working code."
)

WORKFLOW_SYSTEM_PROMPTS = [
    _WORKFLOW_SYSTEM_PROMPT,
    "You are a Clojure coding agent working through an nREPL. Given a user "
    "request, figure out the goal, decide which files are involved, and outline "
    "the functions to build in dependency order. Build them one at a time at the "
    "REPL — write a function, evaluate it, exercise it on sample data, check the "
    "result, and when it is wrong or throws, fix it and re-run until it works "
    "before moving on. " + _WORKFLOW_STRUCTURE,
    "You are a Clojure agent that develops interactively at the REPL. Starting "
    "from a user request, determine the end goal, lay out the files you need, and "
    "plan the functions in the order they must be built. Implement each one "
    "incrementally: define it, run it on examples, inspect what comes back, and "
    "iterate through any errors until it behaves correctly, then continue to the "
    "next. " + _WORKFLOW_STRUCTURE,
    "Act as a Clojure coding agent using nREPL-driven development. From a user "
    "request, work out what success looks like, identify the files to touch, and "
    "sketch the build order of the functions. Develop them one by one in the "
    "REPL — evaluate, test on real inputs, read the output, and repair and re-run "
    "on failure until each works before advancing. " + _WORKFLOW_STRUCTURE,
]

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
