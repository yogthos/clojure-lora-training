"""LLaMA-Factory JSONL formatter for Code Flow training data.

Standardizes records to the format expected by LLaMA-Factory:
  {instruction, input, output, system, history}

The system prompt describes the nREPL-driven Clojure development workflow.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from .assembler import load_jsonl

DEFAULT_SYSTEM_PROMPT = (
    "You are a Clojure coding agent using nREPL-driven development. "
    "The project has a running nREPL server available. "
    "Develop interactively: evaluate forms in the REPL to explore and test, "
    "inspect results, refine your approach, then apply final changes to files. "
    "Output format: start with ;; eval: blocks showing REPL evaluations and "
    "their results, followed by ;; apply: with unified diff patches."
)

# Standard field order for LLaMA-Factory
_FIELD_ORDER = ("instruction", "input", "output", "system", "history")

# Fields recognized as part of the LLaMA-Factory format
_VALID_FIELDS = set(_FIELD_ORDER)


def format_record(record: dict) -> dict:
    """Format a single record for LLaMA-Factory compatibility.

    Ensures:
    - All required fields are present (instruction, input, output, system, history)
    - All field values are strings (except history, which is a list)
    - System prompt defaults to the nREPL-development prompt
    - History defaults to empty list
    - Unrecognized fields are stripped
    - Fields are in standard order
    """
    result = {}

    # instruction (required)
    result["instruction"] = _to_string(record.get("instruction", ""))

    # input (optional, defaults to empty string)
    result["input"] = _to_string(record.get("input", ""))

    # output (required)
    result["output"] = _to_string(record.get("output", ""))

    # system (optional, defaults to standard prompt)
    system = record.get("system", None)
    if system is None or (isinstance(system, str) and not system.strip()):
        result["system"] = DEFAULT_SYSTEM_PROMPT
    else:
        result["system"] = _to_string(system)

    # history (optional, defaults to empty list for single-turn)
    history = record.get("history", [])
    if isinstance(history, list):
        result["history"] = history
    else:
        result["history"] = []

    # Reorder to standard field order
    ordered = {}
    for field in _FIELD_ORDER:
        ordered[field] = result[field]

    return ordered


def _to_string(value) -> str:
    """Coerce a value to string safely."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def format_jsonl(records: List[dict]) -> List[dict]:
    """Format a list of records for LLaMA-Factory compatibility."""
    return [format_record(r) for r in records]


def format_jsonl_file(input_path: Path, output_path: Path) -> int:
    """Read a JSONL file, format all records, and write to output.

    Returns the number of records written.
    """
    records = load_jsonl(input_path)
    formatted = format_jsonl(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in formatted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(formatted)
