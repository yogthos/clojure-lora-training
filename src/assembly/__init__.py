"""Dataset assembly package.

Stages:
  assembler — merge, deduplicate, balance git-mined and synthetic data
  formatter — standardize records for LLaMA-Factory JSONL format
  validator — Clojure syntax, diff coherence, and relevance scoring
"""

from .assembler import assemble_dataset
from .formatter import format_jsonl, format_jsonl_file
from .validator import validate_example, validate_jsonl_file

__all__ = [
    "assemble_dataset",
    "format_jsonl",
    "format_jsonl_file",
    "validate_example",
    "validate_jsonl_file",
]
