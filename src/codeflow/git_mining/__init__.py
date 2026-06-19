"""Git repository mining for Clojure code evolution data.

Extracts commit transitions, formats multi-file before/after states,
and produces JSONL records for LLaMA-Factory training.
"""

from .miner import (
    MinedExample,
    get_commit_diff,
    get_commit_list,
    get_file_content,
    mine_repository,
)

__all__ = [
    "MinedExample",
    "get_commit_diff",
    "get_commit_list",
    "get_file_content",
    "mine_repository",
]