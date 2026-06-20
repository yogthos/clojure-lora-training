"""Prompt templates for Clojure code synthesis.

Each prompt is loaded from a standalone .txt file so it can be read and
edited without parsing Python source. Module-level constants expose them
for backwards-compatible `from .prompts import X` imports.
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def _load(name: str) -> str:
    with open(_PROMPTS_DIR / f"{name}.txt") as f:
        return f.read().rstrip()


ANALYSIS_SYSTEM = _load("analysis_system")
CODE_SYSTEM = _load("code_system")
BREADTH_SYSTEM = _load("breadth_system")
DEPTH_SYSTEM = _load("depth_system")
DETAIL_SYSTEM = _load("detail_system")
EXTRACT_SYSTEM = _load("extract_system")
QUESTION_SYSTEM = _load("question_system")
CONSTRUCT_TREE_SYSTEM = _load("construct_tree_system")
BACKTRANSLATE_SYSTEM = _load("backtranslate_system")
PLAN_SYSTEM = _load("plan_system")
WORKFLOW_SYSTEM = _load("workflow_system")

__all__ = [
    "ANALYSIS_SYSTEM",
    "BACKTRANSLATE_SYSTEM",
    "BREADTH_SYSTEM",
    "CODE_SYSTEM",
    "CONSTRUCT_TREE_SYSTEM",
    "DEPTH_SYSTEM",
    "DETAIL_SYSTEM",
    "EXTRACT_SYSTEM",
    "PLAN_SYSTEM",
    "QUESTION_SYSTEM",
    "WORKFLOW_SYSTEM",
]