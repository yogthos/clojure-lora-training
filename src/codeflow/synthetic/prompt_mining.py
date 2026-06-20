"""Backtranslate commit messages into user-style agent prompts.

The git-mined data carries real developer intent in its commit messages. To
train the model to go from a vague user request to the full workflow, we turn
those messages into the high-level prompts a user would have given an agent
*before* the work — PlanSearch-style backtranslation applied to prompts.
"""

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from ...llm.provider import LLMProvider
from .prompts import BACKTRANSLATE_SYSTEM as _BACKTRANSLATE_SYSTEM

# Commit-message noise that doesn't represent a user-facing development goal.
_TRIVIAL_RE = re.compile(
    r"^\s*(merge\b|revert\b|bump\b|release\b|v?\d+\.\d+|version\b|"
    r"changelog|readme|typo|cljfmt|lint\b|format\b|whitespace|"
    r"wip\b|fixup\b|squash\b|\.gitignore)",
    re.IGNORECASE,
)

_MIN_WORDS = 3


@dataclass
class MinedPrompt:
    """A user-style request backtranslated from a commit message."""
    user_prompt: str
    project_context: str
    source_instruction: str


def is_substantive(instruction: str) -> bool:
    """True if a commit message represents real, agent-promptable intent."""
    msg = (instruction or "").strip()
    if not msg:
        return False
    first = msg.splitlines()[0].strip()
    if len(first.split()) < _MIN_WORDS:
        return False
    if len(first) < 12:
        return False
    if _TRIVIAL_RE.match(first):
        return False
    return True


def backtranslate_prompt(
    instruction: str,
    llm: LLMProvider,
) -> Optional[MinedPrompt]:
    """Turn one commit message into a user request + project context.

    Returns None for trivial messages or unparseable LLM output.
    """
    if not is_substantive(instruction):
        return None

    try:
        raw = llm.call(
            system_prompt=_BACKTRANSLATE_SYSTEM,
            user_prompt=f"Commit message:\n{instruction}",
            temperature=0.4,
            max_tokens=512,
            require_json=True,
        )
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, list):
            data = data[0] if data else {}
    except (ValueError, TypeError, KeyError, IndexError):
        return None

    if not isinstance(data, dict):
        return None
    user_prompt = str(data.get("user_prompt", "")).strip()
    project_context = str(data.get("project_context", "")).strip()
    if not user_prompt:
        return None

    return MinedPrompt(
        user_prompt=user_prompt,
        project_context=project_context,
        source_instruction=instruction,
    )


def mine_prompts(
    records: List[dict],
    llm: LLMProvider,
    max_prompts: int = 500,
) -> List[MinedPrompt]:
    """Mine up to ``max_prompts`` user-style prompts from mined git records.

    Deduplicates by source instruction so the same commit isn't backtranslated
    twice.
    """
    seen = set()
    prompts: List[MinedPrompt] = []
    for rec in records:
        if len(prompts) >= max_prompts:
            break
        instruction = (rec.get("instruction") or "").strip()
        if not instruction or instruction in seen:
            continue
        seen.add(instruction)
        mined = backtranslate_prompt(instruction, llm)
        if mined is not None:
            prompts.append(mined)
    return prompts
