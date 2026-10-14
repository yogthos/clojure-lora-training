"""Backfill reasoning for mined git transitions.

A transition is a request, the before-state code and the real diff. Thinking
mode trains every assistant turn with a <think> block, so each transition gets
the reasoning a developer would have had before making the change, written by
an LLM that sees the change. It must read as forethought: anything that refers
to the change as given (a diff, a patch, "the change above") is rejected.
"""

import json
import re
from typing import List, Optional

from ...llm.provider import LLMAuthError, LLMError, LLMProvider
from ..synthetic.live_trace import think
from ..synthetic.prompts import RATIONALE_SYSTEM

MIN_WORDS, MAX_WORDS = 20, 400
_MAX_INPUT_CHARS = 30_000
_MAX_DIFF_CHARS = 20_000

# Hindsight phrasing: the reasoning talks about the answer it was shown.
# Earlier commits are fair game (inputs carry the files' recent history); the
# change being made is not ("this commit", "the commit below").
_HINDSIGHT = re.compile(
    r"\b(diffs?|patch(es)?|hunks?|(this|the) commit\b(?! (history|log))|commit (above|below)|"
    r"the change (above|below|shown)|as shown|line \d+|lines \d+)\b", re.IGNORECASE)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


def _usable(text: str) -> bool:
    words = len(text.split())
    return MIN_WORDS <= words <= MAX_WORDS and not _HINDSIGHT.search(text)


def write_rationale(record: dict, llm: LLMProvider) -> Optional[str]:
    """Reasoning for one transition, or None if two attempts give nothing usable."""
    prompt = (f"Request:\n{record.get('instruction', '')}\n\n"
              f"Current code:\n{_clip(record.get('input', ''), _MAX_INPUT_CHARS)}\n\n"
              f"The change that was made:\n{_clip(record.get('output', ''), _MAX_DIFF_CHARS)}")
    for _ in range(2):
        try:
            raw = llm.call(system_prompt=RATIONALE_SYSTEM, user_prompt=prompt, temperature=0.4,
                           max_tokens=1024, require_json=True, reject_truncated=True)
        except LLMAuthError:
            raise
        except LLMError:
            return None
        try:
            data = json.loads(raw, strict=False) if isinstance(raw, str) else raw
        except ValueError:
            continue
        text = str(data.get("reasoning", "")).strip() if isinstance(data, dict) else ""
        if _usable(text):
            return text
    return None


def add_reasoning(record: dict, rationale: str) -> dict:
    """A copy of ``record`` whose output opens with the reasoning."""
    return dict(record, output=think(rationale) + record["output"])


def _gpt_turns(record: dict) -> List[int]:
    return [i for i, t in enumerate(record.get("conversations") or []) if t["from"] == "gpt"]


def needs_reasoning(record: dict) -> bool:
    """A transition, or any turn of a history chain, without reasoning yet."""
    from ..assembly.validator import split_think

    conv = record.get("conversations")
    if conv:
        return any(split_think(conv[i]["value"]) is None for i in _gpt_turns(record))
    return split_think(record.get("output", "")) is None


def reason_record(record: dict, llm: LLMProvider) -> Optional[dict]:
    """``record`` with reasoning on every assistant turn, or None if any turn
    can't get usable reasoning. A history chain's turns are each reasoned
    from their own request and code (the human turn is instruction, newline,
    input)."""
    from ..assembly.validator import split_think

    conv = record.get("conversations")
    if not conv:
        text = write_rationale(record, llm)
        return add_reasoning(record, text) if text else None
    conv = [dict(t) for t in conv]
    for i in _gpt_turns(record):
        if split_think(conv[i]["value"]) is not None:
            continue
        instruction, _, inp = conv[i - 1]["value"].partition("\n")
        text = write_rationale({"instruction": instruction, "input": inp,
                                "output": conv[i]["value"]}, llm)
        if not text:
            return None
        conv[i]["value"] = think(text) + conv[i]["value"]
    return dict(record, conversations=conv, output=conv[_gpt_turns(record)[0]]["value"])
