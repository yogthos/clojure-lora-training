"""Backfilled reasoning for mined git transitions."""

import json

import pytest

from src.codeflow.git_mining.rationale import add_reasoning, write_rationale
from src.llm.provider import LLMAuthError, LLMError

_DIFF = ("diff --git a/src/core.clj b/src/core.clj\n--- a/src/core.clj\n+++ b/src/core.clj\n"
         "@@ -1,2 +1,2 @@\n (ns core)\n-(defn parse [x] x)\n+(defn parse [x] (when x x))\n")
_REC = {"instruction": "handle nil in parse", "input": "### src/core.clj\n(ns core)\n(defn parse [x] x)",
        "output": _DIFF, "repo": "o__r"}
_GOOD = ("parse passes its argument straight through, so a nil from the caller flows on "
         "and blows up later. Guarding with when keeps nil out without changing the "
         "result for real values.")


class _LLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def call(self, system_prompt, user_prompt, **kw):
        self.prompts.append(user_prompt)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return json.dumps({"reasoning": r}) if isinstance(r, str) else json.dumps(r)


def test_rationale_sees_request_code_and_change():
    llm = _LLM(_GOOD)
    assert write_rationale(_REC, llm) == _GOOD
    assert "handle nil in parse" in llm.prompts[0]
    assert "(defn parse [x] x)" in llm.prompts[0]
    assert "+(defn parse [x] (when x x))" in llm.prompts[0]


@pytest.mark.parametrize("bad", [
    "The diff adds a when guard. " * 4,          # talks about the answer
    "As the patch shows, " + _GOOD,
    "Looking at the hunk, " + _GOOD,
    "Too short.",
    "word " * 600,                               # rambling
])
def test_unusable_rationale_is_retried_then_dropped(bad):
    assert write_rationale(_REC, _LLM(bad, bad)) is None
    assert write_rationale(_REC, _LLM(bad, _GOOD)) == _GOOD


def test_llm_errors():
    assert write_rationale(_REC, _LLM(LLMError("503"))) is None
    with pytest.raises(LLMAuthError):
        write_rationale(_REC, _LLM(LLMAuthError("401")))


def test_add_reasoning_prefixes_think_block():
    out = add_reasoning(_REC, _GOOD)
    assert out["output"] == f"<think>\n{_GOOD}\n</think>\n\n" + _DIFF
    assert out["instruction"] == _REC["instruction"] and out["repo"] == "o__r"
    assert _REC["output"] == _DIFF          # input record untouched


def test_long_inputs_are_capped_in_the_prompt():
    rec = dict(_REC, input="x" * 200_000)
    llm = _LLM(_GOOD)
    write_rationale(rec, llm)
    assert len(llm.prompts[0]) < 60_000
