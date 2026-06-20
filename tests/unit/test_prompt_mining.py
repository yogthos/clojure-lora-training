"""Tests for backtranslating commit messages into user-style prompts."""

import json
import pytest

from src.codeflow.synthetic.prompt_mining import (
    MinedPrompt,
    is_substantive,
    backtranslate_prompt,
    mine_prompts,
)


class _FakeLLM:
    def __init__(self, payload):
        self._payload = json.dumps(payload)

    def call(self, system_prompt, user_prompt, temperature=None,
             max_tokens=None, require_json=False):
        return self._payload


class TestIsSubstantive:
    def test_keeps_real_intent(self):
        assert is_substantive("fix nil handling in the arg parser")
        assert is_substantive("add support for nested schemas")

    def test_drops_trivial(self):
        assert not is_substantive("bump version to 1.2.3")
        assert not is_substantive("Merge pull request #42 from foo/bar")
        assert not is_substantive("typo")
        assert not is_substantive("")
        assert not is_substantive("wip")

    def test_drops_release_and_formatting_noise(self):
        assert not is_substantive("Release 2.0.0")
        assert not is_substantive("cljfmt: reformat")
        assert not is_substantive("update changelog")


class TestBacktranslate:
    def test_produces_user_prompt_and_context(self):
        llm = _FakeLLM({
            "user_prompt": "the CSV parser chokes on quoted fields, fix it",
            "project_context": "a Clojure data-processing library",
        })
        p = backtranslate_prompt("fix quoted-field handling in parse-csv", llm)
        assert isinstance(p, MinedPrompt)
        assert "quoted" in p.user_prompt
        assert "Clojure" in p.project_context
        assert p.source_instruction == "fix quoted-field handling in parse-csv"

    def test_trivial_message_returns_none(self):
        llm = _FakeLLM({"user_prompt": "x", "project_context": "y"})
        assert backtranslate_prompt("v1.2.3", llm) is None

    def test_malformed_llm_output_returns_none(self):
        class Bad:
            def call(self, *a, **k):
                return "not json at all"
        assert backtranslate_prompt("add a real feature here", Bad()) is None


class TestMinePrompts:
    def test_mines_and_dedups(self):
        llm = _FakeLLM({"user_prompt": "do the thing", "project_context": "a lib"})
        records = [
            {"instruction": "add nested schema support"},
            {"instruction": "bump version"},          # filtered
            {"instruction": "add nested schema support"},  # dup source
            {"instruction": "fix the broken retry loop"},
        ]
        prompts = mine_prompts(records, llm, max_prompts=10)
        # two substantive unique sources kept
        assert len(prompts) == 2
        assert all(isinstance(p, MinedPrompt) for p in prompts)

    def test_respects_max(self):
        llm = _FakeLLM({"user_prompt": "p", "project_context": "c"})
        records = [{"instruction": f"implement feature number {i}"} for i in range(20)]
        assert len(mine_prompts(records, llm, max_prompts=5)) == 5
