"""Resume behaviour of the clj-lora-add-reasoning step."""

import importlib
import json
import sys

import pytest

from src.llm.provider import LLMAuthError

_GOOD = ("parse passes nil straight through to the caller, so a when guard keeps it out "
         "while every real value comes back unchanged, which is all the request asks for.")
_DIFF = "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n-x\n+y\n"


class _LLM:
    def __init__(self, fail_on=()):
        self.calls = 0
        self.fail_on = fail_on

    def call(self, system_prompt, user_prompt, **kw):
        self.calls += 1
        if any(f in user_prompt for f in self.fail_on):
            return json.dumps({"reasoning": "too short"})
        return json.dumps({"reasoning": _GOOD})


def _run(monkeypatch, llm, argv):
    mod = importlib.import_module("src.codeflow.cli.add_reasoning")
    monkeypatch.setattr(mod, "setup_llm", lambda: llm)
    monkeypatch.setattr(sys, "argv", ["add-reasoning"] + argv)
    return mod.main()


@pytest.fixture
def mined(tmp_path):
    p = tmp_path / "mined.jsonl"
    recs = [{"instruction": f"change {i}", "input": "", "output": _DIFF, "repo": "r"} for i in range(4)]
    recs.append({"instruction": "already", "input": "",
                 "output": f"<think>\n{_GOOD}\n</think>\n\n{_DIFF}", "repo": "r"})
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return p


def test_adds_reasoning_and_resumes(tmp_path, mined, monkeypatch):
    out = tmp_path / "out.jsonl"
    argv = ["--input", str(mined), "--output", str(out), "--workers", "2"]
    first = _LLM(fail_on=("change 3",))
    assert _run(monkeypatch, first, argv) == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert sorted(r["instruction"] for r in rows) == ["already", "change 0", "change 1", "change 2"]
    assert all(r["output"].startswith("<think>\n") for r in rows)
    assert first.calls == 5                    # 3 good + 2 tries on the failure

    again = _LLM()
    assert _run(monkeypatch, again, argv) == 0
    assert again.calls == 0                    # kept and failed records both skipped
    assert len(out.read_text().splitlines()) == 4


def test_torn_output_is_trimmed_before_resuming(tmp_path, mined, monkeypatch):
    out = tmp_path / "out.jsonl"
    argv = ["--input", str(mined), "--output", str(out), "--workers", "1"]
    _run(monkeypatch, _LLM(), argv)
    lines = out.read_text().splitlines()
    out.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:10])
    att = tmp_path / "out.jsonl.attempted"
    att.write_text("".join(att.read_text().splitlines(keepends=True)[:-1]))
    _run(monkeypatch, _LLM(), argv)
    assert len([json.loads(l) for l in out.read_text().splitlines()]) == 5


def test_auth_error_stops(tmp_path, mined, monkeypatch):
    class _NoAuth:
        def call(self, *a, **k):
            raise LLMAuthError("401")
    with pytest.raises(LLMAuthError):
        _run(monkeypatch, _NoAuth(), ["--input", str(mined), "--output", str(tmp_path / "o.jsonl")])
