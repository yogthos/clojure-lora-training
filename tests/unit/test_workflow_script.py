"""Resume behaviour of scripts/generate_workflow_data.py."""

import importlib.util
import json
import pathlib
import sys

import pytest

_DIFF = (
    "diff --git a/src/x.clj b/src/x.clj\n--- a/src/x.clj\n+++ b/src/x.clj\n"
    "@@ -1,2 +1,2 @@\n (ns x)\n-(defn f [] 1)\n+(defn f [] 3)\n"
)
_TRACE = ";; Goal: g\n;; nREPL session:\n;; eval: (+ 1 2)\n;; result: ?\n;; apply:\n" + _DIFF


class _CountingLLM:
    def __init__(self):
        self.mining_calls = 0
        self.workflow_calls = 0

    def call(self, system_prompt, user_prompt, require_json=False, **k):
        if user_prompt.startswith("Commit message:"):
            self.mining_calls += 1
            msg = user_prompt.split("\n", 1)[1]
            return json.dumps({"user_prompt": f"please {msg}", "project_context": "a lib"})
        if require_json:
            return json.dumps({"goal": "g", "steps": [{"name": "s"}]})
        self.workflow_calls += 1
        if "drop me" in user_prompt:
            return "no diff here"   # dropped by the trace gates
        return _TRACE


def _script():
    spec = importlib.util.spec_from_file_location(
        "gen_wf", pathlib.Path("scripts/generate_workflow_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(mod, monkeypatch, llm, argv):
    monkeypatch.setattr(mod, "setup_llm", lambda: llm)
    monkeypatch.setattr(sys, "argv", ["gen"] + argv)
    assert mod.main() == 0


@pytest.fixture
def git_data(tmp_path):
    p = tmp_path / "git.jsonl"
    msgs = ["add nested schema support", "fix the broken retry loop",
            "drop me from the output please", "implement streaming csv parser"]
    p.write_text("".join(json.dumps({"instruction": m}) + "\n" for m in msgs))
    return p


def test_rerun_resumes_without_redoing_work(tmp_path, git_data, monkeypatch):
    mod = _script()
    out = tmp_path / "wf.jsonl"
    argv = ["--git-data", str(git_data), "-o", str(out), "--target", "4",
            "--no-verify-execution", "--workers", "1"]
    first = _CountingLLM()
    _run(mod, monkeypatch, first, argv)
    kept = out.read_text().splitlines()
    assert len(kept) == 3 and first.mining_calls == 4 and first.workflow_calls == 4

    again = _CountingLLM()
    _run(mod, monkeypatch, again, argv)
    assert out.read_text().splitlines() == kept       # nothing truncated or duplicated
    assert again.mining_calls == 0                     # prompts came from the cache
    assert again.workflow_calls == 0                   # kept AND dropped prompts skipped


def test_fresh_starts_over(tmp_path, git_data, monkeypatch):
    mod = _script()
    out = tmp_path / "wf.jsonl"
    argv = ["--git-data", str(git_data), "-o", str(out), "--target", "4",
            "--no-verify-execution", "--workers", "1"]
    _run(mod, monkeypatch, _CountingLLM(), argv)
    fresh = _CountingLLM()
    _run(mod, monkeypatch, fresh, argv + ["--fresh"])
    assert fresh.mining_calls == 4
    assert len(out.read_text().splitlines()) == 3


def test_larger_target_tops_up_prompts(tmp_path, git_data, monkeypatch):
    mod = _script()
    out = tmp_path / "wf.jsonl"
    base = ["--git-data", str(git_data), "-o", str(out),
            "--no-verify-execution", "--workers", "1"]
    _run(mod, monkeypatch, _CountingLLM(), base + ["--target", "2"])
    more = _CountingLLM()
    _run(mod, monkeypatch, more, base + ["--target", "4"])
    assert more.mining_calls == 2
    assert more.workflow_calls == 2
