"""Resume behaviour of src/codeflow/cli/generate_workflow_data.py."""

import importlib.util
import json
import pathlib
import sys

import pytest

_DIFF = ("diff --git a/src/x.clj b/src/x.clj\n--- /dev/null\n+++ b/src/x.clj\n"
         "@@ -0,0 +1,2 @@\n+(ns x)\n+(defn f [] 3)\n")


class _CountingLLM:
    """Mines prompts, then plays a 3-eval trace per prompt; counts trace starts."""

    def __init__(self):
        self.mining_calls = 0
        self.started = set()

    @property
    def workflow_calls(self):
        return len(self.started)

    def call(self, system_prompt, user_prompt, require_json=False, **k):
        if user_prompt.startswith("Commit message:"):
            self.mining_calls += 1
            msg = user_prompt.split("\n", 1)[1]
            return json.dumps({"user_prompt": f"please {msg}", "project_context": "a lib"})
        n = user_prompt.count("--- step ")
        if n == 0:
            # a retry is the same prompt plus why the last reply was rejected
            self.started.add(user_prompt.split("\n\nYour last reply was rejected")[0])
        if "drop me" in user_prompt:
            return "no step here"   # malformed every time: dropped
        if n < 3:
            return json.dumps({"thinking": "t", "action": "eval", "code": f"(+ {n} 1)"})
        return json.dumps({"thinking": "t", "action": "apply", "diff": _DIFF})


class _Session:
    alive = True

    def __init__(self, **kw):
        pass

    def eval(self, code):
        from src.codeflow.synthetic.jolt_eval import EvalResult
        return EvalResult(form=code, value="1")

    def close(self):
        pass


def _script():
    return importlib.import_module("src.codeflow.cli.generate_workflow_data")


def _run(mod, monkeypatch, llm, argv):
    monkeypatch.setattr(mod, "setup_llm", lambda: llm)
    monkeypatch.setattr(mod, "jolt_available", lambda path: True)
    monkeypatch.setattr(mod, "session_factory", lambda args: _Session)
    monkeypatch.setattr(sys, "argv", ["gen"] + argv)
    assert mod.main() == 0


@pytest.fixture
def git_data(tmp_path):
    p = tmp_path / "git.jsonl"
    msgs = ["add nested schema support\n\nwith a body", "fix the broken retry loop",
            "drop me from the output please", "implement streaming csv parser"]
    p.write_text("".join(json.dumps({"instruction": m}) + "\n" for m in msgs))
    return p


def test_rerun_resumes_without_redoing_work(tmp_path, git_data, monkeypatch):
    mod = _script()
    out = tmp_path / "wf.jsonl"
    argv = ["--git-data", str(git_data), "-o", str(out), "--target", "4",
            "--workers", "1"]
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
            "--workers", "1"]
    _run(mod, monkeypatch, _CountingLLM(), argv)
    fresh = _CountingLLM()
    _run(mod, monkeypatch, fresh, argv + ["--fresh"])
    assert fresh.mining_calls == 4
    assert len(out.read_text().splitlines()) == 3


def test_larger_target_tops_up_prompts(tmp_path, git_data, monkeypatch):
    mod = _script()
    out = tmp_path / "wf.jsonl"
    base = ["--git-data", str(git_data), "-o", str(out),
            "--workers", "1"]
    _run(mod, monkeypatch, _CountingLLM(), base + ["--target", "2"])
    more = _CountingLLM()
    _run(mod, monkeypatch, more, base + ["--target", "4"])
    assert more.mining_calls == 2
    assert more.workflow_calls == 2


def test_resume_after_torn_write(tmp_path, git_data, monkeypatch):
    mod = _script()
    out = tmp_path / "wf.jsonl"
    argv = ["--git-data", str(git_data), "-o", str(out), "--target", "4",
            "--workers", "1"]
    _run(mod, monkeypatch, _CountingLLM(), argv)
    lines = out.read_text().splitlines()
    # Simulate a crash mid-write: drop the last record's attempt marker and
    # leave half its line behind.
    out.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:20])
    att = tmp_path / "wf.jsonl.attempted"
    att.write_text("".join(att.read_text().splitlines(keepends=True)[:-1]))
    again = _CountingLLM()
    _run(mod, monkeypatch, again, argv)
    assert again.workflow_calls == 1
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == 3


def test_torn_prompt_cache_then_top_up(tmp_path, git_data, monkeypatch):
    mod = _script()
    out = tmp_path / "wf.jsonl"
    base = ["--git-data", str(git_data), "-o", str(out), "--workers", "1"]
    _run(mod, monkeypatch, _CountingLLM(), base + ["--target", "2"])
    cache = tmp_path / "wf.jsonl.prompts"
    lines = cache.read_text().splitlines()
    cache.write_text(lines[0] + "\n" + lines[1][:15])       # crash mid-write
    _run(mod, monkeypatch, _CountingLLM(), base + ["--target", "4"])
    assert len([json.loads(l) for l in cache.read_text().splitlines()]) == 4


def test_session_factory_passes_files(monkeypatch):
    # The acceptance check opens a session with the diff's files on its path.
    import argparse
    from src.codeflow.synthetic import jolt_eval
    mod = _script()
    seen = []
    monkeypatch.setattr(jolt_eval, "JoltSession", lambda **kw: seen.append(kw) or _Session())
    factory = mod.session_factory(argparse.Namespace(jolt="/x/jolt"))
    factory()
    factory(files={"src/a.clj": "(ns a)"})
    assert seen == [{"jolt_path": "/x/jolt"},
                    {"jolt_path": "/x/jolt", "files": {"src/a.clj": "(ns a)"}}]
