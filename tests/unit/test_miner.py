"""Tests for git repository miner."""

import json
import pytest
import tempfile
import subprocess
from pathlib import Path
from src.codeflow.git_mining.miner import (
    mine_repository,
    MinedExample,
    get_commit_list,
    get_commit_diff,
)


class TestGitOperations:
    """Tests that require a real git repo (test against our own repo)."""

    def test_get_commit_list_returns_commits(self):
        commits = get_commit_list(".")
        assert len(commits) > 0
        first = commits[0]
        assert first.hash
        assert first.message
        assert first.timestamp
        assert len(first.files) > 0

    def test_get_commit_diff_returns_diff(self):
        commits = get_commit_list(".", max_count=5)
        # Get the first non-merge commit
        for c in commits:
            if not c.is_merge:
                diff = get_commit_diff(".", c.hash)
                assert len(diff) > 0
                break

    def test_get_commit_list_max_count(self):
        commits = get_commit_list(".", max_count=3)
        assert len(commits) <= 3


class TestMineRepository:
    """Integration test mining our own repo history."""

    def test_mine_local_repo(self):
        examples = mine_repository(
            ".",
            repo_name="clojure-lora-trainer",
            max_commits=20,
        )
        # Our repo has Python files, not Clojure, so examples may be 0
        # But the function should run without error
        assert isinstance(examples, list)


class TestNonUtf8Tolerance:
    """The miner must not crash when git output contains non-UTF-8 bytes.

    Real repos (e.g. babashka) carry files with latin-1/binary bytes; a strict
    UTF-8 decode of `git show` raised UnicodeDecodeError and dropped the whole
    repo. git output should be decoded with errors='replace' instead.
    """

    def _make_repo_with_bad_bytes(self, root: Path) -> None:
        def git(*args):
            subprocess.run(["git", "-C", str(root), *args], check=True,
                           capture_output=True)

        git("init")
        git("config", "user.email", "t@t")
        git("config", "user.name", "t")
        f = root / "core.clj"
        # First commit: valid Clojure.
        f.write_bytes(b"(ns app)\n(defn greet [] :hi)\n")
        git("add", "-A")
        git("commit", "-m", "add greet")
        # Second commit: introduce a non-UTF-8 byte (0x9e) into the source.
        f.write_bytes(b"(ns app)\n(defn greet [] :hi)\n;; \x9e marker\n")
        git("add", "-A")
        git("commit", "-m", "fix greet encoding edge case")

    def test_get_commit_diff_tolerates_bad_bytes(self):
        from src.codeflow.git_mining.miner import get_commit_diff, get_commit_list

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._make_repo_with_bad_bytes(root)
            commits = get_commit_list(str(root))
            # Should not raise UnicodeDecodeError.
            diff = get_commit_diff(str(root), commits[0].hash)
            assert isinstance(diff, str)

    def test_get_file_content_tolerates_bad_bytes(self):
        from src.codeflow.git_mining.miner import get_file_content, get_commit_list

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._make_repo_with_bad_bytes(root)
            head = get_commit_list(str(root))[0].hash
            content = get_file_content(str(root), head, "core.clj")
            assert isinstance(content, str)
            assert "marker" in content

    def test_mine_repository_tolerates_bad_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._make_repo_with_bad_bytes(root)
            examples = mine_repository(str(root), max_commits=20)
            assert isinstance(examples, list)


class TestMinedExample:
    """Unit tests for MinedExample structure."""

    def test_example_creation(self):
        ex = MinedExample(
            repo_name="test-repo",
            instruction="Fix nil handling in parse-args",
            before={"src/core.clj": "(defn parse [args]\n  (first args))"},
            after={"src/core.clj": "(defn parse [args]\n  (when args (first args)))"},
            diff="""diff --git a/src/core.clj b/src/core.clj
--- a/src/core.clj
+++ b/src/core.clj
@@ -1,2 +1,2 @@
-(defn parse [args]
-  (first args))
+(defn parse [args]
+  (when args (first args)))""",
            changed_files=["src/core.clj"],
        )
        d = ex.to_dict()
        assert d["instruction"] == "Fix nil handling in parse-args"
        assert "src/core.clj" in d["input"]
        assert "diff --git" in d["output"]
        # REPL placeholder should be present
        assert ";; eval:" in d["output"]

    def test_to_jsonl_line(self):
        ex = MinedExample(
            repo_name="test-repo",
            instruction="Add validation",
            before={"src/core.clj": "(ns app)"},
            after={"src/core.clj": "(ns app)\n(defn valid? [x] x)"},
            diff="+ (defn valid? [x] x)",
            changed_files=["src/core.clj"],
        )
        line = ex.to_jsonl()
        parsed = json.loads(line)
        assert "system" in parsed
        assert "instruction" in parsed
        assert "input" in parsed
        assert "output" in parsed


class TestFormatting:
    """Test the output formatting utilities."""

    def test_format_before_state(self):
        from src.codeflow.git_mining.miner import _format_file_tree
        state = {
            "src/core.clj": "(ns app.core)\n(defn main [] 42)",
            "src/utils.clj": "(ns app.utils)\n(defn add [a b] (+ a b))",
        }
        formatted = _format_file_tree(state)
        assert "src/core.clj" in formatted
        assert "src/utils.clj" in formatted
        assert "(ns app.core)" in formatted
        assert "(defn add [a b] (+ a b))" in formatted

    def test_format_instruction_with_repl(self):
        from src.codeflow.git_mining.miner import _format_output_with_repl
        diff = "@@ -1 +1 @@\n-(def x 1)\n+(def x 2)"
        instruction = "Refactor: rename to meaningful name"
        files_before = {"src/core.clj": "(def x 1)"}
        files_after = {"src/core.clj": "(def x 2)"}

        output = _format_output_with_repl(diff, instruction, files_before, files_after)
        assert ";; nREPL session:" in output
        assert ";; apply:" in output
        assert diff in output
