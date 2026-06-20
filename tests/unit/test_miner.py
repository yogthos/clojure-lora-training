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


class TestLifecycleWindowing:
    """IQuest §3.1: sample commits from the 40-80% percentile of project life,
    not the most-recent N from HEAD."""

    def _make_linear_repo(self, root: Path, n: int) -> None:
        def git(*args):
            subprocess.run(["git", "-C", str(root), *args], check=True,
                           capture_output=True)

        git("init")
        git("config", "user.email", "t@t")
        git("config", "user.name", "t")
        body = "(ns app)\n"
        for i in range(1, n + 1):
            body += f"(defn f-{i} [] {i})\n"
            (root / "core.clj").write_text(body)
            git("add", "-A")
            git("commit", "-m", f"commit {i}: add feature {i} to the namespace")

    def test_window_selects_middle_band(self):
        from src.codeflow.git_mining.miner import get_lifecycle_commits

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._make_linear_repo(root, 10)
            commits = get_lifecycle_commits(str(root), low=0.4, high=0.8)
            # 10 commits chronological -> indices [4,8) -> commits 5,6,7,8
            assert len(commits) == 4
            msgs = " ".join(c.message for c in commits)
            assert "commit 5:" in msgs and "commit 8:" in msgs
            assert "commit 1:" not in msgs and "commit 10:" not in msgs

    def test_window_is_chronological(self):
        from src.codeflow.git_mining.miner import get_lifecycle_commits

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._make_linear_repo(root, 10)
            commits = get_lifecycle_commits(str(root), low=0.0, high=1.0)
            nums = [int(c.message.split()[1].rstrip(":")) for c in commits]
            assert nums == sorted(nums)  # oldest -> newest

    def test_full_window_keeps_all(self):
        from src.codeflow.git_mining.miner import get_lifecycle_commits

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._make_linear_repo(root, 6)
            assert len(get_lifecycle_commits(str(root), low=0.0, high=1.0)) == 6


class TestTripletSpans:
    """IQuest §3.1: multi-iteration (R_old, P, R_new) triplets via forward spans."""

    def _make_linear_repo(self, root: Path, n: int) -> None:
        def git(*args):
            subprocess.run(["git", "-C", str(root), *args], check=True,
                           capture_output=True)

        git("init")
        git("config", "user.email", "t@t")
        git("config", "user.name", "t")
        body = "(ns app)\n"
        for i in range(1, n + 1):
            body += f"(defn f-{i} [] {i})\n"
            (root / "core.clj").write_text(body)
            git("add", "-A")
            git("commit", "-m", f"commit {i}: add feature {i} to the namespace")

    def test_span_produces_cumulative_diff(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._make_linear_repo(root, 9)
            examples = mine_repository(
                str(root), lifecycle_window=(0.0, 1.0), triplet_span=3,
            )
            assert len(examples) >= 1
            # The first arc spans commits 1-3, so its diff adds f-1, f-2 AND f-3
            # (a single-commit diff would only add one).
            first = examples[0]
            assert "f-1" in first.diff and "f-2" in first.diff and "f-3" in first.diff

    def test_span_one_is_per_commit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._make_linear_repo(root, 6)
            examples = mine_repository(
                str(root), lifecycle_window=(0.0, 1.0), triplet_span=1,
            )
            # One example per commit; each diff adds exactly its own feature.
            assert len(examples) == 6
            assert "f-1" in examples[0].diff and "f-2" not in examples[0].diff

    def test_instruction_combines_arc_messages(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._make_linear_repo(root, 6)
            examples = mine_repository(
                str(root), lifecycle_window=(0.0, 1.0), triplet_span=3,
            )
            # Arc instruction should reference more than one commit's intent.
            assert any(
                ex.instruction.count("feature") >= 2 for ex in examples
            )

    def test_lifecycle_window_limits_arcs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._make_linear_repo(root, 10)
            windowed = mine_repository(
                str(root), lifecycle_window=(0.4, 0.8), triplet_span=2,
            )
            full = mine_repository(
                str(root), lifecycle_window=(0.0, 1.0), triplet_span=2,
            )
            assert len(windowed) < len(full)


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
        # Output is the transition: a unified diff, no fabricated REPL trace.
        assert "diff --git" in d["output"]
        assert ";; eval:" not in d["output"]
        assert ";; nREPL" not in d["output"]
        # Git-mined examples use the transition (patch) system prompt, not the
        # interactive nREPL one.
        assert "unified diff" in d["system"].lower()
        assert "nrepl" not in d["system"].lower()

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

    def test_output_is_raw_transition_diff(self):
        ex = MinedExample(
            repo_name="r",
            instruction="Bump x",
            before={"src/core.clj": "(def x 1)"},
            after={"src/core.clj": "(def x 2)"},
            diff="diff --git a/src/core.clj b/src/core.clj\n@@ -1 +1 @@\n-(def x 1)\n+(def x 2)",
            changed_files=["src/core.clj"],
        )
        # Output is exactly the unified diff — nothing fabricated around it.
        assert ex.to_dict()["output"] == ex.diff
