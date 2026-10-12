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


class TestSynthesizeInstruction:
    """Changelog-style commit messages must not leak file paths into instructions."""

    def _synth(self, msgs):
        from src.codeflow.git_mining.miner import _synthesize_instruction
        return _synthesize_instruction(msgs)

    def test_strips_changelog_path_prefix(self):
        out = self._synth(["* src/main/clojure/foo/bar.clj: add retry logic"])
        assert out == "add retry logic"

    def test_strips_path_prefix_without_leading_star(self):
        out = self._synth(["src/foo.cljs: fix nil handling"])
        assert out == "fix nil handling"

    def test_keeps_plain_messages_with_colons(self):
        # A normal "Title: detail" message (no path) is untouched.
        out = self._synth(["Refactor: extract shared middleware"])
        assert out == "Refactor: extract shared middleware"

    def test_cleans_each_message_in_arc(self):
        out = self._synth([
            "* src/a.clj: add retry logic to client",
            "* test/b.clj: cover timeout handling in specs",
        ])
        assert out == "add retry logic to client; cover timeout handling in specs"
        assert "src/" not in out and "test/" not in out

    def test_drops_changelog_line_with_no_description(self):
        # A bare "* path:" with nothing after it carries no intent.
        out = self._synth(["* src/foo.clj:", "Add real feature"])
        assert out == "Add real feature"


class TestCleanInstruction:
    """Repair already-synthesized instructions without re-mining."""

    def _clean(self, s):
        from src.codeflow.git_mining.miner import clean_instruction
        return clean_instruction(s)

    def test_cleans_joined_changelog_segments(self):
        s = "* src/a.clj: add retry; * test/b.clj: cover timeouts"
        assert self._clean(s) == "add retry; cover timeouts"

    def test_leaves_clean_instruction_untouched(self):
        s = "Fix nil handling in parse-args; Add validation"
        assert self._clean(s) == s

    def test_drops_empty_segments(self):
        assert self._clean("* src/a.clj: ; real change here") == "real change here"


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


class TestBeforeStateTrimming:
    """Input should show only the changed top-level forms (+ ns), not whole
    files — otherwise multi-file arcs blow past the training context window."""

    _BEFORE = (
        "(ns app)\n\n"
        "(defn f-1 [] 1)\n\n"
        "(defn f-2 [] 2)\n\n"
        "(defn f-3 [] 3)\n"
    )
    _DIFF = (
        "diff --git a/core.clj b/core.clj\n"
        "--- a/core.clj\n+++ b/core.clj\n"
        "@@ -5,1 +5,1 @@\n-(defn f-2 [] 2)\n+(defn f-2 [] 22)\n"
    )

    def test_form_spans_identify_top_level_forms(self):
        from src.codeflow.git_mining.miner import _top_level_form_spans
        spans = _top_level_form_spans(self._BEFORE)
        # (ns app) at line 1, f-1 at 3, f-2 at 5, f-3 at 7
        starts = [s for (s, e, t) in spans]
        assert starts == [1, 3, 5, 7]

    def test_keeps_changed_form_and_ns_drops_others(self):
        from src.codeflow.git_mining.miner import _format_changed_regions
        out = _format_changed_regions({"core.clj": self._BEFORE}, self._DIFF)
        assert "(ns app)" in out          # ns kept for context
        assert "f-2" in out               # the changed form kept
        assert "f-1" not in out           # untouched forms dropped
        assert "f-3" not in out
        assert "core.clj" in out          # file header preserved

    def test_trimmed_is_smaller_than_whole_file(self):
        from src.codeflow.git_mining.miner import (
            _format_changed_regions, _format_file_tree,
        )
        trimmed = _format_changed_regions({"core.clj": self._BEFORE}, self._DIFF)
        whole = _format_file_tree({"core.clj": self._BEFORE})
        assert len(trimmed) < len(whole)

    def test_to_dict_input_is_trimmed(self):
        # Files over the whole-file budget show only the changed forms.
        before = self._BEFORE + "\n" + "".join(f"(defn pad-{i} [] {i})\n" for i in range(1000))
        ex = MinedExample(
            repo_name="r",
            instruction="bump f-2",
            before={"core.clj": before},
            after={"core.clj": before.replace("[] 2)", "[] 22)")},
            diff=self._DIFF,
            changed_files=["core.clj"],
        )
        d = ex.to_dict()
        assert "f-2" in d["input"] and "f-1" not in d["input"] and "pad-500" not in d["input"]


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


class TestDiffsApply:
    """Mined diffs must be valid patches against the before-state."""

    def _repo_with_trailing_blank_context(self, root: Path) -> None:
        def git(*args):
            subprocess.run(["git", "-C", str(root), *args], check=True,
                           capture_output=True)

        git("init")
        git("config", "user.email", "t@t")
        git("config", "user.name", "t")
        # Blank lines at the end of the file become trailing ' ' context lines
        # in the last hunk — exactly what .strip() used to eat.
        (root / "core.clj").write_text("(ns app)\n\n(defn a [] 1)\n\n\n\n")
        git("add", "-A")
        git("commit", "-m", "initial namespace with a function")
        (root / "core.clj").write_text("(ns app)\n\n(defn a [] 2)\n\n\n\n")
        git("add", "-A")
        git("commit", "-m", "change the return value of a")
        return git

    def _applies(self, root: Path, base: str, diff: str) -> bool:
        subprocess.run(["git", "-C", str(root), "checkout", "-q", base], check=True)
        r = subprocess.run(["git", "-C", str(root), "apply", "--check", "-"],
                           input=diff, text=True, capture_output=True)
        return r.returncode == 0

    def test_range_diff_keeps_trailing_context(self):
        from src.codeflow.git_mining.miner import get_range_diff
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._repo_with_trailing_blank_context(root)
            diff = get_range_diff(str(root), "HEAD~1", "HEAD")
            assert self._applies(root, "HEAD~1", diff)

    def test_commit_diff_keeps_trailing_context(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._repo_with_trailing_blank_context(root)
            diff = get_commit_diff(str(root), "HEAD")
            assert self._applies(root, "HEAD~1", diff)


def _git(root: Path, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _init(root: Path):
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")


def _commit(root: Path, files: dict, msg: str):
    for path, content in files.items():
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)


def _mine(root: Path, span: int):
    return mine_repository(str(root), lifecycle_window=(0.0, 1.0), triplet_span=span)


def _diff_paths(diff: str):
    return [l.split(" b/", 1)[1] for l in diff.splitlines() if l.startswith("diff --git ")]


class TestOutputMatchesInput:
    """The target diff may only touch files the input shows (or creates)."""

    def test_build_files_are_not_in_output(self, tmp_path):
        _init(tmp_path)
        _commit(tmp_path, {"src/app/core.clj": "(ns app.core)\n(defn f [] 1)\n",
                           "project.clj": '(defproject app "0.1.0")\n'},
                "initial project skeleton with core namespace")
        _commit(tmp_path, {"src/app/core.clj": "(ns app.core)\n(defn f [] 2)\n",
                           "project.clj": '(defproject app "0.2.0")\n',
                           "deps.edn": "{:deps {}}\n"},
                "change f to return two and bump version")
        ex = _mine(tmp_path, 1)[-1]
        assert _diff_paths(ex.diff) == ["src/app/core.clj"]
        assert "### src/app/core.clj" in ex.to_dict()["input"]

    def test_every_modified_output_file_is_in_input(self, tmp_path):
        _init(tmp_path)
        _commit(tmp_path, {"src/a.clj": "(ns a)\n(defn a [] 1)\n",
                           "resources/config.edn": "{:port 1}\n"},
                "initial namespaces and config file")
        _commit(tmp_path, {"src/a.clj": "(ns a)\n(defn a [] 2)\n",
                           "resources/config.edn": "{:port 2}\n"},
                "change a to return two and update port")
        ex = _mine(tmp_path, 1)[-1]
        inp = ex.to_dict()["input"]
        for path in _diff_paths(ex.diff):
            assert f"### {path}" in inp


class TestRenamedFiles:
    def test_renamed_file_input_has_original_content(self, tmp_path):
        _init(tmp_path)
        body = "(ns util)\n" + "".join(f"(defn f{i} [] {i})\n" for i in range(12))
        _commit(tmp_path, {"src/util.clj": body}, "initial utility namespace functions")
        (tmp_path / "src/helpers.clj").write_text(
            body.replace("(ns util)", "(ns helpers)").replace("(defn f3 [] 3)", "(defn f3 [] 33)"))
        (tmp_path / "src/util.clj").unlink()
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-q", "-m", "rename util namespace to helpers and fix f3")
        ex = _mine(tmp_path, 1)[-1]
        assert "helpers.clj" in ex.diff
        assert "(defn f3 [] 3)" in ex.to_dict()["input"]
        for path, content in ex.before.items():
            assert content.strip(), f"empty before-content for {path}"


class TestArcBoundaries:
    """An arc's diff must only contain changes from commits it describes."""

    def test_filtered_commit_breaks_arc(self, tmp_path):
        _init(tmp_path)
        _commit(tmp_path, {"core.clj": "(ns c)\n(def a 1)\n"}, "add var a to the namespace")
        _commit(tmp_path, {"core.clj": "(ns c)\n(def a 1)\n(def b 2)\n"}, "wip")
        _commit(tmp_path, {"core.clj": "(ns c)\n(def a 1)\n(def b 2)\n(def d 4)\n"},
                "add var d to the namespace")
        for ex in _mine(tmp_path, 3):
            assert "+(def b 2)" not in ex.diff, ex.instruction

    def test_side_branch_commits_are_not_folded_into_mainline_arc(self, tmp_path):
        _init(tmp_path)
        _commit(tmp_path, {"core.clj": "(ns c)\n(def a 1)\n"}, "add var a to the namespace")
        _git(tmp_path, "checkout", "-q", "-b", "side")
        _commit(tmp_path, {"side.clj": "(ns side)\n(def s 1)\n"}, "add the side namespace here")
        _git(tmp_path, "checkout", "-q", "main")
        _commit(tmp_path, {"core.clj": "(ns c)\n(def a 2)\n"}, "change var a to two now")
        _git(tmp_path, "merge", "-q", "--no-ff", "-m", "Merge branch side", "side")
        _commit(tmp_path, {"core.clj": "(ns c)\n(def a 3)\n"}, "change var a to three now")
        exs = _mine(tmp_path, 3)
        assert exs
        for ex in exs:
            assert "side.clj" not in ex.diff
            assert "side namespace" not in ex.instruction

    def test_non_clojure_commit_does_not_break_arc(self, tmp_path):
        _init(tmp_path)
        _commit(tmp_path, {"core.clj": "(ns c)\n(def a 1)\n"}, "add var a to the namespace")
        _commit(tmp_path, {"README.md": "hi\n"}, "x")
        _commit(tmp_path, {"core.clj": "(ns c)\n(def a 1)\n(def d 4)\n"},
                "add var d to the namespace")
        exs = _mine(tmp_path, 2)
        assert len(exs) == 1
        assert "add var a" in exs[0].instruction and "add var d" in exs[0].instruction


class TestElisionMarkers:
    _BEFORE = "(ns app)\n\n(defn f-1 [] 1)\n\n(defn f-2 [] 2)\n\n(defn f-3 [] 3)\n"

    def _fmt(self, diff):
        from src.codeflow.git_mining.miner import _format_changed_regions
        return _format_changed_regions({"core.clj": self._BEFORE}, diff)

    def test_no_marker_between_adjacent_kept_forms(self):
        diff = ("diff --git a/core.clj b/core.clj\n--- a/core.clj\n+++ b/core.clj\n"
                "@@ -1,3 +1,3 @@\n (ns app)\n \n-(defn f-1 [] 1)\n+(defn f-1 [] 11)\n")
        out = self._fmt(diff)
        assert ";; ..." not in out
        assert "(ns app)\n\n(defn f-1 [] 1)" in out

    def test_marker_where_forms_were_dropped(self):
        diff = ("diff --git a/core.clj b/core.clj\n--- a/core.clj\n+++ b/core.clj\n"
                "@@ -5,1 +5,1 @@\n-(defn f-2 [] 2)\n+(defn f-2 [] 22)\n")
        out = self._fmt(diff)
        assert "(ns app)\n;; ...\n(defn f-2 [] 2)" in out

    def test_hunk_between_forms_keeps_neighbours(self):
        # Inserting a form into the blank gap after f-2: the old range is just
        # the blank line, which overlaps no form.
        diff = ("diff --git a/core.clj b/core.clj\n--- a/core.clj\n+++ b/core.clj\n"
                "@@ -6,0 +6,2 @@\n+(defn g [] 0)\n+\n")
        out = self._fmt(diff)
        assert "f-2" in out and "f-3" in out


def test_mined_record_carries_repo():
    ex = MinedExample(repo_name="owner__lib", instruction="fix x", before={}, after={},
                      diff="diff --git a/a b/a\n", changed_files=[])
    assert ex.to_dict()["repo"] == "owner__lib"
