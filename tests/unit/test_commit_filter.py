"""Tests for commit filtering logic."""

import pytest
from src.codeflow.git_mining.commit_filter import (
    CommitInfo,
    filter_clojure_commits,
    is_clojure_file,
    has_meaningful_message,
)


class TestCommitFilter:
    """Tests for commit filtering."""

    def _make_commit(self, message="Fix bug in parser", files=None):
        return CommitInfo(
            hash="abc123",
            message=message,
            files=files or ["src/core.clj"],
            is_merge=False,
        )

    def test_is_clojure_file(self):
        assert is_clojure_file("src/core.clj")
        assert is_clojure_file("app/components.cljs")
        assert is_clojure_file("shared/utils.cljc")
        assert is_clojure_file("config.edn")
        assert is_clojure_file("src/main.clj")
        assert not is_clojure_file("README.md")
        assert not is_clojure_file("src/main.py")
        assert not is_clojure_file("deps.edn")  # build config, not code

    def test_has_meaningful_message(self):
        assert has_meaningful_message("Fix nil handling in parse-args")
        assert has_meaningful_message("Add validation for user input")
        assert has_meaningful_message("Refactor: extract shared middleware")
        assert not has_meaningful_message("fix")
        assert not has_meaningful_message(".")
        assert not has_meaningful_message("wip")

    def test_excludes_non_clojure_commits(self):
        commits = [
            self._make_commit(files=["README.md", "docs/index.html"]),
            self._make_commit(files=["src/core.clj"]),
        ]
        result = filter_clojure_commits(commits)
        assert len(result) == 1
        assert result[0].hash == "abc123"

    def test_excludes_merge_commits(self):
        commits = [
            self._make_commit(files=["src/core.clj"]),
            CommitInfo(hash="merge1", message="Merge PR #42", files=["src/core.clj"], is_merge=True),
        ]
        result = filter_clojure_commits(commits)
        assert len(result) == 1
        assert result[0].hash == "abc123"

    def test_excludes_trivial_messages(self):
        commits = [
            self._make_commit(message="fix typo"),
            self._make_commit(message="Add comprehensive test suite for edge cases"),
        ]
        result = filter_clojure_commits(commits)
        assert len(result) == 1

    def test_excludes_build_only_commits(self):
        commits = [
            self._make_commit(files=["project.clj", "build.boot"]),
            self._make_commit(files=["src/core.clj"]),
        ]
        result = filter_clojure_commits(commits)
        assert len(result) == 1

    def test_cljc_and_cljs_included(self):
        commits = [
            self._make_commit(files=["shared/validation.cljc"]),
            self._make_commit(files=["app/components.cljs"]),
        ]
        result = filter_clojure_commits(commits)
        assert len(result) == 2

    def test_excludes_config_only_changes(self):
        commits = [
            self._make_commit(files=["resources/config.edn", "project.clj"]),
            self._make_commit(files=["src/handler.clj"]),
        ]
        result = filter_clojure_commits(commits)
        assert len(result) == 1
        assert result[0].files == ["src/handler.clj"]
