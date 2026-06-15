"""Tests for commit session grouper."""

import pytest
from src.git_mining.session_grouper import (
    CommitWithDiff,
    CommitSession,
    group_by_pr_boundary,
    group_by_time_window,
    group_by_prefix,
)


def _commit(msg, ts="2024-01-01T12:00:00+00:00", hash_="abc"):
    return CommitWithDiff(
        hash=hash_,
        message=msg,
        timestamp=ts,
        diff_text="mock diff",
        changed_files=["src/core.clj"],
    )


class TestGroupByPRBoundary:
    def test_empty_list(self):
        assert group_by_pr_boundary([]) == []

    def test_single_commit(self):
        commits = [_commit("Fix nil check")]
        sessions = group_by_pr_boundary(commits)
        assert len(sessions) == 1
        assert len(sessions[0].commits) == 1

    def test_multiple_commits_one_session(self):
        commits = [
            _commit("Add validation", "2024-01-01T12:00:00Z"),
            _commit("Add tests for validation", "2024-01-01T12:05:00Z"),
            _commit("Fix edge case", "2024-01-01T12:10:00Z"),
        ]
        sessions = group_by_pr_boundary(commits)
        assert len(sessions) == 1
        assert len(sessions[0].commits) == 3

    def test_merge_commit_splits_sessions(self):
        commits = [
            _commit("Add feature X", "2024-01-01T12:00:00Z"),
            _commit("Add tests", "2024-01-01T12:05:00Z"),
            _commit("Merge pull request #42", "2024-01-01T13:00:00Z"),
            _commit("Fix unrelated bug", "2024-01-02T10:00:00Z"),
        ]
        sessions = group_by_pr_boundary(commits)
        assert len(sessions) == 2
        assert len(sessions[0].commits) == 2  # first 2 before merge
        assert len(sessions[1].commits) == 1  # after merge

    def test_merge_branch_splits_sessions(self):
        commits = [
            _commit("Refactor handler"),
            _commit("Merge branch 'feature/x'"),
            _commit("Update docs"),
        ]
        sessions = group_by_pr_boundary(commits)
        assert len(sessions) == 2


class TestGroupByTimeWindow:
    def test_empty_list(self):
        assert group_by_time_window([]) == []

    def test_commits_within_window(self):
        commits = [
            _commit("A", "2024-01-01T12:00:00Z"),
            _commit("B", "2024-01-01T13:00:00Z"),  # 1 hour gap
            _commit("C", "2024-01-01T14:30:00Z"),  # 1.5 hour gap
        ]
        sessions = group_by_time_window(commits, max_window_hours=4)
        assert len(sessions) == 1
        assert len(sessions[0].commits) == 3

    def test_commits_outside_window(self):
        commits = [
            _commit("A", "2024-01-01T12:00:00Z"),
            _commit("B", "2024-01-01T18:00:00Z"),  # 6 hour gap
        ]
        sessions = group_by_time_window(commits, max_window_hours=4)
        assert len(sessions) == 2

    def test_default_window(self):
        commits = [
            _commit("A", "2024-01-01T12:00:00Z"),
            _commit("B", "2024-01-01T15:00:00Z"),  # 3 hours
            _commit("C", "2024-01-01T17:00:00Z"),  # 2 hours
        ]
        sessions = group_by_time_window(commits)  # default 4 hours
        assert len(sessions) == 1


class TestGroupByPrefix:
    def test_empty_list(self):
        assert group_by_prefix([]) == []

    def test_same_prefix_grouped(self):
        commits = [
            _commit("[feature] add handler"),
            _commit("[feature] add tests"),
            _commit("[feature] update docs"),
        ]
        sessions = group_by_prefix(commits)
        assert len(sessions) == 1
        assert len(sessions[0].commits) == 3

    def test_different_prefixes_split(self):
        commits = [
            _commit("[feature] add handler"),
            _commit("[bugfix] fix crash"),
        ]
        sessions = group_by_prefix(commits)
        assert len(sessions) == 2

    def test_no_prefix_creates_no_group(self):
        commits = [
            _commit("random commit without prefix"),
        ]
        sessions = group_by_prefix(commits)
        assert len(sessions) == 0

    def test_wip_colon(self):
        commits = [
            _commit("WIP: start refactoring"),
            _commit("WIP: continue refactoring"),
        ]
        sessions = group_by_prefix(commits)
        assert len(sessions) == 1
        assert len(sessions[0].commits) == 2

    def test_interspersed_no_prefix_splits(self):
        commits = [
            _commit("[feature] add X"),
            _commit("random non-prefixed"),
            _commit("[feature] add Y"),
        ]
        sessions = group_by_prefix(commits)
        assert len(sessions) == 2  # feature group, then another feature group
