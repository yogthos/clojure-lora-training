"""Tests for the stratified, leakage-safe train/val splitter."""

import pytest

from src.codeflow.assembly.splitter import (
    record_groups,
    record_stratum,
    stratified_split,
    assert_no_leakage,
)
from src.shared import WORKFLOW_SYSTEM_PROMPTS, TRANSITION_SYSTEM_PROMPTS


class TestRecordStratum:
    def test_workflow_objective_from_system(self):
        r = {"system": WORKFLOW_SYSTEM_PROMPTS[1], "instruction": "add a login flow"}
        obj, _ = record_stratum(r)
        assert obj == "workflow"

    def test_transition_objective_from_system(self):
        r = {"system": TRANSITION_SYSTEM_PROMPTS[2], "instruction": "fix the null bug"}
        obj, ctype = record_stratum(r)
        assert obj == "transition"
        assert ctype == "bug-fix"


class TestStratifiedSplit:
    def _data(self):
        recs = []
        for i in range(80):
            recs.append({"system": TRANSITION_SYSTEM_PROMPTS[0],
                         "instruction": f"fix bug {i}", "output": "diff", "source": "git"})
        for i in range(20):
            recs.append({"system": WORKFLOW_SYSTEM_PROMPTS[0],
                         "instruction": f"build feature {i}", "output": ";; Goal: x",
                         "source": "synthetic"})
        return recs

    def test_split_sizes_sum_to_total(self):
        train, val = stratified_split(self._data(), val_frac=0.1, seed=1)
        assert len(train) + len(val) == 100
        assert 8 <= len(val) <= 12

    def test_every_stratum_represented_in_both(self):
        train, val = stratified_split(self._data(), val_frac=0.2, seed=1)
        train_objs = {record_stratum(r)[0] for r in train}
        val_objs = {record_stratum(r)[0] for r in val}
        assert "transition" in train_objs and "workflow" in train_objs
        assert "transition" in val_objs and "workflow" in val_objs

    def test_no_leakage_between_splits(self):
        train, val = stratified_split(self._data(), val_frac=0.2, seed=1)
        assert_no_leakage(train, val)  # raises on overlap

    def test_deterministic_for_seed(self):
        a = stratified_split(self._data(), val_frac=0.1, seed=7)
        b = stratified_split(self._data(), val_frac=0.1, seed=7)
        assert [r["instruction"] for r in a[1]] == [r["instruction"] for r in b[1]]

    def test_empty(self):
        train, val = stratified_split([], val_frac=0.1)
        assert train == [] and val == []


class TestAssertNoLeakage:
    def test_raises_on_shared_record(self):
        dup = {"instruction": "x", "output": "y"}
        with pytest.raises(ValueError):
            assert_no_leakage([dup], [dict(dup)])

    def test_passes_when_disjoint(self):
        assert_no_leakage(
            [{"instruction": "a", "output": "1"}],
            [{"instruction": "b", "output": "2"}],
        )


class TestGroupedSplit:
    def _repo_data(self):
        recs = []
        for repo in ("a", "b", "c", "d", "e"):
            for i in range(20):
                recs.append({"system": TRANSITION_SYSTEM_PROMPTS[0], "repo": repo,
                             "instruction": f"fix bug {repo}{i}", "output": f"d{repo}{i}"})
            for i in range(4):
                recs.append({"system": WORKFLOW_SYSTEM_PROMPTS[0], "repo": repo,
                             "instruction": f"build {repo}{i}", "output": f"w{repo}{i}"})
        return recs

    def test_a_repo_lands_in_one_split(self):
        train, val = stratified_split(self._repo_data(), val_frac=0.2, seed=3)
        assert {r["repo"] for r in train}.isdisjoint({r["repo"] for r in val})
        assert val

    def test_same_input_or_instruction_grouped_without_repo(self):
        recs = [
            {"instruction": "Fix the  bug!", "input": "", "output": "1"},
            {"instruction": "fix the bug", "input": "", "output": "2"},
            {"instruction": "other", "input": "### src/a.clj\n(ns a)", "output": "3"},
            {"instruction": "another", "input": "### src/a.clj\n(ns a)", "output": "4"},
            {"instruction": "lonely", "input": "", "output": "5"},
        ]
        g = record_groups(recs)
        assert g[0] == g[1] and g[2] == g[3]
        assert len({g[0], g[2], g[4]}) == 3

    def test_leakage_check_covers_groups(self):
        with pytest.raises(ValueError, match="group"):
            assert_no_leakage([{"instruction": "a", "output": "1", "repo": "x"}],
                              [{"instruction": "b", "output": "2", "repo": "x"}])
