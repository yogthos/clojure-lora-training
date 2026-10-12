"""Tests for the mining CLI's pattern filter and clone layout."""

import importlib.util
import pathlib

from src.codeflow.git_mining.pattern_classifier import DiffClassification, classify_diff


def _script():
    return importlib.import_module("src.codeflow.cli.mine_clojure_repos")


class TestPatternChoices:
    def test_every_cli_choice_maps_to_a_classification_field(self):
        mod = _script()
        fields = set(DiffClassification.__dataclass_fields__)
        for choice, attr in mod.PATTERN_ATTRS.items():
            assert attr in fields, choice

    def test_pattern_filter_matches_macro(self):
        mod = _script()
        c = classify_diff("+(defmacro m [x] x)", ["src/core.clj"])
        assert mod.matches_patterns(c, ["macro"])
        assert not mod.matches_patterns(c, ["spec"])


class TestCloneTargets:
    def _mod(self):
        mod = _script()
        return mod

    def test_same_repo_name_different_owner_get_distinct_dirs(self):
        mod = _script()
        a = mod.clone_target("owner1/utils", "/tmp/c")
        b = mod.clone_target("https://github.com/owner2/utils.git", "/tmp/c")
        assert a != b
        assert a.name == "owner1__utils" and b.name == "owner2__utils"

    def test_ssh_url(self):
        mod = _script()
        assert mod.clone_target("git@github.com:owner/repo.git", "/tmp/c").name == "owner__repo"
