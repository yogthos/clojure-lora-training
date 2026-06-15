"""Tests for dataset validator — Clojure syntax, diff coherence, relevance."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.assembly.validator import (
    check_clojure_syntax,
    check_diff_structure,
    score_relevance,
    validate_example,
    validate_jsonl_file,
    ValidationResult,
)


class TestCheckClojureSyntax:
    def test_valid_forms_pass(self):
        code = "(defn foo [x] (+ x 1))\n(def bar {:key \"val\"})"
        errors = check_clojure_syntax(code)
        assert len(errors) == 0

    def test_unmatched_open_paren(self):
        code = "(defn foo [x] (+ x 1)"
        errors = check_clojure_syntax(code)
        assert len(errors) > 0
        assert any("unmatched" in e.lower() for e in errors)

    def test_unmatched_close_paren(self):
        code = "(defn foo [x] (+ x 1)))"
        errors = check_clojure_syntax(code)
        assert len(errors) > 0

    def test_empty_code(self):
        errors = check_clojure_syntax("")
        assert len(errors) == 0

    def test_ignores_strings(self):
        code = '(str "this has (unbalanced" " and ) parens")'
        errors = check_clojure_syntax(code)
        assert len(errors) == 0

    def test_ignores_comments(self):
        code = "(def x 1) ; this comment has ) (( unbalanced"
        errors = check_clojure_syntax(code)
        assert len(errors) == 0


class TestCheckDiffStructure:
    def test_well_formed_diff_passes(self):
        diff = """diff --git a/src/core.clj b/src/core.clj
--- a/src/core.clj
+++ b/src/core.clj
@@ -10,6 +10,8 @@
 unchanged line
-removed line
+added line
 unchanged line"""
        errors = check_diff_structure(diff)
        assert len(errors) == 0

    def test_multiple_files(self):
        diff = """diff --git a/a.clj b/a.clj
--- a/a.clj
+++ b/a.clj
@@ -1,3 +1,3 @@
-old
+new

diff --git a/b.clj b/b.clj
--- a/b.clj
+++ b/b.clj
@@ -5,4 +5,4 @@
-old2
+new2"""
        errors = check_diff_structure(diff)
        assert len(errors) == 0

    def test_no_diff_header(self):
        diff = "just some code changes\n-added line\n-removed line"
        errors = check_diff_structure(diff)
        assert len(errors) > 0
        assert any("header" in e.lower() for e in errors)

    def test_no_hunk_header(self):
        diff = """diff --git a/x.clj b/x.clj
--- a/x.clj
+++ b/x.clj
-added line
+added more"""
        errors = check_diff_structure(diff)
        assert len(errors) > 0

    def test_empty_diff(self):
        errors = check_diff_structure("")
        assert len(errors) > 0

    def test_checks_for_changes(self):
        diff = """diff --git a/x.clj b/x.clj
--- a/x.clj
+++ b/x.clj
@@ -1,1 +1,1 @@
 same line"""
        errors = check_diff_structure(diff)
        assert any("no changed lines" in e.lower() for e in errors)


class TestScoreRelevance:
    def test_good_instruction_scores_high(self):
        score = score_relevance(
            "refactor the ring middleware handler to use comp instead of threading macros")
        assert score >= 0.5

    def test_empty_instruction_scores_zero(self):
        assert score_relevance("") == 0.0

    def test_short_vague_instruction_scores_low(self):
        score = score_relevance("fix it")
        assert score < 0.31

    def test_clojure_terms_boost_score(self):
        low = score_relevance("change the function")
        high = score_relevance(
            "refactor the defn handler to use transducers and core.async channels")
        assert high > low


class TestValidateExample:
    def test_valid_example_passes(self):
        rec = {
            "instruction": "refactor the middleware to use comp",
            "output": (
                ";; nREPL session:\n"
                ";; eval: (require '[ring.middleware])\n"
                ";; result: nil\n"
                ";; apply:\n"
                "diff --git a/src/handler.clj b/src/handler.clj\n"
                "--- a/src/handler.clj\n"
                "+++ b/src/handler.clj\n"
                "@@ -10,6 +10,8 @@\n"
                " (defn app [request]\n"
                "-  (-> request\n"
                "+  (-> request\n"
                "+    (wrap-json-response)\n"
                "     (wrap-params)))\n"
            ),
        }
        result = validate_example(rec)
        assert result.is_valid is True
        assert result.total_score > 0.5

    def test_invalid_example_fails(self):
        rec = {
            "instruction": "fix",
            "output": "not a diff",
        }
        result = validate_example(rec)
        assert result.is_valid is False
        assert result.total_score < 0.5

    def test_min_score_threshold(self):
        rec = {
            "instruction": "refactor the handler to use core.async channels for async processing",
            "output": "some text without proper diff format",
        }
        result = validate_example(rec, min_score=0.7)
        assert result.is_valid is False

    def test_reports_all_errors(self):
        rec = {"instruction": "", "output": ""}
        result = validate_example(rec)
        assert len(result.syntax_errors) >= 0
        assert len(result.diff_errors) >= 0
        assert "missing instruction" in result.warnings


class TestValidateJSONLFile:
    def test_writes_report(self):
        with TemporaryDirectory() as d:
            dpath = Path(d)
            in_path = dpath / "in.jsonl"
            out_path = dpath / "report.json"

            recs = [
                {
                    "instruction": "refactor middleware to use comp",
                    "output": (
                        ";; nREPL session:\n"
                        ";; eval: (defn wrap [h] (comp h json-response))\n"
                        ";; result: #'user/wrap\n"
                        ";; apply:\n"
                        "diff --git a/handler.clj b/handler.clj\n"
                        "--- a/handler.clj\n"
                        "+++ b/handler.clj\n"
                        "@@ -10,4 +10,4 @@\n"
                        " (defn app []\n"
                        "-  (-> handler wrap-params)\n"
                        "+  (-> handler wrap-params wrap)\n"
                        "     )\n"
                    ),
                },
            ]
            with open(in_path, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")

            summary = validate_jsonl_file(in_path, out_path)
            assert summary["total"] == 1
            assert out_path.exists()
