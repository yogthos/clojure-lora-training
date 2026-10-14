"""Tests for dataset validator — Clojure syntax, diff coherence, relevance."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.codeflow.assembly.validator import (
    check_clojure_syntax,
    check_diff_structure,
    score_relevance,
    validate_example,
    validate_jsonl_file,
    ValidationResult,
)


_THINK = "<think>\nThe handler should return a response map.\n</think>\n\n"


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


    def test_escaped_backslash_ends_string(self):
        assert check_clojure_syntax('(str "\\\\" "x")') == []

    def test_character_literals_are_not_brackets(self):
        assert check_clojure_syntax(r'(= c \( ) (= d \") (= e \;)') == []

    def test_mismatched_bracket_types(self):
        assert check_clojure_syntax("(let [x 1) x]")

    def test_unclosed_vector(self):
        assert check_clojure_syntax("(defn f [x (inc x))")


class TestCheckDiffStructure:
    def test_well_formed_diff_passes(self):
        diff = """diff --git a/src/core.clj b/src/core.clj
--- a/src/core.clj
+++ b/src/core.clj
@@ -10,3 +10,3 @@
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
@@ -1,1 +1,1 @@
-old
+new

diff --git a/b.clj b/b.clj
--- a/b.clj
+++ b/b.clj
@@ -5,1 +5,1 @@
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
                "<think>\nWrap the response as JSON in the middleware chain.\n</think>\n\n"
                "diff --git a/src/handler.clj b/src/handler.clj\n"
                "--- a/src/handler.clj\n"
                "+++ b/src/handler.clj\n"
                "@@ -10,3 +10,4 @@\n"
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


class TestStrictDiffGate:
    """A corrupt diff makes a record invalid no matter how well it scores."""

    GOOD = (
        "diff --git a/src/app/core.clj b/src/app/core.clj\n"
        "--- a/src/app/core.clj\n"
        "+++ b/src/app/core.clj\n"
        "@@ -1,2 +1,2 @@\n"
        " (ns app.core)\n"
        "-(defn handler [req] nil)\n"
        "+(defn handler [req] {:status 200})\n"
    )

    def _ex(self, output):
        return {
            "instruction": "Fix the ring handler in app.core to return a 200 response map",
            "input": "",
            "output": output,
        }

    def test_good_diff_is_valid(self):
        from src.codeflow.assembly.validator import validate_example
        assert validate_example(self._ex(_THINK + self.GOOD)).is_valid

    def test_bad_hunk_counts_are_invalid(self):
        from src.codeflow.assembly.validator import validate_example
        bad = self.GOOD.replace("@@ -1,2 +1,2 @@", "@@ -1,4 +1,4 @@")
        vr = validate_example(self._ex(bad))
        assert not vr.is_valid
        assert vr.diff_errors

    def test_workflow_without_apply_marker_is_invalid(self):
        from src.codeflow.assembly.validator import validate_example
        out = ";; eval: (+ 1 2)\n;; result: 3\n" + self.GOOD
        assert not validate_example(self._ex(out)).is_valid

    def test_minus_sign_alone_is_not_a_change(self):
        from src.codeflow.assembly.validator import check_diff_structure
        d = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n x-y\n"
        assert check_diff_structure(d)


class TestValidateJsonlFiltering:
    def test_invalid_records_are_removed_from_output(self, tmp_path):
        from src.codeflow.assembly.validator import validate_jsonl_file
        from src.shared import load_jsonl, write_jsonl
        good = {
            "instruction": "Fix the ring handler in app.core to return a 200 response map",
            "input": "",
            "output": _THINK + TestStrictDiffGate.GOOD,
        }
        bad = dict(good, output=TestStrictDiffGate.GOOD.replace("@@ -1,2 +1,2 @@", "@@ -1,9 +1,9 @@"))
        src = tmp_path / "in.jsonl"
        write_jsonl([good, bad, good], src)
        out = tmp_path / "out.jsonl"
        summary = validate_jsonl_file(src, tmp_path / "report.json", filtered_path=out)
        assert summary["invalid"] == 1
        assert len(load_jsonl(out)) == 2

    def test_report_only_mode_writes_no_filtered_file(self, tmp_path):
        from src.codeflow.assembly.validator import validate_jsonl_file
        from src.shared import write_jsonl
        src = tmp_path / "in.jsonl"
        write_jsonl([{"instruction": "x", "output": "y"}], src)
        validate_jsonl_file(src, tmp_path / "report.json")
        assert not (tmp_path / "out.jsonl").exists()


def _live_record(**over):
    from src.codeflow.synthetic.jolt_eval import EvalResult
    from src.codeflow.synthetic.live_trace import LiveTrace, Step
    diff = ("diff --git a/src/sq.clj b/src/sq.clj\n--- /dev/null\n+++ b/src/sq.clj\n"
            "@@ -0,0 +1,2 @@\n+(ns sq)\n+(defn square [x] (* x x))\n")
    steps = [Step("Plan: define square, check it.", "(defn square [x] (* x x))",
                  EvalResult(form="", value="#'user/square")),
             Step("Check a value.", "(square 4)", EvalResult(form="", value="16"))]
    rec = LiveTrace(request="add a square function to the math helpers", context="",
                    steps=steps, final_thinking="Works; write it.", diff=diff).to_record()
    rec.update(over)
    return rec


class TestMultiTurnValidation:
    def test_live_trace_is_valid(self):
        r = validate_example(_live_record())
        assert r.is_valid, r.to_dict()

    def test_eval_code_is_syntax_checked(self):
        rec = _live_record()
        call = rec["conversations"][1]
        call["value"] = call["value"].replace("(* x x))", "(* x x)")
        assert validate_example(rec).syntax_errors

    def test_function_call_needs_reasoning(self):
        rec = _live_record()
        rec["conversations"][3]["value"] = rec["conversations"][3]["value"].split("</think>\n\n")[1]
        r = validate_example(rec)
        assert not r.is_valid and any("reasoning" in e for e in r.format_errors)

    def test_function_call_must_be_an_eval_call(self):
        rec = _live_record()
        rec["conversations"][1]["value"] = "<think>\nx\n</think>\n\n{\"name\": \"shell\"}"
        assert not validate_example(rec).is_valid

    def test_turn_order(self):
        rec = _live_record()
        conv = rec["conversations"]
        conv[1], conv[2] = conv[2], conv[1]
        r = validate_example(rec)
        assert not r.is_valid and r.format_errors

    def test_final_turn_applies_a_diff(self):
        rec = _live_record()
        final = rec["conversations"][-1]
        final["value"] = final["value"].replace(";; apply:\n", "")
        rec["output"] = final["value"]
        assert not validate_example(rec).is_valid


class TestTransitionReasoning:
    _DIFF = ("diff --git a/src/core.clj b/src/core.clj\n--- a/src/core.clj\n+++ b/src/core.clj\n"
             "@@ -1,2 +1,2 @@\n (ns core)\n-(defn parse [x] x)\n+(defn parse [x] (when x x))\n")

    def test_transition_with_reasoning_is_valid(self):
        out = "<think>\nparse must tolerate nil input.\n</think>\n\n" + self._DIFF
        assert validate_example({"instruction": "handle nil in parse fn", "output": out}).is_valid

    def test_transition_without_reasoning_is_invalid(self):
        r = validate_example({"instruction": "handle nil in parse fn", "output": self._DIFF})
        assert not r.is_valid and any("reasoning" in e for e in r.format_errors)
