"""Tests for grounding synthetic solutions in real babashka execution."""

import shutil
import pytest

from src.codeflow.synthetic.verify import (
    extract_eval_blocks,
    ground_solution,
    verify_and_ground,
    verify_workflow,
    _is_definition,
)

_BB = shutil.which("bb") is not None


class TestExtractEvalBlocks:
    def test_single_form(self):
        sol = ";; nREPL session:\n;; eval: (+ 1 2)\n;; result: 3\n;; apply:\n"
        blocks = extract_eval_blocks(sol)
        assert len(blocks) == 1
        assert blocks[0].form == "(+ 1 2)"

    def test_multi_line_form(self):
        sol = (
            ";; eval: (defn sq [x]\n"
            ";;        (* x x))\n"
            ";; result: #'user/sq\n"
        )
        blocks = extract_eval_blocks(sol)
        assert len(blocks) == 1
        assert "(defn sq [x]" in blocks[0].form
        assert "(* x x))" in blocks[0].form

    def test_multiple_blocks(self):
        sol = (
            ";; eval: (def x 1)\n;; result: #'user/x\n"
            ";; eval: (inc x)\n;; result: 2\n"
        )
        blocks = extract_eval_blocks(sol)
        assert [b.form for b in blocks] == ["(def x 1)", "(inc x)"]

    def test_no_eval_blocks(self):
        assert extract_eval_blocks("just some text\n;; apply:\n") == []


class TestGroundSolution:
    def test_replaces_result_with_real_value(self):
        sol = (
            ";; eval: (+ 1 2)\n;; result: 999\n"
            ";; apply:\ndiff --git a/x b/x\n"
        )
        blocks = extract_eval_blocks(sol)

        class R:  # minimal stand-in for EvalResult
            def __init__(self, value, ok=True, error="", stdout=""):
                self.value, self.ok, self.error, self.stdout = value, ok, error, stdout

        grounded = ground_solution(sol, blocks, [R("3")])
        assert ";; result: 3" in grounded
        assert "999" not in grounded
        # The apply/diff tail is preserved.
        assert "diff --git a/x b/x" in grounded


@pytest.mark.skipif(not _BB, reason="babashka not installed")
class TestVerifyAndGround:
    def test_self_contained_session_grounds_to_real_values(self):
        sol = (
            ";; nREPL session:\n"
            ";; eval: (defn sq [x] (* x x))\n;; result: WRONG\n"
            ";; eval: (sq 5)\n;; result: WRONG\n"
            ";; apply:\ndiff --git a/core.clj b/core.clj\n@@ -1 +1 @@\n"
        )
        g = verify_and_ground(sol)
        assert g.total == 2
        assert g.passed == 2
        assert g.pass_rate == 1.0
        assert ";; result: 25" in g.solution
        assert "#'user/sq" in g.solution
        assert "WRONG" not in g.solution

    def test_erroring_form_lowers_pass_rate(self):
        sol = (
            ";; eval: (+ 1 1)\n;; result: x\n"
            ";; eval: (/ 1 0)\n;; result: x\n"
        )
        g = verify_and_ground(sol)
        assert g.total == 2
        assert g.passed == 1
        assert g.pass_rate == 0.5
        assert not g.all_ok

    def test_no_forms_is_not_verifiable(self):
        g = verify_and_ground(";; apply:\ndiff --git a/x b/x\n")
        assert g.total == 0
        assert g.all_ok is False  # nothing executed -> can't vouch for it


class TestIsDefinition:
    def test_definitions(self):
        assert _is_definition("(defn f [x] x)")
        assert _is_definition("(def x 1)")
        assert _is_definition("(require '[clojure.string :as s])")

    def test_demonstrations(self):
        assert not _is_definition("(f 5)")
        assert not _is_definition("(+ 1 2)")
        assert not _is_definition("(map inc [1 2 3])")


@pytest.mark.skipif(not _BB, reason="babashka not installed")
class TestVerifyWorkflow:
    def test_intermediate_failure_then_recovery_is_kept(self):
        # First version of sq is wrong (adds), gets run, then fixed and re-run.
        sol = (
            ";; --- Step 1: sq ---\n"
            ";; eval: (defn sq [x] (+ x x))\n;; result: ?\n"
            ";; eval: (sq 5)\n;; result: ?\n"          # 10 — wrong, but runs
            ";; eval: (defn sq [x] (* x x))\n;; result: ?\n"
            ";; eval: (sq 5)\n;; result: ?\n"          # 25 — correct, final demo
            ";; apply:\ndiff --git a/c.clj b/c.clj\n@@ -1 +1 @@\n"
        )
        g = verify_workflow(sol)
        assert g.total == 4
        assert g.reaches_correct_end_state is True   # final demo (sq 5)=25 ok
        assert "25" in g.solution

    def test_failing_final_demo_is_rejected(self):
        sol = (
            ";; eval: (defn f [x] (/ x 0))\n;; result: ?\n"
            ";; eval: (f 5)\n;; result: ?\n"           # throws — final demo fails
            ";; apply:\ndiff --git a/c.clj b/c.clj\n"
        )
        g = verify_workflow(sol)
        assert g.reaches_correct_end_state is False

    def test_only_definitions_has_no_end_state(self):
        sol = ";; eval: (defn f [x] x)\n;; result: ?\n;; apply:\ndiff --git a/x b/x\n"
        g = verify_workflow(sol)
        # A definition alone isn't a demonstration of correctness.
        assert g.reaches_correct_end_state is False


class _R:
    """Minimal stand-in for EvalResult."""
    def __init__(self, value, ok=True, error="", stdout=""):
        self.value, self.ok, self.error, self.stdout = value, ok, error, stdout


class TestNarrativePreserved:
    """Prose after a result line is narrative, not part of the result."""

    def test_step_header_after_result_survives_grounding(self):
        sol = (
            ";; --- Step 1: sq ---\n"
            ";; eval: (defn sq [x] (+ x x))\n;; result: ?\n"
            ";; eval: (sq 5)\n;; result: 10\n"
            ";; That's wrong: it doubles instead of squaring. Fix:\n"
            ";; eval: (defn sq [x] (* x x))\n;; result: ?\n"
            ";; --- Step 2: use ---\n"
            ";; eval: (sq 5)\n;; result: ?\n"
            ";; apply:\n"
        )
        blocks = extract_eval_blocks(sol)
        assert [b.form for b in blocks] == [
            "(defn sq [x] (+ x x))", "(sq 5)", "(defn sq [x] (* x x))", "(sq 5)",
        ]
        out = ground_solution(sol, blocks, [_R("#'user/sq"), _R("10"), _R("#'user/sq"), _R("25")])
        assert ";; That's wrong: it doubles instead of squaring. Fix:" in out
        assert ";; --- Step 2: use ---" in out
        assert ";; result: 25" in out

    def test_narrative_after_form_without_result_is_not_part_of_form(self):
        sol = (
            ";; eval: (inc 1)\n"
            ";; Now check the edge case.\n"
            ";; eval: (inc 2)\n;; result: ?\n"
        )
        blocks = extract_eval_blocks(sol)
        assert [b.form for b in blocks] == ["(inc 1)", "(inc 2)"]
        out = ground_solution(sol, blocks, [_R("2"), _R("3")])
        assert out.splitlines() == [
            ";; eval: (inc 1)", ";; result: 2",
            ";; Now check the edge case.",
            ";; eval: (inc 2)", ";; result: 3",
        ]

    def test_multiline_fabricated_result_is_replaced_whole(self):
        sol = (
            ";; eval: (f)\n"
            ";; result: {:a 1\n"
            ";;          :b 2}\n"
            ";; Looks right.\n"
        )
        blocks = extract_eval_blocks(sol)
        out = ground_solution(sol, blocks, [_R("{:a 1, :b 2}")])
        assert out.splitlines() == [";; eval: (f)", ";; result: {:a 1, :b 2}", ";; Looks right."]

    def test_extra_fabricated_result_lines_are_dropped(self):
        sol = ";; eval: (/ 1 0)\n;; result: ERROR: invented\n;; result: also invented\n;; next\n"
        blocks = extract_eval_blocks(sol)
        out = ground_solution(sol, blocks, [_R("", ok=False, error="Divide by zero")])
        assert "invented" not in out
        assert out.splitlines()[-1] == ";; next"


class TestRawCodeForms:
    """Forms written as raw (non-';;') code after the eval marker."""

    def test_raw_multiline_form_is_extracted(self):
        sol = (
            ";; eval:\n"
            "(defn sq [x]\n"
            "  (* x x))\n"
            ";; result: ?\n"
            ";; eval: (sq 3)\n;; result: ?\n"
        )
        blocks = extract_eval_blocks(sol)
        assert [b.form for b in blocks] == ["(defn sq [x]\n  (* x x))", "(sq 3)"]
        out = ground_solution(sol, blocks, [_R("#'user/sq"), _R("9")])
        assert out.splitlines() == [
            ";; eval:", "(defn sq [x]", "  (* x x))", ";; result: #'user/sq",
            ";; eval: (sq 3)", ";; result: 9",
        ]

    def test_missing_result_is_inserted_after_whole_form(self):
        sol = ";; eval: (defn sq [x]\n;;   (* x x))\n;; eval: (sq 2)\n"
        blocks = extract_eval_blocks(sol)
        out = ground_solution(sol, blocks, [_R("#'user/sq"), _R("4")])
        assert out.splitlines() == [
            ";; eval: (defn sq [x]", ";;   (* x x))", ";; result: #'user/sq",
            ";; eval: (sq 2)", ";; result: 4",
        ]

    def test_brackets_in_strings_and_chars_do_not_confuse_balance(self):
        sol = ';; eval: (str "(" \\) "[")\n;; result: ?\n;; done\n'
        blocks = extract_eval_blocks(sol)
        assert blocks[0].form == '(str "(" \\) "[")'
        assert not blocks[0].malformed

    def test_unbalanced_form_is_flagged(self):
        sol = ";; eval: (defn broken [x]\n;; result: ?\n;; apply:\n"
        blocks = extract_eval_blocks(sol)
        assert blocks[0].malformed


@pytest.mark.skipif(not _BB, reason="babashka not installed")
class TestMalformedTracesRejected:
    def test_malformed_block_fails_verification(self):
        sol = (
            ";; eval: (defn sq [x] (* x x))\n;; result: ?\n"
            ";; eval: (sq 5\n;; result: ?\n"
            ";; eval: (sq 5)\n;; result: ?\n"
        )
        g = verify_workflow(sol)
        assert g.malformed == 1
        assert not g.all_ok
        assert g.reaches_correct_end_state is False

    def test_results_pair_with_their_own_forms(self):
        sol = (
            ";; eval:\n;; result: orphan\n"   # empty form: nothing to run
            ";; eval: (+ 1 2)\n;; result: ?\n"
        )
        g = verify_and_ground(sol)
        assert ";; result: 3" in g.solution
        assert g.malformed == 1


class TestExtraCloserIsARealReplError:
    def test_extra_closing_paren_is_runnable_not_malformed(self):
        # A stray ')' is a reader error the REPL reports — a legitimate
        # error->fix step, not an unparseable trace.
        sol = ";; eval: (defn f []\n;;   1))\n;; result: ?\n;; eval: (f)\n"
        blocks = extract_eval_blocks(sol)
        assert blocks[0].form == "(defn f []\n  1))"
        assert not blocks[0].malformed
        assert blocks[1].form == "(f)"
