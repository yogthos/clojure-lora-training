"""Tests for grounding synthetic solutions in real babashka execution."""

import shutil
import pytest

from src.codeflow.synthetic.verify import (
    extract_eval_blocks,
    ground_solution,
    verify_and_ground,
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
