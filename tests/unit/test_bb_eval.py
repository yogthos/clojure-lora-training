"""Tests for the babashka eval harness.

Skipped when babashka (bb) is not installed (e.g. CI runners).
"""

import shutil
import pytest

from src.codeflow.synthetic.bb_eval import eval_forms, bb_available

pytestmark = pytest.mark.skipif(
    shutil.which("bb") is None, reason="babashka not installed"
)


class TestEvalForms:
    def test_simple_value(self):
        r = eval_forms(["(+ 1 2)"])
        assert len(r) == 1
        assert r[0].ok is True
        assert r[0].value == "3"

    def test_state_persists_across_forms(self):
        r = eval_forms(["(def x 10)", "(* x x)"])
        assert [e.ok for e in r] == [True, True]
        assert r[1].value == "100"

    def test_def_returns_var(self):
        r = eval_forms(["(defn f [x] (* x x))", "(f 6)"])
        assert "f" in r[0].value and "#'" in r[0].value
        assert r[1].value == "36"

    def test_collection_value(self):
        r = eval_forms(["(map inc [1 2 3])"])
        assert r[0].value == "(2 3 4)"

    def test_error_is_caught_not_raised(self):
        r = eval_forms(["(/ 1 0)"])
        assert r[0].ok is False
        assert "Divide by zero" in r[0].error

    def test_error_does_not_abort_later_forms(self):
        r = eval_forms(["(/ 1 0)", "(+ 2 2)"])
        assert r[0].ok is False
        assert r[1].ok is True and r[1].value == "4"

    def test_stdout_captured_separately_from_value(self):
        r = eval_forms(['(do (println "hello") 42)'])
        assert r[0].ok is True
        assert r[0].value == "42"
        assert "hello" in r[0].stdout

    def test_empty_forms(self):
        assert eval_forms([]) == []

    def test_string_value_with_quotes(self):
        r = eval_forms(['(clojure.string/upper-case "abc")'])
        assert r[0].value == '"ABC"'

    def test_timeout_marks_not_ok(self):
        r = eval_forms(["(loop [] (recur))"], timeout=2.0)
        assert r[0].ok is False
        assert "timeout" in r[0].error.lower()


def test_bb_available_true_here():
    assert bb_available() is True
