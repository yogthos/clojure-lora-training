"""Lexical scanning of Clojure text."""

from src.codeflow.clj_text import top_level_forms


def test_top_level_forms():
    code = '(def a "(")\n;; (not a form)\n(f a) [1 2] :kw'
    assert top_level_forms(code) == ['(def a "(")', "(f a)", "[1 2]", ":kw"]
