"""Tests for grounding REPL forms in a sandboxed Jolt nREPL session."""

import sys

import pytest

from src.codeflow.synthetic.nrepl_client import bdecode_stream, bencode
from src.codeflow.synthetic.jolt_eval import eval_forms, jolt_available

_JOLT = jolt_available()
needs_jolt = pytest.mark.skipif(not _JOLT, reason="jolt not installed")


class TestBencode:
    def test_round_trip(self):
        import io
        msg = {"op": "eval", "code": "(+ 1 2) ;; ünïcode", "n": 3, "l": ["a", 1]}
        out = list(bdecode_stream(io.BytesIO(bencode(msg))))
        assert out == [msg]

    def test_keys_sorted_and_utf8_lengths(self):
        assert bencode({"b": 1, "a": "é"}) == b"d1:a2:\xc3\xa91:bi1ee"


@needs_jolt
class TestEvalForms:
    def test_values_and_output(self):
        rs = eval_forms(['(defn sq [x] (* x x))', '(sq 5)', '(println "hi") 42'])
        assert [r.value for r in rs] == ["#'user/sq", "25", "42"]
        assert all(r.ok for r in rs)
        assert rs[2].stdout == "hi\n"

    def test_namespace_persists_like_a_real_session(self):
        rs = eval_forms(["(ns my.app)", "(defn g [] 1)", "(str *ns*)"])
        assert rs[1].value == "#'my.app/g"
        assert rs[2].value == '"my.app"'

    def test_error_is_reported_and_session_continues(self):
        rs = eval_forms(["(/ 1 0)", "(+ 1 1)"])
        assert not rs[0].ok
        assert "Divide by zero" in rs[0].error
        assert rs[1].ok and rs[1].value == "2"

    def test_functions_print_jvm_style(self):
        rs = eval_forms(["(fn [x] x)"])
        assert rs[0].value.startswith("#object[")
        assert "sci." not in rs[0].value

    def test_runaway_form_times_out_and_later_forms_still_run(self):
        rs = eval_forms(["(loop [] (recur))", "(+ 2 3)"], timeout=2.0)
        assert not rs[0].ok
        assert "timed out" in rs[0].error.lower()
        assert rs[1].ok and rs[1].value == "5"

    def test_runaway_recursion_is_killed_by_memory_cap(self):
        import time
        t = time.monotonic()
        rs = eval_forms(["(defn f [x] (inc (f x)))", "(f 1)", "(+ 1 1)"],
                        timeout=30.0, max_rss_mb=600)
        assert time.monotonic() - t < 20
        assert not rs[1].ok
        assert "memory" in rs[1].error
        assert not rs[2].ok  # the server is gone; nothing after it ran

    def test_infinite_seq_value_is_print_bounded(self):
        rs = eval_forms(["(range)"], timeout=5.0)
        assert rs[0].ok
        assert rs[0].value.endswith("...)")

    def test_traces_are_isolated(self):
        eval_forms(["(def leaked 1)"])
        rs = eval_forms(["leaked"])
        assert rs[0].value != "1"

    def test_unresolved_symbol_is_an_error(self):
        # JVM nREPL: "Unable to resolve symbol: nope in this context". Jolt
        # <= 0.8.21 interns an unbound var instead (fix in progress upstream).
        rs = eval_forms(["nope"])
        if rs[0].ok and "Unbound" in rs[0].value:
            pytest.xfail("jolt evaluate allows unresolved vars")
        assert not rs[0].ok
        assert "Unable to resolve symbol" in rs[0].error


@needs_jolt
@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS-only")
class TestSandbox:
    def test_api_keys_are_not_visible(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret")
        rs = eval_forms(['(System/getenv "DEEPSEEK_API_KEY")'])
        assert rs[0].value == "nil"

    def test_home_is_unreadable(self):
        rs = eval_forms(['(slurp (str (System/getenv "HOME") "/.ssh/known_hosts"))'])
        assert not rs[0].ok

    def test_writes_outside_workdir_are_denied(self, tmp_path):
        target = tmp_path / "pwned.txt"
        rs = eval_forms([f'(spit "{target}" "x")'])
        assert not rs[0].ok
        assert not target.exists()

    def test_workdir_is_writable(self):
        rs = eval_forms(['(spit "scratch.txt" "ok")', '(slurp "scratch.txt")'])
        assert rs[1].value == '"ok"'

    def test_network_is_denied(self):
        rs = eval_forms(['(java.net.Socket. "1.1.1.1" 80)'], timeout=5.0)
        assert not rs[0].ok
