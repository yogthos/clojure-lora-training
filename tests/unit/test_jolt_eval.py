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


class TestServerEnv:
    def test_keeps_path_and_jolt_settings_drops_secrets(self, monkeypatch):
        from src.codeflow.synthetic.jolt_eval import _server_env
        monkeypatch.setenv("PATH", "/opt/homebrew/bin:/usr/bin")
        monkeypatch.setenv("JOLT_CHEZ", "/opt/homebrew/bin/chez")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret")
        env = _server_env("/x/jolt/bin/jolt", "/w/tmp")
        assert env["PATH"].split(":")[:2] == ["/x/jolt/bin", "/opt/homebrew/bin"]
        assert env["JOLT_CHEZ"] == "/opt/homebrew/bin/chez"
        assert env["TMPDIR"] == "/w/tmp"
        assert "DEEPSEEK_API_KEY" not in env


    def test_stack_limit_follows_memory_cap(self, monkeypatch):
        from src.codeflow.synthetic.jolt_eval import _server_env
        monkeypatch.delenv("JOLT_STACK_LIMIT", raising=False)
        assert _server_env("/j/bin/jolt", "/t", max_rss_mb=2048)["JOLT_STACK_LIMIT"] == "512m"
        assert "JOLT_STACK_LIMIT" not in _server_env("/j/bin/jolt", "/t", max_rss_mb=None)

    def test_explicit_stack_limit_wins(self, monkeypatch):
        from src.codeflow.synthetic.jolt_eval import _server_env
        monkeypatch.setenv("JOLT_STACK_LIMIT", "off")
        assert _server_env("/j/bin/jolt", "/t", max_rss_mb=2048)["JOLT_STACK_LIMIT"] == "off"


class TestSandboxProfile:
    def test_jolt_install_root_is_readable(self, tmp_path):
        # A source checkout under $HOME runs from its own tree (bin/jolt
        # loads host/chez/*.ss), so the sandbox must let it read that tree.
        from src.codeflow.synthetic.jolt_eval import _sandbox_profile
        root = tmp_path / "jolt"
        (root / "bin").mkdir(parents=True)
        (root / "bin" / "jolt").write_text("")
        home = tmp_path.resolve()
        profile = _sandbox_profile("/w", str(home), str(root / "bin" / "jolt"))
        assert f'(subpath "{root.resolve()}")' in profile

    def test_dirs_between_home_and_root_are_listable_not_open(self, tmp_path):
        from src.codeflow.synthetic.jolt_eval import _sandbox_profile
        home = tmp_path.resolve()
        root = home / "src" / "jolt"
        (root / "bin").mkdir(parents=True)
        (root / "bin" / "jolt").write_text("")
        profile = _sandbox_profile("/w", str(home), str(root / "bin" / "jolt"))
        assert f'(literal "{home / "src"}")' in profile
        assert f'(subpath "{home / "src"}")' not in profile

    def test_root_containing_home_is_not_opened(self, tmp_path):
        from src.codeflow.synthetic.jolt_eval import _sandbox_profile
        home = tmp_path.resolve()
        (home / "bin").mkdir()
        (home / "bin" / "jolt").write_text("")
        profile = _sandbox_profile("/w", str(home), str(home / "bin" / "jolt"))
        assert f'(subpath "{home}")' not in profile.split("(allow file-read*")[1]


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

    def test_runaway_recursion_overflows_the_stack(self):
        rs = eval_forms(["(defn f [x] (inc (f x)))", "(f 1)", "(+ 1 1)"],
                        timeout=30.0, max_rss_mb=600)
        if "out of memory" in rs[1].error:
            pytest.xfail("jolt without JOLT_STACK_LIMIT support")
        assert not rs[1].ok
        assert "StackOverflowError" in rs[1].error
        assert rs[2].ok and rs[2].value == "2"  # the session survives

    def test_runaway_allocation_is_killed_by_memory_cap(self):
        import time
        t = time.monotonic()
        rs = eval_forms(["(def v (atom []))",
                         "(doseq [i (range)] (swap! v conj (str i)))", "(+ 1 1)"],
                        timeout=30.0, max_rss_mb=600)
        assert time.monotonic() - t < 25
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

    def test_jolt_settings_reach_the_server(self, monkeypatch):
        monkeypatch.setenv("JOLT_STACK_LIMIT", "64m")
        rs = eval_forms(['(System/getenv "JOLT_STACK_LIMIT")'])
        assert rs[0].value == '"64m"'

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


class TestRenderObservation:
    def test_value(self):
        from src.codeflow.synthetic.jolt_eval import EvalResult, render_observation
        assert render_observation(EvalResult(form="(+ 1 2)", value="3")) == "=> 3"

    def test_output_then_value(self):
        from src.codeflow.synthetic.jolt_eval import EvalResult, render_observation
        r = EvalResult(form="x", value="42", stdout="hi\nthere\n")
        assert render_observation(r) == "hi\nthere\n=> 42"

    def test_error_after_output(self):
        from src.codeflow.synthetic.jolt_eval import EvalResult, render_observation
        r = EvalResult(form="x", stdout="partial\n", ok=False,
                       error="Execution error (ArithmeticException) at user/eval1 (REPL:1).\nDivide by zero")
        assert render_observation(r) == ("partial\nExecution error (ArithmeticException) at "
                                         "user/eval1 (REPL:1).\nDivide by zero")


@needs_jolt
class TestJoltSession:
    def test_state_persists_between_evals(self):
        from src.codeflow.synthetic.jolt_eval import JoltSession
        with JoltSession() as s:
            assert s.eval("(def x 41)").ok
            assert s.eval("(inc x)").value == "42"

    def test_eval_after_lost_session_is_not_run(self):
        from src.codeflow.synthetic.jolt_eval import JoltSession
        with JoltSession(max_rss_mb=400) as s:
            s.eval('(def v (atom []))')
            r = s.eval('(doseq [i (range)] (swap! v conj (str i)))')
            assert "memory" in r.error
            assert not s.alive
            assert s.eval("(+ 1 1)").error == "not run: session lost"


@needs_jolt
class TestSessionFiles:
    def test_files_are_on_the_source_path(self):
        from src.codeflow.synthetic.jolt_eval import JoltSession
        files = {"src/a/core.clj": "(ns a.core)\n(defn f [] 42)\n",
                 "test/a/core_test.clj": "(ns a.core-test (:require [a.core :refer [f]]))\n(def r (f))\n"}
        with JoltSession(files=files) as s:
            assert s.eval("(require 'a.core-test)").ok
            assert s.eval("a.core-test/r").value == "42"

    def test_paths_outside_the_workdir_are_refused(self):
        from src.codeflow.synthetic.jolt_eval import JoltSession
        for bad in ("../x.clj", "/tmp/x.clj"):
            with pytest.raises(ValueError):
                JoltSession(files={bad: "(ns x)"})
