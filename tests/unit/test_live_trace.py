"""The live trace loop: one generated step at a time against real results."""

import json
import shutil

import pytest

from src.codeflow.synthetic.jolt_eval import EvalResult, render_observation
from src.codeflow.synthetic.live_trace import EVAL_TOOL, run_live_trace
from src.llm.provider import LLMAuthError, LLMError

_DIFF = ("diff --git a/src/csv.clj b/src/csv.clj\n--- /dev/null\n+++ b/src/csv.clj\n"
         "@@ -0,0 +1,2 @@\n+(ns csv)\n+(defn valid-row? [r] (= 3 (count r)))\n")


def _eval(thinking, code):
    return json.dumps({"thinking": thinking, "action": "eval", "code": code})


def _apply(thinking, diff=_DIFF):
    return json.dumps({"thinking": thinking, "action": "apply", "diff": diff})


class _ScriptedLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def call(self, system_prompt, user_prompt, **kw):
        self.prompts.append(user_prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class _FakeSession:
    """Evaluates nothing; answers from a table, like a REPL would."""

    def __init__(self, answers=None, die_on=None):
        self.answers = answers or {}
        self.die_on = die_on
        self.alive = True
        self.closed = False

    def eval(self, code):
        if code == self.die_on:
            self.alive = False
            return EvalResult(form=code, ok=False, error="out of memory")
        value = self.answers.get(code, "nil")
        if isinstance(value, EvalResult):
            return value
        return EvalResult(form=code, value=value)

    def close(self):
        self.closed = True


def _run(llm, session=None, **kw):
    session = session or _FakeSession()
    return run_live_trace("add a CSV row validator", "a data library", llm,
                          session_factory=lambda **_: session, **kw), session


_HAPPY = [
    _eval("Goal: validate rows. File: src/csv.clj. Plan: valid-row?", "(defn valid-row? [r] (= 3 (count r)))"),
    _eval("Try a good row.", '(valid-row? ["a" "b" "c"])'),
    _eval("And nil.", "(valid-row? nil)"),
    _apply("All cases pass; write the file."),
]


def test_turns_alternate_with_real_results():
    session = _FakeSession({'(valid-row? ["a" "b" "c"])': "true", "(valid-row? nil)": "false"})
    trace, _ = _run(_ScriptedLLM(*_HAPPY), session)
    conv = trace.to_record()["conversations"]
    assert [t["from"] for t in conv] == (["human"] + ["function_call", "observation"] * 3 + ["gpt"])
    assert conv[0]["value"] == "add a CSV row validator\nProject: a data library"
    first = conv[1]["value"]
    assert first.startswith("<think>\nGoal: validate rows.")
    think, call = first.split("\n</think>\n\n")
    assert json.loads(call) == {"name": "eval",
                                "arguments": {"code": "(defn valid-row? [r] (= 3 (count r)))"}}
    assert conv[4]["value"] == "=> true"
    assert conv[-1]["value"] == "<think>\nAll cases pass; write the file.\n</think>\n\n;; apply:\n" + _DIFF


def test_record_fields():
    trace, _ = _run(_ScriptedLLM(*_HAPPY))
    rec = trace.to_record()
    assert rec["instruction"] == "add a CSV row validator"
    assert rec["input"] == "Project: a data library"
    assert rec["output"] == rec["conversations"][-1]["value"]
    assert json.loads(rec["tools"]) == [EVAL_TOOL]


def test_generator_sees_the_real_result():
    session = _FakeSession({"(defn valid-row? [r] (= 3 (count r)))": EvalResult(
        form="", ok=False, error="Syntax error compiling at (REPL:1:1).\nboom")})
    llm = _ScriptedLLM(*_HAPPY)
    _run(llm, session)
    assert "Syntax error compiling at (REPL:1:1).\nboom" in llm.prompts[1]


def test_session_is_closed():
    _, session = _run(_ScriptedLLM(*_HAPPY))
    assert session.closed


@pytest.mark.parametrize("replies", [
    [_eval("t", "(+ 1 2)"), _apply("too soon")],                       # < 3 evals
    [_eval("t", "(+ 1 2)")] * 5,                                        # never applies
    [_eval("t", "(+ 1 2)"), _eval("", "(+ 1 2)")] + _HAPPY[2:],         # empty thinking
    _HAPPY[:3] + [_apply("bad", "no diff here")],                       # not a patch
    _HAPPY[:3] + [json.dumps({"thinking": "t", "action": "deploy"})] * 2,  # unknown action, retried
])
def test_bad_traces_are_dropped(replies):
    trace, session = _run(_ScriptedLLM(*replies), max_steps=5)
    assert trace is None and session.closed


def test_malformed_step_gets_one_retry():
    trace, _ = _run(_ScriptedLLM("not json", *_HAPPY))
    assert trace is not None
    trace, _ = _run(_ScriptedLLM("not json", "still not", *_HAPPY))
    assert trace is None


def test_lost_session_drops_the_trace():
    session = _FakeSession(die_on='(valid-row? ["a" "b" "c"])')
    trace, _ = _run(_ScriptedLLM(*_HAPPY), session)
    assert trace is None


def test_hunk_counts_are_repaired():
    bad = _DIFF.replace("@@ -0,0 +1,2 @@", "@@ -0,0 +1,7 @@")
    trace, _ = _run(_ScriptedLLM(*_HAPPY[:3], _apply("done", bad)))
    assert "@@ -0,0 +1,2 @@" in trace.to_record()["output"]


def test_llm_failure_drops_auth_failure_raises():
    trace, _ = _run(_ScriptedLLM(_HAPPY[0], LLMError("503")))
    assert trace is None
    with pytest.raises(LLMAuthError):
        _run(_ScriptedLLM(LLMAuthError("401")))


def test_observation_text_is_the_shared_rendering():
    r = EvalResult(form="x", value="42", stdout="hi\n")
    session = _FakeSession({"(defn valid-row? [r] (= 3 (count r)))": r})
    trace, _ = _run(_ScriptedLLM(*_HAPPY), session)
    assert trace.to_record()["conversations"][2]["value"] == render_observation(r)


@pytest.mark.skipif(shutil.which("jolt") is None, reason="jolt not installed")
def test_against_real_jolt():
    trace = run_live_trace("add a CSV row validator", "", _ScriptedLLM(*_HAPPY))
    conv = trace.to_record()["conversations"]
    assert conv[2]["value"] == "=> #'user/valid-row?"
    assert conv[4]["value"] == "=> true"
    assert conv[6]["value"] == "=> false"


@pytest.mark.parametrize("code,expected", [
    ("(defn sq [x] (* x x))", True),
    ("(require '[clojure.string :as str])", True),
    ("  (def data [1 2 3])", True),
    ("(sq 5)", False),
    ("(defn sq [x] (* x x))\n(sq 5)", False),        # defines, then demonstrates
    ("(def a 1) (def b 2)", True),
    ('(println "(defn x)")', False),
])
def test_is_definition(code, expected):
    from src.codeflow.synthetic.live_trace import is_definition
    assert is_definition(code) is expected


def test_raw_newlines_in_json_strings_are_accepted():
    # DeepSeek's JSON mode sometimes emits literal newlines inside strings.
    raw = '{"thinking": "line one\nline two", "action": "eval", "code": "(+ 1\n 2)"}'
    steps = [raw] + _HAPPY[1:]
    trace, _ = _run(_ScriptedLLM(*steps))
    assert trace is not None and trace.steps[0].code == "(+ 1\n 2)"


def test_drop_reasons_are_counted():
    from src.codeflow.synthetic.live_trace import drop_counts
    before = drop_counts().get("applied before 3 evals", 0)
    _run(_ScriptedLLM(_eval("t", "(+ 1 2)"), _apply("too soon")))
    assert drop_counts()["applied before 3 evals"] == before + 1


def test_only_the_first_of_several_steps_is_taken():
    two = _eval("Plan.", "(defn valid-row? [r] (= 3 (count r)))") + "\n\n" + _eval("next", "(boom)")
    trace, _ = _run(_ScriptedLLM(two, *_HAPPY[1:]))
    assert trace is not None
    assert [s.code for s in trace.steps][0] == "(defn valid-row? [r] (= 3 (count r)))"
    assert "(boom)" not in [s.code for s in trace.steps]


@pytest.mark.parametrize("leak", [
    "Per the instructions I should create the file.",
    "I've evaluated 6 times, well past the 3-eval minimum.",
    "The session is empty and no files are shown, so I start fresh.",
    "As the rules say, keep each eval small.",
])
def test_reasoning_about_the_generator_prompt_is_retried(leak):
    leaky = _eval(leak, "(defn valid-row? [r] (= 3 (count r)))")
    trace, _ = _run(_ScriptedLLM(leaky, *_HAPPY))
    assert trace is not None and leak not in trace.steps[0].thinking
    trace, _ = _run(_ScriptedLLM(leaky, leaky))
    assert trace is None


def test_retry_says_what_was_wrong():
    leaky = _eval("Per the instructions I start here.", "(defn valid-row? [r] (= 3 (count r)))")
    llm = _ScriptedLLM(leaky, *_HAPPY)
    _run(llm)
    assert "rejected" not in llm.prompts[0]
    assert "rejected" in llm.prompts[1] and "instructions" in llm.prompts[1]
    assert "rejected" not in llm.prompts[2]


def test_first_step_prompt_does_not_call_the_session_empty():
    llm = _ScriptedLLM(*_HAPPY)
    _run(llm)
    assert "empty" not in llm.prompts[0].lower()


@pytest.mark.skipif(shutil.which("jolt") is None, reason="jolt not installed")
class TestNewFilesLoad:
    def _trace(self, diff):
        from src.codeflow.synthetic.live_trace import LiveTrace
        return LiveTrace(request="r", context="", diff=diff)

    def _diff(self, *files):
        out = ""
        for path, body in files:
            lines = body.splitlines()
            out += (f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n"
                    f"@@ -0,0 +1,{len(lines)} @@\n" + "".join(f"+{l}\n" for l in lines))
        return out

    def test_test_file_requiring_its_source_loads(self):
        from src.codeflow.synthetic.jolt_eval import JoltSession
        from src.codeflow.synthetic.live_trace import _new_files_load
        diff = self._diff(("src/m/core.clj", "(ns m.core)\n(defn sq [x] (* x x))"),
                          ("test/m/core_test.clj",
                           "(ns m.core-test\n  (:require [clojure.test :refer [deftest is]]\n"
                           "            [m.core :refer [sq]]))\n(deftest t (is (= 4 (sq 2))))"))
        assert _new_files_load(self._trace(diff), JoltSession)

    def test_broken_ns_form_fails(self):
        from src.codeflow.synthetic.jolt_eval import JoltSession, eval_forms
        from src.codeflow.synthetic.live_trace import _new_files_load
        if eval_forms(["(nope 1)"])[0].ok or "unbound" in eval_forms(["(nope 1)"])[0].error.lower():
            pytest.xfail("jolt without unresolved-symbol errors (jolt-lang/jolt#1323)")
        diff = self._diff(("src/db/jdbc.clj",
                           '(ns db.jdbc\n  "Doc.")\n  (:require [clojure.string :as str]))\n(defn f [] 1)'))
        assert not _new_files_load(self._trace(diff), JoltSession)

    def test_namespace_that_does_not_match_its_path_fails(self):
        from src.codeflow.synthetic.jolt_eval import JoltSession
        from src.codeflow.synthetic.live_trace import _new_files_load
        diff = self._diff(("src/csv.clj", "(ns app.csv)\n(defn f [] 1)"))
        assert not _new_files_load(self._trace(diff), JoltSession)
