"""Tests for workflow generation from mined prompts via the live trace loop."""

import json

import pytest

from src.codeflow.synthetic.jolt_eval import EvalResult
from src.codeflow.synthetic.prompt_mining import MinedPrompt
from src.codeflow.synthetic.workflow_gen import (
    WorkflowResult,
    generate_one,
    generate_workflows,
    generate_workflows_concurrent,
)
from src.llm.provider import LLMAuthError, LLMError

_DIFF = ("diff --git a/src/csv.clj b/src/csv.clj\n--- /dev/null\n+++ b/src/csv.clj\n"
         "@@ -0,0 +1,2 @@\n+(ns csv)\n+(defn valid-row? [r] (= 3 (count r)))\n")

_CODE = ["(defn valid-row? [r] (= 3 (count r)))", '(valid-row? ["a" "b" "c"])', "(valid-row? nil)"]


class _StepLLM:
    """Stateless: picks the next step from how many the transcript holds."""

    def __init__(self, codes=_CODE, diff=_DIFF):
        self.codes = codes
        self.diff = diff

    def call(self, system_prompt, user_prompt, **kw):
        n = user_prompt.count("--- step ")
        if n < len(self.codes):
            return json.dumps({"thinking": f"step {n}", "action": "eval", "code": self.codes[n]})
        return json.dumps({"thinking": "done", "action": "apply", "diff": self.diff})


class _FakeSession:
    def __init__(self, answers=None):
        self.answers = answers or {}
        self.alive = True

    def eval(self, code):
        a = self.answers.get(code, "true")
        return a if isinstance(a, EvalResult) else EvalResult(form=code, value=a)

    def close(self):
        pass


def _factory(answers=None):
    return lambda **kw: _FakeSession(answers)


_PROMPT = MinedPrompt(
    user_prompt="add a CSV row validator",
    project_context="a Clojure data library",
    source_instruction="add valid-row? to csv ns",
    source_repo="o__lib",
)


def _prompt(i):
    return MinedPrompt(user_prompt=f"task {i}", project_context="lib", source_instruction=f"c{i}")


def _err(code):
    return EvalResult(form=code, ok=False, error="Execution error (NullPointerException)")


class TestGenerateOne:
    def test_produces_a_multi_turn_record(self):
        r = generate_one(_PROMPT, _StepLLM(), session_factory=_factory())
        rec = r.to_training_example()
        assert [t["from"] for t in rec["conversations"]][:3] == ["human", "function_call", "observation"]
        assert rec["repo"] == "o__lib"
        assert rec["system"]

    def test_recovered_failure_is_kept(self):
        answers = {_CODE[1]: _err(_CODE[1])}
        r = generate_one(_PROMPT, _StepLLM(), session_factory=_factory(answers))
        assert r is not None and r.pass_rate == pytest.approx(2 / 3)

    def test_failing_final_check_is_dropped(self):
        # The last non-definition eval is the end state; it must run cleanly.
        answers = {_CODE[2]: _err(_CODE[2])}
        assert generate_one(_PROMPT, _StepLLM(), session_factory=_factory(answers)) is None

    def test_low_pass_rate_is_dropped(self):
        codes = ["(f)", "(f)", "(f)", "(g)"]
        answers = {"(f)": _err("(f)")}
        r = generate_one(_PROMPT, _StepLLM(codes=codes), session_factory=_factory(answers),
                         min_pass_rate=0.6)
        assert r is None

    def test_new_file_that_does_not_load_is_dropped(self):
        # The diff's file is loaded in a fresh session; a diff that doesn't
        # match the code that ran is caught here.
        answers = {"(require 'csv)": _err("")}
        assert generate_one(_PROMPT, _StepLLM(), session_factory=_factory(answers)) is None

    def test_llm_errors_drop_auth_errors_raise(self):
        class _Fail:
            def __init__(self, exc):
                self.exc = exc

            def call(self, *a, **k):
                raise self.exc

        assert generate_one(_PROMPT, _Fail(LLMError("busy")), session_factory=_factory()) is None
        with pytest.raises(LLMAuthError):
            generate_one(_PROMPT, _Fail(LLMAuthError("401")), session_factory=_factory())


class TestWorkflowResult:
    def test_jsonl_round_trips(self):
        r = generate_one(_PROMPT, _StepLLM(), session_factory=_factory())
        assert json.loads(r.to_jsonl()) == r.to_training_example()
        assert isinstance(r, WorkflowResult)


class TestConcurrentGeneration:
    def test_same_kept_set_as_serial(self):
        prompts = [_prompt(i) for i in range(12)]
        serial = generate_workflows(prompts, _StepLLM(), max_examples=12, session_factory=_factory())
        conc = generate_workflows_concurrent(prompts, _StepLLM(), max_workers=4,
                                             session_factory=_factory())
        assert len(conc) == len(serial) == 12
        assert {r.trace.request for r in conc} == {r.trace.request for r in serial}

    def test_callbacks(self):
        prompts = [_prompt(i) for i in range(10)]
        streamed, progress, attempted = [], [], []
        generate_workflows_concurrent(
            prompts, _StepLLM(), max_workers=4, session_factory=_factory(),
            on_result=streamed.append,
            on_progress=lambda done, total, kept: progress.append((done, total, kept)),
            on_attempt=lambda p, r: attempted.append(p),
        )
        assert len(streamed) == 10 and len(attempted) == 10
        assert progress[-1] == (10, 10, 10)

    def test_worker_exception_is_dropped_not_fatal(self):
        class _Boom:
            def call(self, *a, **k):
                raise RuntimeError("boom")
        out = generate_workflows_concurrent([_prompt(i) for i in range(5)], _Boom(),
                                            max_workers=3, session_factory=_factory())
        assert out == []

    def test_auth_error_stops_the_run(self):
        class _NoAuth:
            def call(self, *a, **k):
                raise LLMAuthError("401")
        with pytest.raises(LLMAuthError):
            generate_workflows_concurrent([_prompt(i) for i in range(4)], _NoAuth(),
                                          max_workers=2, session_factory=_factory())
