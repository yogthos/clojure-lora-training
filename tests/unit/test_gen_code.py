"""Tests for EpiCoder task solutions generated through the live trace loop."""

import json

import pytest

from src.codeflow.synthetic.gen_code import CodeGenResult, generate_training_examples
from src.codeflow.synthetic.jolt_eval import EvalResult
from src.codeflow.synthetic.live_trace import accept_trace, run_live_trace
from src.llm.provider import LLMAuthError, LLMError

_DIFF = ("diff --git a/src/sq.clj b/src/sq.clj\n--- /dev/null\n+++ b/src/sq.clj\n"
         "@@ -0,0 +1,2 @@\n+(ns sq)\n+(defn sq [x] (* x x))\n")
_CODE = ["(defn sq [x] (* x x))", "(sq 5)", "(sq -2)"]


class _StepLLM:
    def call(self, system_prompt, user_prompt, **kw):
        n = user_prompt.count("--- step ")
        if n < len(_CODE):
            return json.dumps({"thinking": f"step {n}", "action": "eval", "code": _CODE[n]})
        return json.dumps({"thinking": "done", "action": "apply", "diff": _DIFF})


class _Session:
    def __init__(self, failing=()):
        self.failing = failing
        self.alive = True

    def eval(self, code):
        if code in self.failing:
            return EvalResult(form=code, ok=False, error="boom")
        return EvalResult(form=code, value="25")

    def close(self):
        pass


def _factory(failing=()):
    return lambda **kw: _Session(failing)


class TestGenerateTrainingExamples:
    def test_task_becomes_a_multi_turn_example(self):
        out = generate_training_examples([{"instruction": "square a number", "feature_used": "defn"}],
                                         _StepLLM(), session_factory=_factory())
        assert len(out) == 1
        ex = out[0].to_training_example()
        assert ex["instruction"] == "square a number"
        assert ex["input"] == ""
        assert ex["conversations"][1]["from"] == "function_call"
        assert out[0].feature_used == "defn"

    def test_failing_end_state_is_dropped(self):
        out = generate_training_examples([{"instruction": "square"}], _StepLLM(),
                                         session_factory=_factory(failing=("(sq -2)",)))
        assert out == []

    def test_blank_instruction_skipped(self):
        assert generate_training_examples([{"instruction": " "}], _StepLLM(),
                                          session_factory=_factory()) == []

    def test_max_examples(self):
        tasks = [{"instruction": f"t{i}"} for i in range(5)]
        assert len(generate_training_examples(tasks, _StepLLM(), max_examples=2,
                                              session_factory=_factory())) == 2

    def test_llm_errors_drop_auth_errors_raise(self):
        class _Fail:
            def __init__(self, exc):
                self.exc = exc

            def call(self, *a, **k):
                raise self.exc

        assert generate_training_examples([{"instruction": "x"}], _Fail(LLMError("503")),
                                          session_factory=_factory()) == []
        with pytest.raises(LLMAuthError):
            generate_training_examples([{"instruction": "x"}], _Fail(LLMAuthError("401")),
                                       session_factory=_factory())


class TestAcceptTrace:
    def test_pass_rate_and_end_state(self):
        trace = run_live_trace("square", "", _StepLLM(), session_factory=_factory(("(sq 5)",)))
        assert accept_trace(trace, _factory(), min_pass_rate=0.6) == pytest.approx(2 / 3)
        assert accept_trace(trace, _factory(), min_pass_rate=0.9) is None


def test_codegen_result_jsonl():
    out = generate_training_examples([{"instruction": "square"}], _StepLLM(),
                                     session_factory=_factory())
    assert isinstance(out[0], CodeGenResult)
    assert json.loads(out[0].to_jsonl()) == out[0].to_training_example()
