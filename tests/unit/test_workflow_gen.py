"""Tests for the plan-first, iterative-REPL workflow generation."""

import json
import pytest

from src.codeflow.synthetic.prompt_mining import MinedPrompt
from src.codeflow.synthetic.workflow_gen import (
    WorkflowResult,
    generate_plan,
    generate_workflow,
    generate_one,
    generate_workflows,
    generate_workflows_concurrent,
    _fallback_plan,
)


class _FakeLLM:
    """Returns a JSON plan for require_json calls, a fixed trace otherwise."""

    def __init__(self, plan=None, trace=""):
        self._plan = plan or {}
        self._trace = trace

    def call(self, system_prompt, user_prompt, temperature=None,
             max_tokens=None, require_json=False, reject_truncated=False):
        if require_json:
            return json.dumps(self._plan)
        return self._trace


_PROMPT = MinedPrompt(
    user_prompt="add a CSV row validator",
    project_context="a Clojure data library",
    source_instruction="add valid-row? to csv ns",
)


class TestGeneratePlan:
    def test_returns_goal_files_steps(self):
        plan = {
            "goal": "validate CSV rows have exactly 3 non-blank fields",
            "files": [{"path": "src/csv.clj", "purpose": "validation"}],
            "steps": [
                {"name": "parse-line", "purpose": "split", "depends_on": []},
                {"name": "valid-row?", "purpose": "check", "depends_on": ["parse-line"]},
            ],
        }
        result = generate_plan(_PROMPT, _FakeLLM(plan=plan))
        assert result["goal"].startswith("validate")
        assert result["files"][0]["path"] == "src/csv.clj"
        assert [s["name"] for s in result["steps"]] == ["parse-line", "valid-row?"]

    def test_falls_back_on_bad_json(self):
        class Bad:
            def call(self, *a, **k):
                return "not json"
        result = generate_plan(_PROMPT, Bad())
        assert "goal" in result and "steps" in result  # fallback shape


class TestGenerateWorkflow:
    def test_passes_through_trace(self):
        trace = (
            ";; Goal: validate rows\n;; Files:\n;;   - src/csv.clj — x\n"
            ";; Plan (build order):\n;;   1. valid-row? — check\n"
            ";; nREPL session:\n;; --- Step 1: valid-row? ---\n"
            ";; eval: (defn valid-row? [r] (= 3 (count r)))\n;; result: ?\n"
            ";; apply:\ndiff --git a/src/csv.clj b/src/csv.clj\n"
        )
        out = generate_workflow(_PROMPT, _fallback_plan(_PROMPT), _FakeLLM(trace=trace))
        assert ";; Goal:" in out
        assert ";; Plan" in out
        assert "diff --git" in out


class TestWorkflowResult:
    def test_training_example_shape(self):
        r = WorkflowResult(
            user_prompt="add a CSV row validator",
            project_context="a Clojure data library",
            plan={},
            solution=";; Goal: x\n;; apply:\ndiff --git a/x b/x\n",
        )
        ex = r.to_training_example()
        # Input is the user request; planning is in the OUTPUT, not the input.
        assert ex["instruction"] == "add a CSV row validator"
        assert "Clojure data library" in ex["input"]
        assert ";; Goal:" in ex["output"]
        # The training system prompt frames the agent workflow, not a fixed plan.
        assert "plan" in ex["system"].lower()
        assert ";; Goal:" not in ex["instruction"]  # model must produce the plan


_DIFF = (
    "diff --git a/src/x.clj b/src/x.clj\n"
    "--- a/src/x.clj\n"
    "+++ b/src/x.clj\n"
    "@@ -1,2 +1,2 @@\n"
    " (ns x)\n"
    "-(defn f [] 1)\n"
    "+(defn f [] 3)\n"
)

_TRACE = (
    ";; Goal: validate\n"
    ";; nREPL session:\n;; eval: (+ 1 2)\n;; result: ?\n"
    ";; apply:\n" + _DIFF
)


def _prompt(i):
    return MinedPrompt(
        user_prompt=f"task {i}", project_context="lib", source_instruction=f"c{i}"
    )


class TestGenerateOne:
    def test_returns_result_unverified(self):
        llm = _FakeLLM(plan={"goal": "g", "steps": [{"name": "s"}]}, trace=_TRACE)
        r = generate_one(_PROMPT, llm, verify=False)
        assert r is not None
        assert r.user_prompt == _PROMPT.user_prompt
        assert ";; Goal:" in r.solution

    def test_drops_empty_solution(self):
        llm = _FakeLLM(plan={"goal": "g", "steps": [{"name": "s"}]}, trace="   ")
        assert generate_one(_PROMPT, llm, verify=False) is None


class TestGenerateOneRejectsBadTraces:
    _PLAN = {"goal": "g", "steps": [{"name": "s"}]}

    def test_truncated_generation_is_dropped(self):
        from src.llm.provider import LLMTruncatedError

        class _Truncating(_FakeLLM):
            def call(self, *a, reject_truncated=False, require_json=False, **k):
                if require_json:
                    return json.dumps(self._plan)
                assert reject_truncated, "workflow call must ask to reject truncation"
                raise LLMTruncatedError("finish_reason=length")

        assert generate_one(_PROMPT, _Truncating(plan=self._PLAN), verify=False) is None

    def test_trace_without_apply_marker_is_dropped(self):
        trace = _TRACE.replace(";; apply:\n", "")
        assert generate_one(_PROMPT, _FakeLLM(plan=self._PLAN, trace=trace)) is None

    def test_trace_without_diff_is_dropped(self):
        trace = _TRACE.split(";; apply:")[0] + ";; apply:\n"
        assert generate_one(_PROMPT, _FakeLLM(plan=self._PLAN, trace=trace)) is None

    def test_repetition_loop_is_dropped(self):
        loop = ";; eval: (require " + "'zip/xml-zip 'zip/seq-zip " * 40 + ")\n"
        trace = _TRACE.replace(";; apply:", loop + ";; apply:")
        assert generate_one(_PROMPT, _FakeLLM(plan=self._PLAN, trace=trace)) is None

    def test_repetitive_results_and_padding_are_not_loops(self):
        row = '{:name "Alice", :total 100} '
        trace = _TRACE.replace(
            ";; result: ?",
            ";; result: (" + row * 12 + ")\n;; eval: (apply str (repeat 50 \\x))\n"
            ";; result: \"" + "x" * 50 + "\"\n;;" + " " * 120 + "aligned",
        )
        assert generate_one(_PROMPT, _FakeLLM(plan=self._PLAN, trace=trace)) is not None

    def test_wrong_hunk_counts_are_repaired(self):
        trace = _TRACE.replace("@@ -1,2 +1,2 @@", "@@ -1,7 +1,30 @@")
        r = generate_one(_PROMPT, _FakeLLM(plan=self._PLAN, trace=trace))
        assert r is not None
        assert "@@ -1,2 +1,2 @@" in r.solution

    def test_unrepairable_diff_is_dropped(self):
        trace = _TRACE + "@@ -1,0 +"
        assert generate_one(_PROMPT, _FakeLLM(plan=self._PLAN, trace=trace)) is None


class TestConcurrentGeneration:
    def test_same_kept_set_as_serial(self):
        prompts = [_prompt(i) for i in range(12)]
        llm = _FakeLLM(plan={"goal": "g", "steps": [{"name": "s"}]}, trace=_TRACE)
        serial = generate_workflows(prompts, llm, max_examples=12, verify=False)
        conc = generate_workflows_concurrent(prompts, llm, max_workers=4, verify=False)
        assert len(conc) == len(serial) == 12
        assert {r.user_prompt for r in conc} == {r.user_prompt for r in serial}

    def test_on_result_fires_per_kept_and_on_progress_per_prompt(self):
        prompts = [_prompt(i) for i in range(10)]
        llm = _FakeLLM(plan={"goal": "g", "steps": [{"name": "s"}]}, trace=_TRACE)
        streamed = []
        progress = []
        generate_workflows_concurrent(
            prompts, llm, max_workers=4, verify=False,
            on_result=lambda r: streamed.append(r),
            on_progress=lambda done, total, kept: progress.append((done, total, kept)),
        )
        assert len(streamed) == 10
        assert len(progress) == 10            # one tick per completed prompt
        assert progress[-1][0] == 10 and progress[-1][1] == 10  # done/total
        assert progress[-1][2] == 10          # final kept count

    def test_worker_exception_is_dropped_not_fatal(self):
        class _BoomLLM(_FakeLLM):
            def call(self, *a, **k):
                raise RuntimeError("boom")
        prompts = [_prompt(i) for i in range(5)]
        # Must not raise; all samples drop to None.
        out = generate_workflows_concurrent(prompts, _BoomLLM(), max_workers=3)
        assert out == []


class TestLLMFailures:
    def test_llm_error_drops_sample(self):
        from src.llm.provider import LLMError

        class _Fail(_FakeLLM):
            def call(self, *a, require_json=False, **k):
                if require_json:
                    return json.dumps({"goal": "g", "steps": [{"name": "s"}]})
                raise LLMError("server busy")

        assert generate_one(_PROMPT, _Fail()) is None

    def test_auth_error_stops_serial_and_concurrent_runs(self):
        from src.llm.provider import LLMAuthError

        class _NoAuth(_FakeLLM):
            def call(self, *a, **k):
                raise LLMAuthError("401")

        with pytest.raises(LLMAuthError):
            generate_workflows([_prompt(0)], _NoAuth())
        with pytest.raises(LLMAuthError):
            generate_workflows_concurrent([_prompt(i) for i in range(4)], _NoAuth(), max_workers=2)
