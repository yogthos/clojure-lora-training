"""Tests for synthetic data generation: gen_code.py"""

import json
import shutil
import pytest
from src.codeflow.synthetic.gen_code import (
    CodeGenResult,
    generate_analysis,
    generate_code,
    generate_training_examples,
    validate_solution,
)


class _FakeLLM:
    """Returns a fixed solution for the code pass, stub analysis for JSON."""

    def __init__(self, solution: str):
        self.solution = solution

    def call(self, system_prompt, user_prompt, temperature=None,
             max_tokens=None, require_json=False, reject_truncated=False):
        self.reject_truncated = reject_truncated
        if require_json:
            return json.dumps({"files_affected": ["src/core.clj"],
                               "repl_exploration": []})
        return self.solution


@pytest.mark.skipif(shutil.which("jolt") is None, reason="jolt not installed")
class TestExecutionVerification:
    _GOOD = (
        ";; nREPL session:\n"
        ";; eval: (defn sq [x] (* x x))\n;; result: WRONG\n"
        ";; eval: (sq 5)\n;; result: WRONG\n"
        ";; apply:\ndiff --git a/core.clj b/core.clj\n@@ -1 +1 @@\n+x\n"
    )
    _BAD = (
        ";; eval: (require 'nope.phantom.core)\n;; result: nil\n"
        ";; apply:\ndiff --git a/x b/x\n@@ -1 +1 @@\n+x\n"
    )

    def test_verified_solution_is_grounded(self):
        results = generate_training_examples(
            [{"instruction": "square"}], _FakeLLM(self._GOOD),
            verify=True, min_pass_rate=1.0,
        )
        assert len(results) == 1
        assert results[0].verified
        assert results[0].pass_rate == 1.0
        assert "25" in results[0].solution        # real result, grounded
        assert "WRONG" not in results[0].solution

    def test_low_pass_rate_is_filtered_out(self):
        results = generate_training_examples(
            [{"instruction": "phantom"}], _FakeLLM(self._BAD),
            verify=True, min_pass_rate=0.9,
        )
        assert results == []

    def test_verify_false_keeps_solution_unchanged(self):
        results = generate_training_examples(
            [{"instruction": "phantom"}], _FakeLLM(self._BAD),
            verify=False,
        )
        assert len(results) == 1
        assert not results[0].verified


class TestCodeGenResult:
    def test_to_training_example(self):
        result = CodeGenResult(
            instruction="Fix nil handling",
            analysis={"problem": "nil handling", "files_affected": ["src/core.clj"]},
            solution=""";; nREPL session:
;; eval: (require '[app.core] :reload)
;; result: nil
;; eval: (app.core/parse {:a 1})
;; result: {:a 1}
;; apply:
diff --git a/src/core.clj b/src/core.clj
--- a/src/core.clj
+++ b/src/core.clj
@@ -1 +1 @@
-(defn parse [x] x)
+(defn parse [x] (if x (assoc x :parsed true) {}))""",
        )
        example = result.to_training_example()
        assert "system" in example
        assert "instruction" in example
        assert "input" in example
        assert "output" in example
        assert example["instruction"] == "Fix nil handling"
        assert ";; eval:" in example["output"]
        assert "diff --git" in example["output"]

    def test_to_jsonl(self):
        result = CodeGenResult(
            instruction="Add validation",
            analysis={},
            solution=";; nREPL session:\n;; eval: (+ 1 2)\n;; result: 3\n;; apply:\ndiff --git a/f.clj b/f.clj",
        )
        line = result.to_jsonl()
        parsed = json.loads(line)
        assert parsed["instruction"] == "Add validation"

    def test_input_holds_nothing_unavailable_at_inference(self):
        # The analysis (files, REPL plan) is the model's own scratch work; at
        # inference the user supplies only the instruction, so neither the
        # plan nor a copy of the instruction belongs in the input.
        result = CodeGenResult(
            instruction="Fix bug in parser",
            analysis={
                "files_affected": ["src/parser.clj", "src/utils.clj"],
                "repl_exploration": ["step 1: require ns", "step 2: test fn"],
            },
            solution=";; solution",
        )
        example = result.to_training_example()
        assert example["input"] == ""
        assert example["instruction"] == "Fix bug in parser"


class _FailingLLM:
    def __init__(self, exc):
        self.exc = exc

    def call(self, *a, **k):
        raise self.exc


class TestLLMFailuresDropExamples:
    """A failed LLM call drops the task; nothing canned is substituted."""

    def test_failed_calls_produce_no_example(self):
        from src.codeflow.synthetic.gen_code import generate_training_examples
        from src.llm.provider import LLMError
        out = generate_training_examples(
            [{"instruction": "Fix nil check in handler"}], _FailingLLM(LLMError("503")))
        assert out == []

    def test_auth_error_stops_the_run(self):
        from src.codeflow.synthetic.gen_code import generate_training_examples
        from src.llm.provider import LLMAuthError
        with pytest.raises(LLMAuthError):
            generate_training_examples(
                [{"instruction": "Fix nil check in handler"}], _FailingLLM(LLMAuthError("401")))


class TestValidateSolution:
    def test_valid_solution(self):
        solution = """;; nREPL session:
;; eval: (+ 1 2)
;; result: 3
;; apply:
diff --git a/core.clj b/core.clj
--- a/core.clj
+++ b/core.clj
@@ -1,1 +1,2 @@
 (ns core)
+(def x 1)"""
        assert validate_solution(solution)

    def test_missing_session_header(self):
        solution = """diff --git a/core.clj b/core.clj
--- a/core.clj
+++ b/core.clj
@@ -1,1 +1,2 @@"""
        assert not validate_solution(solution)

    def test_missing_eval(self):
        solution = """;; nREPL session:
;; apply:
diff --git a/core.clj b/core.clj
--- a/core.clj
+++ b/core.clj
@@ -1,1 +1,2 @@"""
        assert not validate_solution(solution)

    def test_missing_diff(self):
        solution = """;; nREPL session:
;; eval: (+ 1 2)
;; result: 3"""
        assert not validate_solution(solution)

    def test_minimal_valid(self):
        solution = ";; nREPL session:\n;; eval: x\n;; result: y\n;; apply:\ndiff --git @@"
        assert validate_solution(solution)


class TestCodeGeneration:
    _SOLUTION = (
        ";; nREPL session:\n;; eval: (+ 1 2)\n;; result: 3\n"
        ";; apply:\ndiff --git a/f.clj b/f.clj\n@@ -0,0 +1 @@\n+x\n"
    )

    def test_code_pass_rejects_truncation(self):
        llm = _FakeLLM(self._SOLUTION)
        generate_code("task", {"problem": "p"}, llm)
        assert llm.reject_truncated is True

    def test_truncated_solution_is_dropped(self):
        from src.llm.provider import LLMTruncatedError

        class _Truncating(_FakeLLM):
            def call(self, *a, require_json=False, **k):
                if require_json:
                    return json.dumps({"problem": "p"})
                raise LLMTruncatedError("cut off")

        out = generate_training_examples([{"instruction": "x"}], _Truncating(""))
        assert out == []

    def test_feature_used_carried_through(self):
        out = generate_training_examples(
            [{"instruction": "x", "feature_used": "defmacro"}], _FakeLLM(self._SOLUTION))
        assert out[0].feature_used == "defmacro"


class TestCodePromptConsistency:
    def test_no_invented_before_state(self):
        from src.codeflow.synthetic.prompts import CODE_SYSTEM
        assert "sensible file paths" not in CODE_SYSTEM
        assert "/dev/null" in CODE_SYSTEM
