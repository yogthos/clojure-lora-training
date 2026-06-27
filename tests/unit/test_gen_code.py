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
    _fallback_analysis,
    _fallback_code,
)


class _FakeLLM:
    """Returns a fixed solution for the code pass, stub analysis for JSON."""

    def __init__(self, solution: str):
        self.solution = solution

    def call(self, system_prompt, user_prompt, temperature=None,
             max_tokens=None, require_json=False):
        if require_json:
            return json.dumps({"files_affected": ["src/core.clj"],
                               "repl_exploration": []})
        return self.solution


@pytest.mark.skipif(shutil.which("bb") is None, reason="babashka not installed")
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

    def test_input_context(self):
        result = CodeGenResult(
            instruction="Fix bug in parser",
            analysis={
                "files_affected": ["src/parser.clj", "src/utils.clj"],
                "repl_exploration": ["step 1: require ns", "step 2: test fn"],
            },
            solution=";; solution",
        )
        inp = result._build_input()
        assert "Fix bug in parser" in inp
        assert "src/parser.clj" in inp
        assert "src/utils.clj" in inp
        assert "step 1: require ns" in inp


class TestFallbackAnalysis:
    def test_returns_valid_structure(self):
        analysis = _fallback_analysis("Fix nil check in handler")
        assert "problem" in analysis
        assert "files_affected" in analysis
        assert "approach" in analysis
        assert "clojure_patterns" in analysis
        assert "repl_exploration" in analysis
        assert "incremental_plan" in analysis

    def test_plan_has_steps(self):
        analysis = _fallback_analysis("Refactor middleware")
        plan = analysis["incremental_plan"]
        assert len(plan) >= 3
        assert all("step" in s and "eval" in s for s in plan)


class TestFallbackCode:
    def test_returns_repl_format(self):
        analysis = _fallback_analysis("Fix nil check")
        code = _fallback_code("Fix nil check", analysis)
        assert ";; nREPL session:" in code
        assert ";; eval:" in code
        assert ";; result:" in code
        assert ";; apply:" in code
        assert "diff --git" in code
        assert "@@" in code

    def test_code_is_valid(self):
        analysis = _fallback_analysis("Add feature")
        code = _fallback_code("Add feature", analysis)
        assert validate_solution(code)


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
