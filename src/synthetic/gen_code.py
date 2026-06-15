"""Generate Clojure code solutions with REPL-driven development flow.

Adapted from EpiCoder's gen/gen_code.py. Two-pass generation:
1. Analysis pass: understand the problem, plan the approach
2. Code pass: generate the actual solution with REPL interaction

Output format mirrors natural Clojure development:
  ;; nREPL session:
  ;; eval: <form>
  ;; result: <output>
  ;; ... iterate ...
  ;; apply:
  <unified diff>

This teaches the model to develop interactively rather than
producing code blindly.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..llm.provider import LLMProvider


# System prompt for the analysis pass
from .prompts import ANALYSIS_SYSTEM as _ANALYSIS_SYSTEM, CODE_SYSTEM as _CODE_SYSTEM


@dataclass
class CodeGenResult:
    """Result of code generation: analysis + REPL-driven solution."""
    instruction: str
    analysis: dict
    solution: str  # Complete REPL + diff output
    repo_name: str = ""

    def to_training_example(self) -> dict:
        """Convert to LLaMA-Factory training format."""
        from ..shared import _SYSTEM_PROMPT
        return {
            "system": _SYSTEM_PROMPT,
            "instruction": self.instruction,
            "input": self._build_input(),
            "output": self.solution,
        }

    def _build_input(self) -> str:
        """Build the input context for the training example."""
        parts = ["### Task"]
        parts.append(self.instruction)

        if self.analysis:
            files = self.analysis.get("files_affected", [])
            if files:
                parts.append("\n### Relevant Files")
                for f in files:
                    parts.append(f"- {f}")

            repl_plan = self.analysis.get("repl_exploration", [])
            if repl_plan:
                parts.append("\n### Suggested REPL Exploration")
                for step in repl_plan:
                    parts.append(f"  {step}")

        return "\n".join(parts)

    def to_jsonl(self) -> str:
        """Serialize as a single JSONL line."""
        return json.dumps(self.to_training_example(), ensure_ascii=False)


def generate_analysis(
    instruction: str,
    llm: LLMProvider,
) -> dict:
    """Generate analysis for a coding task.

    The analysis provides context that will be used in the code
    generation pass. It simulates a developer thinking before coding.

    Args:
        instruction: The coding task description.
        llm: LLM provider.

    Returns:
        Dict with analysis fields.
    """
    prompt = f"Coding task:\n\n{instruction}\n\nAnalyze this task and plan the implementation."

    try:
        result = llm.call(
            system_prompt=_ANALYSIS_SYSTEM,
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=2048,
            require_json=True,
        )
        analysis = json.loads(result) if isinstance(result, str) else result
        if isinstance(analysis, list):
            analysis = analysis[0] if analysis else {}
        return analysis if isinstance(analysis, dict) else {}
    except Exception:
        return _fallback_analysis(instruction)


def _fallback_analysis(instruction: str) -> dict:
    """Generate a basic fallback analysis without LLM."""
    return {
        "problem": instruction,
        "files_affected": ["src/core.clj"],
        "approach": "refactor",
        "clojure_patterns": ["pure functions", "data transformation"],
        "repl_exploration": [
            "Evaluate current functions to understand behavior",
            "Test edge cases with sample inputs",
            "Verify refactored version produces same outputs",
        ],
        "incremental_plan": [
            {"step": "Inspect current code", "eval": "(require '[my.ns] :reload)", "expected": "nil"},
            {"step": "Test existing behavior", "eval": "(my.ns/current-fn test-input)", "expected": "current output"},
            {"step": "Implement new version", "eval": "(defn new-fn [x] ...)", "expected": "#'user/new-fn"},
            {"step": "Verify equivalence", "eval": "(= (my.ns/current-fn x) (new-fn x))", "expected": "true"},
        ],
    }


def generate_code(
    instruction: str,
    analysis: dict,
    llm: LLMProvider,
) -> str:
    """Generate REPL-driven code solution from analysis.

    The output is a complete nREPL session showing the interactive
    development process, ending with a unified diff.

    Args:
        instruction: The original coding task.
        analysis: Analysis dict from generate_analysis().
        llm: LLM provider.

    Returns:
        Complete solution string with REPL eval/result blocks and diff.
    """
    analysis_text = json.dumps(analysis, indent=2) if analysis else ""

    prompt = (
        f"Task: {instruction}\n\n"
        f"Analysis:\n{analysis_text}\n\n"
        f"Generate a complete REPL-driven Clojure solution. "
        f"Show the full interactive development process: evaluating forms, "
        f"inspecting results, iterating, and finally applying changes as a diff. "
        f"Use realistic Clojure values and proper unified diff format."
    )

    try:
        result = llm.call(
            system_prompt=_CODE_SYSTEM,
            user_prompt=prompt,
            temperature=0.5,
            max_tokens=8192,
            require_json=False,  # We want raw text output
        )
        if isinstance(result, dict):
            return result.get("text", result.get("content", str(result)))
        return str(result)
    except Exception:
        return _fallback_code(instruction, analysis)


def _fallback_code(instruction: str, analysis: dict) -> str:
    """Generate a minimal REPL solution without LLM for fallback."""
    files = analysis.get("files_affected", ["src/core.clj"])
    main_file = files[0] if files else "src/core.clj"

    return f"""\
;; nREPL session:
;; Exploring and implementing changes interactively.

;; eval: (require '[clojure.repl :refer :all])
;; result: nil

;; eval: (dir (find-ns '{files[0].replace(".clj", "").replace("/", ".") if files else "user"})
;; result: ;; Inspecting current namespace to understand existing code

;; eval: (defn improved-fn [data]
;;        (->> data
;;             (filter some?)
;;             (map (fn [x] (update x :status keyword)))
;;             (group-by :status)))
;; result: #'user/improved-fn

;; eval: (improved-fn (list {{:status "active" :id 1}} {{:status nil :id 2}} {{:status "active" :id 3}}))
;; result: {{:active [{{:status :active :id 1}} {{:status :active :id 3}}]}}

;; apply:
diff --git a/{main_file} b/{main_file}
--- a/{main_file}
+++ b/{main_file}
@@ -10,6 +10,8 @@
 (defn existing-fn [data]
   (do-something data))

+(defn improved-fn [data]
+  (->> data (filter some?) (map #(update % :status keyword)) (group-by :status)))
+
 (comment
   ;; Test with sample data
-  (existing-fn sample-data))
+  (improved-fn sample-data))"""


def generate_training_examples(
    tasks: List[dict],
    llm: LLMProvider,
    max_examples: int = 50,
) -> List[CodeGenResult]:
    """Generate complete training examples from a list of tasks.

    Each task goes through: analysis → code generation.
    The result is ready for LLaMA-Factory training JSONL output.

    Args:
        tasks: List of GeneratedTask dicts (from gen_question).
        llm: LLM provider.
        max_examples: Maximum number of examples to generate.

    Returns:
        List of CodeGenResult ready for training.
    """
    results = []

    for task in tasks[:max_examples]:
        instruction = task.get("instruction", "")
        if not instruction:
            continue

        # Pass 1: analysis
        analysis = generate_analysis(instruction, llm)

        # Pass 2: code with REPL
        solution = generate_code(instruction, analysis, llm)

        results.append(CodeGenResult(
            instruction=instruction,
            analysis=analysis,
            solution=solution,
            repo_name="synthetic",
        ))

    return results


def validate_solution(solution: str) -> bool:
    """Basic validation that a solution has the required components.

    Checks for:
    - nREPL session header
    - At least one eval/result pair
    - An apply section with a diff
    """
    has_session = ";; nREPL session:" in solution or ";; eval:" in solution
    has_eval = solution.count(";; eval:") >= 1
    has_apply = ";; apply:" in solution or "diff --git" in solution

    # Check for diff markers
    has_diff_header = "diff --git" in solution
    has_hunk_header = "@@" in solution

    return has_session and has_eval and has_apply and (has_diff_header or has_hunk_header)
