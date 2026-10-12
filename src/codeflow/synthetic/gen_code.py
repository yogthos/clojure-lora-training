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
from dataclasses import dataclass
from typing import List, Optional

from ...llm.provider import LLMAuthError, LLMError, LLMProvider


# System prompt for the analysis pass
from .prompts import ANALYSIS_SYSTEM as _ANALYSIS_SYSTEM, CODE_SYSTEM as _CODE_SYSTEM


@dataclass
class CodeGenResult:
    """Result of code generation: analysis + REPL-driven solution."""
    instruction: str
    analysis: dict
    solution: str  # Complete REPL + diff output
    repo_name: str = ""
    verified: bool = False    # solution's forms were executed in a Jolt nREPL session
    pass_rate: float = 1.0    # fraction of forms that executed cleanly
    feature_used: str = ""    # taxonomy feature the task came from (for coreset selection)

    def to_training_example(self) -> dict:
        """Convert to LLaMA-Factory training format."""
        from ...shared import _SYSTEM_PROMPT
        return {
            "system": _SYSTEM_PROMPT,
            "instruction": self.instruction,
            "input": self._build_input(),
            "output": self.solution,
        }

    def _build_input(self) -> str:
        """Build the input context for the training example.

        Only what a user would supply at inference belongs here. The analysis
        (affected files, REPL plan) is generation scaffolding that leaks the
        answer's plan, and the instruction is already its own field. Synthetic
        tasks are self-contained, so there is no project context to add.
        """
        return ""

    def to_jsonl(self) -> str:
        """Serialize as a single JSONL line."""
        return json.dumps(self.to_training_example(), ensure_ascii=False)


def generate_analysis(
    instruction: str,
    llm: LLMProvider,
) -> Optional[dict]:
    """Generate analysis for a coding task.

    The analysis provides context that will be used in the code
    generation pass. It simulates a developer thinking before coding.

    Args:
        instruction: The coding task description.
        llm: LLM provider.

    Returns:
        Dict with analysis fields, or None if the call failed or its output
        wasn't a JSON object.
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
        return analysis if isinstance(analysis, dict) else None
    except LLMAuthError:
        raise
    except (LLMError, ValueError, TypeError, IndexError):
        return None


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
            # A cut-off solution loses its diff or ends mid-hunk: drop it.
            reject_truncated=True,
        )
        if isinstance(result, dict):
            return result.get("text", result.get("content", str(result)))
        return str(result)
    except LLMAuthError:
        raise
    except LLMError:
        return ""


def generate_training_examples(
    tasks: List[dict],
    llm: LLMProvider,
    max_examples: int = 50,
    verify: bool = False,
    jolt_path: str = "jolt",
    min_pass_rate: float = 1.0,
) -> List[CodeGenResult]:
    """Generate complete training examples from a list of tasks.

    Each task goes through: analysis → code generation. When ``verify`` is set,
    the generated REPL forms are executed in a Jolt nREPL session, their ``;; result:``
    lines are rewritten with the real output, and solutions whose forms execute
    cleanly below ``min_pass_rate`` (or have no runnable forms) are dropped.

    Args:
        tasks: List of GeneratedTask dicts (from gen_question).
        llm: LLM provider.
        max_examples: Maximum number of examples to generate.
        verify: Execute and ground each solution's forms before keeping it.
        jolt_path: jolt binary to use for verification.
        min_pass_rate: Minimum fraction of forms that must run cleanly to keep
            a verified solution.

    Returns:
        List of CodeGenResult ready for training.
    """
    results = []

    for task in tasks[:max_examples]:
        instruction = task.get("instruction", "")
        if not instruction:
            continue

        # Pass 1: analysis. A failed call drops the task — nothing canned is
        # substituted, since a generic plan or trace isn't about this task.
        analysis = generate_analysis(instruction, llm)
        if analysis is None:
            continue

        # Pass 2: code with REPL
        solution = generate_code(instruction, analysis, llm)
        if not solution.strip():
            continue

        verified = False
        pass_rate = 1.0
        if verify:
            from .verify import verify_and_ground

            grounded = verify_and_ground(solution, jolt_path=jolt_path)
            # Drop solutions with no runnable forms or too many failures.
            if grounded.total == 0 or grounded.pass_rate < min_pass_rate:
                continue
            solution = grounded.solution
            verified = True
            pass_rate = grounded.pass_rate

        results.append(CodeGenResult(
            instruction=instruction,
            analysis=analysis,
            solution=solution,
            repo_name="synthetic",
            verified=verified,
            pass_rate=pass_rate,
            feature_used=task.get("feature_used", ""),
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
