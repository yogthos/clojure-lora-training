"""Generate plan-first, iterative-REPL workflow training examples.

Teaches the full agent loop: a vague user request -> work out the goal -> lay
out files -> sketch an ordered plan -> build the functions one at a time in the
REPL, showing real failures and recovery -> a final diff. Grounded in RPG
(plan-first, dependency-ordered build) and PlanSearch (the NL sketch up front
dominates success).
"""

import json
from dataclasses import dataclass
from typing import List

from ...llm.provider import LLMProvider
from ...shared import _WORKFLOW_SYSTEM_PROMPT
from .prompt_mining import MinedPrompt
from .prompts import PLAN_SYSTEM as _PLAN_SYSTEM, WORKFLOW_SYSTEM as _WORKFLOW_SYSTEM


@dataclass
class WorkflowResult:
    """A generated workflow trace ready for training."""
    user_prompt: str
    project_context: str
    plan: dict
    solution: str
    verified: bool = False
    pass_rate: float = 1.0

    def to_training_example(self) -> dict:
        """LLaMA-Factory record. Planning lives in the output, not the input."""
        ctx = self.project_context.strip()
        return {
            "system": _WORKFLOW_SYSTEM_PROMPT,
            "instruction": self.user_prompt,
            "input": f"Project: {ctx}" if ctx else "",
            "output": self.solution,
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_training_example(), ensure_ascii=False)


def _fallback_plan(prompt: MinedPrompt) -> dict:
    """A minimal plan when the planning call fails to parse."""
    return {
        "goal": prompt.user_prompt,
        "files": [{"path": "src/core.clj", "purpose": "implementation"}],
        "steps": [
            {"name": "solve", "purpose": prompt.user_prompt, "depends_on": []},
        ],
    }


def generate_plan(prompt: MinedPrompt, llm: LLMProvider) -> dict:
    """Plan pass: produce {goal, files, steps} from the user request."""
    user = (
        f"User request: {prompt.user_prompt}\n"
        f"Project context: {prompt.project_context}\n\n"
        f"Plan the implementation."
    )
    try:
        raw = llm.call(
            system_prompt=_PLAN_SYSTEM,
            user_prompt=user,
            temperature=0.3,
            max_tokens=1536,
            require_json=True,
        )
        plan = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(plan, list):
            plan = plan[0] if plan else {}
    except (ValueError, TypeError, IndexError):
        return _fallback_plan(prompt)

    if not isinstance(plan, dict) or not plan.get("steps"):
        return _fallback_plan(prompt)
    plan.setdefault("goal", prompt.user_prompt)
    plan.setdefault("files", [{"path": "src/core.clj", "purpose": "implementation"}])
    return plan


def generate_workflow(
    prompt: MinedPrompt,
    plan: dict,
    llm: LLMProvider,
) -> str:
    """Workflow pass: produce the full iterative REPL trace from the plan."""
    user = (
        f"User request: {prompt.user_prompt}\n"
        f"Project context: {prompt.project_context}\n\n"
        f"Plan:\n{json.dumps(plan, indent=2)}\n\n"
        f"Implement this plan as a complete nREPL-driven development trace, "
        f"building one function at a time and showing at least one "
        f"error-and-recovery."
    )
    try:
        result = llm.call(
            system_prompt=_WORKFLOW_SYSTEM,
            user_prompt=user,
            temperature=0.5,
            max_tokens=8192,
            require_json=False,
        )
        if isinstance(result, dict):
            return result.get("text", result.get("content", str(result)))
        return str(result)
    except Exception:
        return ""


def generate_workflows(
    prompts: List[MinedPrompt],
    llm: LLMProvider,
    max_examples: int = 50,
    verify: bool = False,
    bb_path: str = "bb",
    min_pass_rate: float = 0.6,
) -> List[WorkflowResult]:
    """Generate workflow traces for a list of mined prompts.

    When ``verify`` is set, each trace is executed and grounded; traces are kept
    when they reach a correct end state (see verify_workflow), preserving the
    intermediate failure-and-recovery steps.
    """
    results: List[WorkflowResult] = []
    for prompt in prompts[:max_examples]:
        plan = generate_plan(prompt, llm)
        solution = generate_workflow(prompt, plan, llm)
        if not solution.strip():
            continue

        verified = False
        pass_rate = 1.0
        if verify:
            from .verify import verify_workflow

            graded = verify_workflow(solution, bb_path=bb_path)
            if not graded.reaches_correct_end_state or graded.pass_rate < min_pass_rate:
                continue
            solution = graded.solution
            verified = True
            pass_rate = graded.pass_rate

        results.append(WorkflowResult(
            user_prompt=prompt.user_prompt,
            project_context=prompt.project_context,
            plan=plan,
            solution=solution,
            verified=verified,
            pass_rate=pass_rate,
        ))
    return results
