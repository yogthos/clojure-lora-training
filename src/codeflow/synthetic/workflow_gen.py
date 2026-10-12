"""Generate plan-first, iterative-REPL workflow training examples.

Teaches the full agent loop: a vague user request -> work out the goal -> lay
out files -> sketch an ordered plan -> build the functions one at a time in the
REPL, showing real failures and recovery -> a final diff. Grounded in RPG
(plan-first, dependency-ordered build) and PlanSearch (the NL sketch up front
dominates success).
"""

import json
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from ...llm.provider import LLMAuthError, LLMError, LLMProvider
from ..diffs import apply_section_errors, check_patch, extract_diff, fix_hunk_headers
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
    except (ValueError, TypeError, IndexError, LLMError) as e:
        if isinstance(e, LLMAuthError):
            raise
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
            reject_truncated=True,  # a cut-off trace has no (or half a) diff
        )
        if isinstance(result, dict):
            return result.get("text", result.get("content", str(result)))
        return str(result)
    except LLMAuthError:
        raise
    except LLMError:
        return ""  # retries exhausted or truncated: drop this sample


# A short unit (8-200 chars) repeated back-to-back 10+ times: a degenerate
# sampling loop, e.g. "'zip/xml-zip 'zip/seq-zip 'zip/xml-zip ...".
_REPETITION_RE = re.compile(r"(.{8,200}?)\1{9,}", re.DOTALL)


def _has_repetition_loop(text: str) -> bool:
    """True if the model's own text degenerates into a repeating loop.

    ``;; result:`` lines are skipped (they're replaced by, or are, real REPL
    output, which can legitimately repeat), as are units with under 4 distinct
    characters (whitespace padding, ``"xxxx"`` test strings).
    """
    own = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith(";; result:"))
    return any(len(set(m.group(1))) >= 4 for m in _REPETITION_RE.finditer(own))


def _repair_diff(solution: str) -> Optional[str]:
    """Recompute the trailing diff's hunk counts; None if it's still not a patch.

    LLMs get @@ counts wrong far more often than the hunk bodies, so the counts
    are rebuilt from the bodies before the patch is checked.
    """
    diff = extract_diff(solution)
    if not diff:
        return None
    fixed = fix_hunk_headers(diff)
    if check_patch(fixed) is not None:
        return None
    return solution[: len(solution) - len(diff)] + fixed


def generate_one(
    prompt: MinedPrompt,
    llm: LLMProvider,
    verify: bool = False,
    jolt_path: str = "jolt",
    min_pass_rate: float = 0.6,
) -> Optional[WorkflowResult]:
    """Produce one workflow trace for a prompt, or None if it is dropped.

    Plan -> workflow -> (optional) execute+ground. With ``verify`` set, the
    trace is kept only when it reaches a correct end state (see verify_workflow),
    preserving the intermediate failure-and-recovery steps. This is the unit of
    work; it is self-contained so it can run concurrently across prompts.
    """
    plan = generate_plan(prompt, llm)
    solution = generate_workflow(prompt, plan, llm)
    if not solution.strip() or apply_section_errors(solution) or _has_repetition_loop(solution):
        return None

    verified = False
    pass_rate = 1.0
    if verify:
        from .verify import verify_workflow

        graded = verify_workflow(solution, jolt_path=jolt_path)
        if not graded.reaches_correct_end_state or graded.pass_rate < min_pass_rate:
            return None
        solution = graded.solution
        verified = True
        pass_rate = graded.pass_rate

    solution = _repair_diff(solution)
    if solution is None:
        return None

    return WorkflowResult(
        user_prompt=prompt.user_prompt,
        project_context=prompt.project_context,
        plan=plan,
        solution=solution,
        verified=verified,
        pass_rate=pass_rate,
    )


def generate_workflows(
    prompts: List[MinedPrompt],
    llm: LLMProvider,
    max_examples: int = 50,
    verify: bool = False,
    jolt_path: str = "jolt",
    min_pass_rate: float = 0.6,
) -> List[WorkflowResult]:
    """Generate workflow traces for a list of mined prompts (serially)."""
    results: List[WorkflowResult] = []
    for prompt in prompts[:max_examples]:
        r = generate_one(prompt, llm, verify, jolt_path, min_pass_rate)
        if r is not None:
            results.append(r)
    return results


def generate_workflows_concurrent(
    prompts: List[MinedPrompt],
    llm: LLMProvider,
    max_workers: int = 8,
    verify: bool = False,
    jolt_path: str = "jolt",
    min_pass_rate: float = 0.6,
    on_result: Optional[Callable[[WorkflowResult], None]] = None,
    on_progress: Optional[Callable[[int, int, int], None]] = None,
    on_attempt: Optional[Callable[[MinedPrompt, Optional[WorkflowResult]], None]] = None,
) -> List[WorkflowResult]:
    """Generate workflow traces concurrently across prompts.

    The work is I/O-bound (LLM API calls plus jolt subprocesses), so a
    thread pool gives real speedup. ``on_result`` is invoked from the calling
    thread for each kept trace as it completes — use it to stream results to
    disk incrementally. ``on_progress(done, total, kept)`` fires once per
    completed prompt for progress reporting. A worker that raises is treated as
    a dropped sample (None), never aborting the run — except LLMAuthError,
    which stops it. ``on_attempt(prompt, result_or_None)`` fires once per
    finished prompt, kept or dropped, after ``on_result`` — use it to record
    progress for resuming.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[WorkflowResult] = []
    total = len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(generate_one, p, llm, verify, jolt_path, min_pass_rate): p
            for p in prompts
        }
        for done, fut in enumerate(as_completed(futures), 1):
            try:
                r = fut.result()
            except LLMAuthError:
                for f in futures:
                    f.cancel()
                raise
            except Exception:
                r = None
            if r is not None:
                results.append(r)
                if on_result is not None:
                    on_result(r)
            if on_attempt is not None:
                on_attempt(futures[fut], r)
            if on_progress is not None:
                on_progress(done, total, len(results))
    return results
