"""Generate REPL-driven solutions for EpiCoder coding tasks.

Adapted from EpiCoder's gen/gen_code.py. Each task (a self-contained coding
problem from the feature tree) is solved through the live trace loop: the
generator reasons, evaluates one step at a time in a real Jolt nREPL session,
sees the actual result, and finishes with a diff. The trace is kept only if it
reaches a working end state (see live_trace.accept_trace).
"""

import json
from dataclasses import dataclass
from typing import Callable, List, Optional

from ...llm.provider import LLMProvider
from ...shared import _WORKFLOW_SYSTEM_PROMPT
from .jolt_eval import JoltSession
from .live_trace import LiveTrace, accept_trace, run_live_trace


@dataclass
class CodeGenResult:
    """A solved task: the live trace plus what selection needs."""
    trace: LiveTrace
    pass_rate: float
    feature_used: str = ""    # taxonomy feature the task came from (for coreset selection)

    def to_training_example(self) -> dict:
        return {"system": _WORKFLOW_SYSTEM_PROMPT, **self.trace.to_record()}

    def to_jsonl(self) -> str:
        return json.dumps(self.to_training_example(), ensure_ascii=False)


def generate_training_examples(
    tasks: List[dict],
    llm: LLMProvider,
    max_examples: int = 50,
    session_factory: Optional[Callable[[], JoltSession]] = None,
    min_pass_rate: float = 0.6,
) -> List[CodeGenResult]:
    """Solve up to ``max_examples`` tasks (GeneratedTask dicts) as live traces.

    A task whose trace fails or doesn't reach a working end state is dropped;
    nothing canned is substituted. LLMAuthError propagates.
    """
    factory = session_factory or JoltSession
    results = []
    for task in tasks[:max_examples]:
        instruction = (task.get("instruction") or "").strip()
        if not instruction:
            continue
        trace = run_live_trace(instruction, "", llm, session_factory=factory)
        pass_rate = accept_trace(trace, factory, min_pass_rate)
        if pass_rate is None:
            continue
        results.append(CodeGenResult(trace=trace, pass_rate=pass_rate,
                                     feature_used=task.get("feature_used", "")))
    return results
