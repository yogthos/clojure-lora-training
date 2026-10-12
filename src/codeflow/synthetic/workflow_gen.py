"""Generate plan-first, iterative-REPL workflow training examples.

Teaches the full agent loop: a vague user request -> work out the goal -> lay
out files -> sketch an ordered plan -> build the functions one at a time in the
REPL, meeting real failures and recovering -> a final diff. Grounded in RPG
(plan-first, dependency-ordered build) and PlanSearch (the NL sketch up front
dominates success).

Each trace is generated live (see live_trace): every eval runs in a real Jolt
nREPL session before the next step is written.
"""

import json
from dataclasses import dataclass
from typing import Callable, List, Optional

from ...llm.provider import LLMAuthError, LLMProvider
from ...shared import _WORKFLOW_SYSTEM_PROMPT
from .jolt_eval import JoltSession
from .live_trace import LiveTrace, accept_trace, run_live_trace
from .prompt_mining import MinedPrompt


@dataclass
class WorkflowResult:
    """A generated workflow trace ready for training."""
    trace: LiveTrace
    pass_rate: float

    def to_training_example(self) -> dict:
        return {"system": _WORKFLOW_SYSTEM_PROMPT, **self.trace.to_record()}

    def to_jsonl(self) -> str:
        return json.dumps(self.to_training_example(), ensure_ascii=False)


def generate_one(
    prompt: MinedPrompt,
    llm: LLMProvider,
    session_factory: Optional[Callable[[], JoltSession]] = None,
    min_pass_rate: float = 0.6,
) -> Optional[WorkflowResult]:
    """Produce one workflow trace for a prompt, or None if it is dropped.

    Kept only if the trace reaches a working end state, at least
    ``min_pass_rate`` of its evals ran cleanly, and the files its diff creates
    load. Self-contained so it can run concurrently across prompts.
    """
    factory = session_factory or JoltSession
    trace = run_live_trace(prompt.user_prompt, prompt.project_context, llm,
                           repo=prompt.source_repo, session_factory=factory)
    pass_rate = accept_trace(trace, factory, min_pass_rate)
    if pass_rate is None:
        return None
    return WorkflowResult(trace=trace, pass_rate=pass_rate)


def generate_workflows(
    prompts: List[MinedPrompt],
    llm: LLMProvider,
    max_examples: int = 50,
    session_factory: Optional[Callable[[], JoltSession]] = None,
    min_pass_rate: float = 0.6,
) -> List[WorkflowResult]:
    """Generate workflow traces for a list of mined prompts (serially)."""
    results: List[WorkflowResult] = []
    for prompt in prompts[:max_examples]:
        r = generate_one(prompt, llm, session_factory, min_pass_rate)
        if r is not None:
            results.append(r)
    return results


def generate_workflows_concurrent(
    prompts: List[MinedPrompt],
    llm: LLMProvider,
    max_workers: int = 8,
    session_factory: Optional[Callable[[], JoltSession]] = None,
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
            pool.submit(generate_one, p, llm, session_factory, min_pass_rate): p
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
