"""Ground synthetic REPL solutions in real babashka execution.

A generated solution interleaves ``;; eval: <form>`` with ``;; result: <out>``
lines and ends with a ``;; apply:`` diff. The results are LLM-fabricated. This
module extracts the forms, runs them through the bb_eval harness, rewrites each
``;; result:`` with the real value, and reports how many forms actually ran —
the signal the pipeline uses to keep or drop a solution.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

from .bb_eval import EvalResult, eval_forms

_EVAL_RE = re.compile(r"^\s*;;\s*eval:\s?(.*)$")
_RESULT_RE = re.compile(r"^\s*;;\s*result:\s?(.*)$")
_APPLY_RE = re.compile(r"^\s*;;\s*apply:")
_COMMENT_RE = re.compile(r"^\s*;;\s?(.*)$")


@dataclass
class EvalBlock:
    """An extracted ``;; eval:`` form and the line span of its result block."""
    form: str
    eval_start: int           # index of the ';; eval:' line
    result_start: Optional[int] = None  # index of the ';; result:' line, if any
    result_end: Optional[int] = None    # last line index of the result block


@dataclass
class GroundedSolution:
    """A solution rewritten with real eval results plus execution metrics."""
    solution: str
    total: int
    passed: int
    results: List[EvalResult]

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def all_ok(self) -> bool:
        return self.total > 0 and self.passed == self.total


_DEF_PREFIXES = (
    "def ", "def\n", "defn ", "defn\n", "defn-", "defmacro", "defmulti",
    "defmethod", "defrecord", "deftype", "defprotocol", "defonce",
    "require", "ns ", "ns\n", "import", "use ", "in-ns", "refer",
)


def _is_definition(form: str) -> bool:
    """True if a form just defines/loads something rather than demonstrating it.

    Definitions return a var/nil and so are "ok" trivially; the meaningful
    end-state signal is whether the last *demonstration* (a call producing a
    value) ran correctly.
    """
    inner = form.lstrip().lstrip("(").lstrip()
    return inner.startswith(_DEF_PREFIXES)


@dataclass
class GradedWorkflow(GroundedSolution):
    """A grounded workflow trace plus a tolerant end-state judgement.

    Unlike all_ok, this allows intermediate failures (the write -> fail -> fix
    arc we want to teach) as long as the final demonstration runs correctly.
    """

    @property
    def reaches_correct_end_state(self) -> bool:
        demos = [r for r in self.results if not _is_definition(r.form)]
        return bool(demos) and demos[-1].ok


def _is_marker(line: str) -> bool:
    return bool(_EVAL_RE.match(line) or _RESULT_RE.match(line) or _APPLY_RE.match(line))


def extract_eval_blocks(solution: str) -> List[EvalBlock]:
    """Parse ``;; eval:`` blocks (multi-line forms) and their result spans."""
    lines = solution.splitlines()
    blocks: List[EvalBlock] = []
    i, n = 0, len(lines)

    while i < n:
        m = _EVAL_RE.match(lines[i])
        if not m:
            i += 1
            continue

        eval_start = i
        form_parts = []
        if m.group(1).strip():
            form_parts.append(m.group(1))
        i += 1

        # Continuation lines: ';;' comments that aren't markers extend the form.
        while i < n and not _is_marker(lines[i]):
            cm = _COMMENT_RE.match(lines[i])
            if not cm:
                break
            form_parts.append(cm.group(1))
            i += 1

        result_start = result_end = None
        if i < n and _RESULT_RE.match(lines[i]):
            result_start = i
            result_end = i
            i += 1
            # Result continuation: trailing ';;' lines that aren't markers.
            while i < n and _COMMENT_RE.match(lines[i]) and not _is_marker(lines[i]):
                result_end = i
                i += 1

        blocks.append(EvalBlock(
            form="\n".join(form_parts).strip(),
            eval_start=eval_start,
            result_start=result_start,
            result_end=result_end,
        ))

    return blocks


def _result_text(r: EvalResult) -> str:
    """Render a real eval result as a ``;; result:`` line."""
    if not r.ok:
        return f";; result: ERROR: {r.error}"
    if r.stdout:
        printed = r.stdout.rstrip("\n").replace("\n", " ")
        return f";; result: {r.value}  ;; stdout: {printed}"
    return f";; result: {r.value}"


def ground_solution(
    solution: str,
    blocks: List[EvalBlock],
    results: List[EvalResult],
) -> str:
    """Rewrite each block's result line(s) with the real eval result."""
    lines = solution.splitlines()
    # Map original result-line spans to their replacement text.
    replace_at = {}
    drop = set()
    for block, res in zip(blocks, results):
        new_line = _result_text(res)
        if block.result_start is not None:
            replace_at[block.result_start] = new_line
            for j in range(block.result_start + 1, (block.result_end or block.result_start) + 1):
                drop.add(j)
        else:
            # No result block in the source — append one after the eval form.
            replace_at[block.eval_start] = lines[block.eval_start] + "\n" + new_line

    out = []
    for idx, line in enumerate(lines):
        if idx in drop:
            continue
        out.append(replace_at.get(idx, line))
    return "\n".join(out)


def verify_and_ground(
    solution: str,
    timeout: float = 30.0,
    bb_path: str = "bb",
) -> GroundedSolution:
    """Extract, execute, and ground a solution's REPL forms.

    Returns the rewritten solution and metrics. A solution with no forms is
    reported with all_ok False — there is nothing to vouch for.
    """
    blocks = extract_eval_blocks(solution)
    forms = [b.form for b in blocks if b.form]
    if not forms:
        return GroundedSolution(solution=solution, total=0, passed=0, results=[])

    results = eval_forms(forms, timeout=timeout, bb_path=bb_path)
    grounded = ground_solution(solution, blocks, results)
    passed = sum(1 for r in results if r.ok)
    return GroundedSolution(
        solution=grounded,
        total=len(results),
        passed=passed,
        results=results,
    )


def verify_workflow(
    solution: str,
    timeout: float = 30.0,
    bb_path: str = "bb",
) -> GradedWorkflow:
    """Verify+ground a workflow trace, judging end-state correctness tolerantly.

    Same execution+grounding as verify_and_ground, but returns a GradedWorkflow
    whose reaches_correct_end_state allows intermediate failures (the taught
    error->fix arc) provided the final demonstration runs cleanly.
    """
    blocks = extract_eval_blocks(solution)
    forms = [b.form for b in blocks if b.form]
    if not forms:
        return GradedWorkflow(solution=solution, total=0, passed=0, results=[])

    results = eval_forms(forms, timeout=timeout, bb_path=bb_path)
    grounded = ground_solution(solution, blocks, results)
    passed = sum(1 for r in results if r.ok)
    return GradedWorkflow(
        solution=grounded,
        total=len(results),
        passed=passed,
        results=results,
    )
