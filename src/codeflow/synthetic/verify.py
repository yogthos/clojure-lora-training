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
    """An extracted ``;; eval:`` form and the line spans around it."""
    form: str
    eval_start: int           # index of the ';; eval:' line
    form_end: int = -1        # last line index of the form
    result_start: Optional[int] = None  # index of the ';; result:' line, if any
    result_end: Optional[int] = None    # last line index of the result block
    malformed: bool = False   # empty or unbalanced form — can't be run as written


@dataclass
class GroundedSolution:
    """A solution rewritten with real eval results plus execution metrics."""
    solution: str
    total: int
    passed: int
    results: List[EvalResult]
    malformed: int = 0   # eval blocks that couldn't be parsed into a runnable form

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def all_ok(self) -> bool:
        return self.total > 0 and self.passed == self.total and not self.malformed


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
        if self.malformed:
            return False
        demos = [r for r in self.results if not _is_definition(r.form)]
        return bool(demos) and demos[-1].ok


def _is_marker(line: str) -> bool:
    return bool(_EVAL_RE.match(line) or _RESULT_RE.match(line) or _APPLY_RE.match(line))


# A fabricated multi-line result continues on ';;' lines indented past the
# marker; narrative prose after a result is written as ';; Text'.
_RESULT_CONT_RE = re.compile(r"^\s*;;\s{2,}\S")
_MAX_RESULT_CONT = 40


def _balance(text: str):
    """Net bracket depth of Clojure text and whether it ends inside a string.

    Skips string contents, character literals (``\\(``) and ``;`` comments so
    brackets inside them don't count.
    """
    depth, in_str, i, n = 0, False, 0, len(text)
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "\\":
            i += 1  # character literal: skip the escaped char
        elif c == ";":
            nl = text.find("\n", i)
            i = n if nl < 0 else nl
            continue
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        i += 1
    return depth, in_str


def _form_complete(text: str) -> bool:
    depth, in_str = _balance(text)
    return bool(text.strip()) and depth <= 0 and not in_str


def extract_eval_blocks(solution: str) -> List[EvalBlock]:
    """Parse ``;; eval:`` blocks and their result spans.

    A form runs from the eval marker until its brackets balance, continuing on
    ``;;``-prefixed or raw code lines. Anything after a balanced form or result
    that isn't a marker is narrative and left alone. Each eval owns exactly one
    result: a multi-line result continues only on indented ``;;`` lines while
    its brackets are open, and repeated ``;; result:`` lines are absorbed so no
    fabricated result survives grounding.
    """
    lines = solution.splitlines()
    blocks: List[EvalBlock] = []
    i, n = 0, len(lines)

    while i < n:
        m = _EVAL_RE.match(lines[i])
        if not m:
            i += 1
            continue
        eval_start = i
        form_parts = [m.group(1)] if m.group(1).strip() else []
        i += 1

        while i < n and not _form_complete("\n".join(form_parts)):
            line = lines[i]
            if _is_marker(line):
                break
            cm = _COMMENT_RE.match(line)
            if cm:
                form_parts.append(cm.group(1))
            elif line.strip() or form_parts:
                form_parts.append(line)
            i += 1
        form = "\n".join(form_parts).strip()
        depth, in_str = _balance(form)
        block = EvalBlock(
            form=form,
            eval_start=eval_start,
            form_end=i - 1,
            malformed=not form or depth > 0 or in_str,
        )

        if i < n and _RESULT_RE.match(lines[i]):
            block.result_start = block.result_end = i
            text = _RESULT_RE.match(lines[i]).group(1)
            i += 1
            cont = 0
            while i < n:
                if _RESULT_RE.match(lines[i]):
                    text = _RESULT_RE.match(lines[i]).group(1)
                elif (cont < _MAX_RESULT_CONT and _balance(text)[0] > 0
                      and _RESULT_CONT_RE.match(lines[i]) and not _is_marker(lines[i])):
                    text += "\n" + _COMMENT_RE.match(lines[i]).group(1)
                    cont += 1
                else:
                    break
                block.result_end = i
                i += 1

        blocks.append(block)

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
    """Rewrite each runnable block's result line(s) with the real eval result.

    ``results`` pairs with the non-malformed blocks, in order.
    """
    lines = solution.splitlines()
    # Map original result-line spans to their replacement text.
    replace_at = {}
    drop = set()
    runnable = [b for b in blocks if not b.malformed]
    for block, res in zip(runnable, results):
        new_line = _result_text(res)
        if block.result_start is not None:
            replace_at[block.result_start] = new_line
            for j in range(block.result_start + 1, block.result_end + 1):
                drop.add(j)
        else:
            # No result block in the source — append one after the whole form.
            replace_at[block.form_end] = lines[block.form_end] + "\n" + new_line

    out = []
    for idx, line in enumerate(lines):
        if idx in drop:
            continue
        out.append(replace_at.get(idx, line))
    return "\n".join(out)


def _ground(solution, timeout, bb_path, cls):
    blocks = extract_eval_blocks(solution)
    malformed = sum(1 for b in blocks if b.malformed)
    forms = [b.form for b in blocks if not b.malformed]
    if not forms:
        return cls(solution=solution, total=0, passed=0, results=[], malformed=malformed)

    results = eval_forms(forms, timeout=timeout, bb_path=bb_path)
    grounded = ground_solution(solution, blocks, results)
    passed = sum(1 for r in results if r.ok)
    return cls(
        solution=grounded,
        total=len(results),
        passed=passed,
        results=results,
        malformed=malformed,
    )


def verify_and_ground(
    solution: str,
    timeout: float = 30.0,
    bb_path: str = "bb",
) -> GroundedSolution:
    """Extract, execute, and ground a solution's REPL forms.

    Returns the rewritten solution and metrics. A solution with no forms is
    reported with all_ok False — there is nothing to vouch for.
    """
    return _ground(solution, timeout, bb_path, GroundedSolution)


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
    return _ground(solution, timeout, bb_path, GradedWorkflow)
