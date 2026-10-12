"""Unified-diff integrity checks shared by mining, synthesis and assembly.

A training diff must be a patch ``git apply`` accepts: every hunk's header
counts must match its body. LLM-written diffs routinely get the counts wrong,
so ``fix_hunk_headers`` recomputes them from the body; ``check_patch`` is the
gate every record passes before it ships.
"""

import re
import subprocess
from typing import List, Optional

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
_APPLY_LINE_RE = re.compile(r"^;;\s*apply:\s*$")


def split_lines(text: str) -> List[str]:
    """Split on ``\\n`` only, like git. ``str.splitlines`` also breaks on form
    feeds (Lisp page breaks), ``\\r`` and other separators, which corrupts
    hunk line counts. A final newline doesn't produce an empty last line."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def extract_diff(text: str) -> str:
    """The patch portion of an output: from the first ``diff --git`` on."""
    idx = text.find("diff --git ")
    while idx > 0 and text[idx - 1] != "\n":
        idx = text.find("diff --git ", idx + 1)
    return text[idx:] if idx >= 0 else ""


def _is_body(line: str, nxt: Optional[str]) -> bool:
    if line.startswith("diff --git ") or line.startswith("@@"):
        return False
    # '--- a/..' followed by '+++ ' is the next file's header, not a removal.
    if line.startswith("--- ") and nxt is not None and nxt.startswith("+++ "):
        return False
    return line == "" or line[0] in " +-\\"


def _hunks(lines: List[str]):
    """Yield (header_match, header_index, body_start, body_end) for each hunk.

    The body runs to the next header or the first line that can't be part of a
    hunk; trailing bare blank lines are left out.
    """
    i, n = 0, len(lines)
    while i < n:
        m = _HUNK_RE.match(lines[i])
        if not m:
            i += 1
            continue
        start = j = i + 1
        while j < n and _is_body(lines[j], lines[j + 1] if j + 1 < n else None):
            j += 1
        end = j
        while end > start and lines[end - 1] == "":
            end -= 1
        yield m, i, start, end
        i = j


def _body_counts(body: List[str]):
    old = sum(1 for l in body if l == "" or l[0] in " -")
    new = sum(1 for l in body if l == "" or l[0] in " +")
    return old, new


def _header_count(raw: Optional[str]) -> int:
    return 1 if raw is None else int(raw)


def hunk_count_errors(diff: str) -> List[str]:
    """Hunks whose header line counts disagree with their bodies."""
    lines = split_lines(diff)
    errors = []
    for m, h, start, end in _hunks(lines):
        want_old, want_new = _header_count(m.group(2)), _header_count(m.group(4))
        old, new = _body_counts(lines[start:end])
        if (old, new) != (want_old, want_new):
            errors.append(
                f"hunk at line {h + 1} claims -{want_old}/+{want_new} "
                f"but body has -{old}/+{new}"
            )
    return errors


def fix_hunk_headers(diff: str) -> str:
    """Recompute every hunk header's counts from its body.

    Bare blank lines inside a body become ``' '`` context (what git emits).
    Only the counts change — this can't recover a truncated or invented body.
    """
    lines = split_lines(diff)
    for m, h, start, end in _hunks(lines):
        for k in range(start, end):
            if lines[k] == "":
                lines[k] = " "
        old, new = _body_counts(lines[start:end])
        lines[h] = f"@@ -{m.group(1)},{old} +{m.group(3)},{new} @@{m.group(5)}"
    out = "\n".join(lines)
    if diff.endswith("\n"):
        out += "\n"
    return out


def check_patch(diff: str) -> Optional[str]:
    """None if ``diff`` is a well-formed patch, else the first problem found.

    Checks hunk counts, then has ``git apply --numstat`` parse it (no base tree
    needed), which also catches truncated headers and bodies.
    """
    if not diff.strip():
        return "empty diff"
    if not any(_HUNK_RE.match(l) for l in split_lines(diff)):
        return "no hunks"
    errors = hunk_count_errors(diff)
    if errors:
        return errors[0]
    if not diff.endswith("\n"):
        diff += "\n"
    r = subprocess.run(
        ["git", "apply", "--numstat", "-"],
        input=diff, capture_output=True, text=True, errors="replace",
    )
    if r.returncode != 0:
        return (r.stderr.strip().splitlines() or ["git apply rejected the patch"])[0]
    if not r.stdout.strip():
        return "patch contains no file changes"
    return None


def apply_section_errors(output: str) -> List[str]:
    """A REPL trace must end in ``;; apply:`` followed directly by its diff.

    Plain diff outputs (no ``;; eval:``) need no marker.
    """
    lines = split_lines(output)
    if not any(l.lstrip().startswith(";; eval:") for l in lines):
        return []
    for i, line in enumerate(lines):
        if _APPLY_LINE_RE.match(line):
            rest = [l for l in lines[i + 1:] if l.strip()]
            if rest and rest[0].startswith("diff --git "):
                return []
            return ["';; apply:' is not followed by a diff --git header"]
    return ["trace has no ';; apply:' line"]
